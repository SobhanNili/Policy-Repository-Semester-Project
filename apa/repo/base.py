"""
Base class for policy repository implementations.
"""
from abc import ABC, abstractmethod
import jax.numpy as jnp
from apa.utils.network import make_network, init_random_policy
from apa.utils.ppo import make_ppo_step, perform_rollout
from kinetix.render.renderer_pixels import make_render_pixels_rl
from apa.utils.env import make_env_from_config, duplicate_env_states
import jax.random as jr
import jax.numpy as jnp
import jax
from typing import NamedTuple
from flax.training.train_state import TrainState
import numpy as np
from tqdm.auto import tqdm

def get_pytree_element_by_index(pytree, i):
    return jax.tree.map(lambda x: x[i], pytree)

def get_pytree_batch_size(pytree):
    """Returns the leading batch size of a pytree."""
    leaf = jax.tree_util.tree_leaves(pytree)[0]
    return leaf.shape[0]

def _random_argmax(rng: jax.Array, scores: jax.Array) -> jax.Array:
    """Stochastic argmax: break ties uniformly at random.

    Args:
        rng: JAX PRNG key.
        scores: 1D array of scores.

    Returns:
        Index of the maximum score (ties resolved uniformly at random).
    """
    n = scores.shape[0]
    perm = jr.permutation(rng, n) # permutation serves as tie-breaking noise without affecting the argmax outcome
    return perm[jnp.argmax(scores[perm])]

class PolicyState(NamedTuple):
    """Data structure to hold the policy network parameters and RNN hidden states for multiple policies in the repository.
    Each member is a pytree with leading shape (P, ...), where P = num_policies."""
    train_state: TrainState # parameters and optimzer state for policy nets
    hstate: jax.Array # RNN hidden state. As the non-RNN variant of transformer policy is used, this serves as a dummy for API compatibility.

class RepoStepLog(NamedTuple):
    # FIXME
    """Data structure for logging information about a single repository step."""
    candidate_env_indices: jnp.ndarray   # (E_subset,)
    affinity_scores: jnp.ndarray         # (P, E_subset)
    active_mask: jnp.ndarray             # (E_subset, P) binary
    solve_rate: jnp.ndarray              # (P, E_subset) post-training, unmasked
    avg_return: jnp.ndarray              # (P, E_subset) post-training, unmasked
    pre_frozen_solve_rate: jnp.ndarray   # (P, E_subset) pre-training frozen
    pre_frozen_avg_return: jnp.ndarray   # (P, E_subset)
    pre_finetune_solve_rate: jnp.ndarray # (P, E_subset) pre-training finetuned
    pre_finetune_avg_return: jnp.ndarray # (P, E_subset)


class PolicyRepo(ABC):
        """
        Base class for implemetation of policy repository.
        Trains the policies on their assigned environments (based on the subclass-defined affinity scoring logic) and evaluates them on the same set of environments.
        Each subclass must implement its own affinity scoring logic in ``compute_affinity``.
        """

        def __init__(self,config,train_env_reset_states,rng) -> None:
            """Initializes the policy repository.

            Args:
                config: Configuration dictionary (in kinetix conventions)
                train_env_reset_states: Initial states for training environments.
                rng: JAX random key.
            """
            env, env_params, static_env_params = make_env_from_config(config)
            self.pixel_renderer = make_render_pixels_rl(env_params, static_env_params)
            self.network = make_network(config,env,env_params,static_env_params)

            self.env = env
            self.env_params = env_params
            self.static_env_params = static_env_params
            self.config = config
            self.rng = rng
            self.num_policies = int(config['num_policies'])
            self.num_distinct_envs = int(config['num_train_envs']) # number of unique envs in the training pool (E)
            self.num_copies_per_env = int(config['num_copies_per_env']) # number of copies to create of each env in the training pool (C)
            self.candidate_pool_size = int(config['candidate_pool_size']) # number of candidate envs to sample per repo step (E_subset)
            self.eval_rollout_steps = int(config['eval_rollout_steps']) # number of rollout steps for evaluation
            self.num_ppo_updates_per_repo_step = int(config['num_updates']) # number of PPO updates to perform per repository step on envs assigned to each policy in parallel

            # Validiation: total env batch per policy must be divisible by num_minibatches
            num_minibatches = int(config['num_minibatches'])
            batch_per_policy = self.candidate_pool_size * self.num_copies_per_env
            if batch_per_policy % num_minibatches != 0:
                raise ValueError(
                    f"candidate_pool_size ({self.candidate_pool_size}) * num_copies_per_env "
                    f"({self.num_copies_per_env}) = {batch_per_policy} "
                    f"must be divisible by num_minibatches ({num_minibatches})"
                )

            self.train_env_reset_states = train_env_reset_states
            assert get_pytree_batch_size(self.train_env_reset_states) == self.num_distinct_envs

            # Duplicate the training env states for parallel rollouts, increasing the the PPO training stability.
            # Resulting pytree shape: (E, C, ...) where E = num_distinct_envs and C = num_copies_per_env.
            self.batched_train_env_reset_states = duplicate_env_states(
                train_env_reset_states, self.num_copies_per_env
            )

            self.policy_repo = self.init_policies()
            self.repo_step_counter:int = 0
            self.ppo_step = make_ppo_step(config, env, env_params)
            self.step_logs: list[RepoStepLog] = []

        def init_policies(self):
            """Random initialization of all policies in the repository
                Results expect E_subset * C envs per policy as input."""
            self.rng, sub_rng = jr.split(self.rng)

            n_concurrent_envs = self.candidate_pool_size * self.num_copies_per_env

            def make_init_random_policy(rng):
                return init_random_policy(self.config, self.network, self.env, self.env_params, n_concurrent_envs, rng)
            
            network_rngs = jr.split(sub_rng, self.num_policies)
            train_state, hstate = jax.vmap(make_init_random_policy)(network_rngs)
            assert get_pytree_batch_size(train_state) == self.num_policies

            return PolicyState(train_state,hstate)

        def train(self, batched_env_states, active_mask):
            """Trains all policies in parallel with PPO on the same set of provided environments, with some envs maked as inactive for each policy.

            Args:
                batched_env_states: (E_subset, C, ...): batched env reset states. Each of the E_subset envs is duplicated C times for parallel rollouts.
                active_mask: (E_subset, P): binary mask. 1 indicates this env (all its copies naturally) is active for this policy.

            Returns:
                PolicyState with updated train_state and hstate.
            """

            E_subset = get_pytree_batch_size(batched_env_states)
            C = self.num_copies_per_env
            B = E_subset * C # total batch size per policy

            # Flatten env states to (B, ...) from (E_subset, C, ...).
            reset_states = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]),
                batched_env_states
            )

            # Tile to all policies: (P, B, ...)
            env_state = jax.tree.map(
                lambda x: jnp.repeat(x[None, ...], self.num_policies, axis=0),
                reset_states
            )
            obs = jax.vmap(jax.vmap(self.env.get_obs))(env_state.env_state)

            # Expand mask for env copies: (E_subset, P) → (P, E_subset) → repeat C times → (P, B)
            flat_mask = jnp.repeat(active_mask.T, C, axis=1).astype(jnp.float32)

            train_state = self.policy_repo.train_state
            hstate = self.policy_repo.hstate

            for ppo_update_idx in tqdm(range(self.num_ppo_updates_per_repo_step), desc='PPO Updates...', leave=False):
                self.rng, rollout_rng, update_rng = jax.random.split(self.rng, 3)
                rollout_rngs = jax.random.split(rollout_rng, self.num_policies)
                update_rngs  = jax.random.split(update_rng, self.num_policies)

                train_state, hstate, obs, env_state, losses, info = self.ppo_step(
                    train_state,
                    hstate,
                    obs,
                    env_state,
                    rollout_rngs,
                    update_rngs,
                    env_state,  # reset_state_override: to make gymnax auto-reset to the same env
                    env_mask=flat_mask,
                )

            return PolicyState(train_state, hstate)

        def _make_eval_closures(self, n_pairs: int, render_pixels: bool = False):
            """Factory for (network_apply, env_step) closures used in frozen-policy rollouts.

            Args:
                n_pairs: number of parallel (policy, env) pairs being evaluated.
                render_pixels: if True, env_step also returns pixel observations of the rollout.

            Returns:
                (network_apply, env_step) callables compatible with perform_rollout.
            """
            def network_apply(policy, hstate, obs):
                def single_policy_apply(p_, h_, o_):
                    ac_in = (
                        jax.tree.map(lambda x: x[None, None, ...], o_),
                        jnp.zeros((1, 1), dtype=bool)
                    )
                    new_h, pi, _ = self.network.apply(p_, h_, ac_in)
                    return new_h, pi
                return jax.vmap(single_policy_apply)(policy, hstate, obs)

            def env_step(rng, states, actions, resets):
                step_rngs = jax.random.split(rng, n_pairs)
                next_obs, next_env_state, _, done, info = jax.vmap(
                    self.env.step, in_axes=(0, 0, 0, None, 0)
                )(step_rngs, states, actions, self.env_params, resets)
                if render_pixels:
                    pixel_obs = jax.vmap(self.pixel_renderer)(next_env_state.env_state).image
                    return next_obs, next_env_state, _, done, pixel_obs
                return next_obs, next_env_state, _, done, info

            return network_apply, env_step

        def evaluate_pairing(self, pairing: jax.Array, env_states,
                             rollout_steps: int,
                             params,
                             hstate) -> jax.Array:
            """Evaluates a specific pairing of policies to environments (frozen).

            Supports two modes based on input shape:
              1. env2policy pairing: shape (E,) -> each env evaluated using one policy.
                Values are policy indices w.r.t. policy params pytree.
                 Returns (E, 2) with [solve_rate, avg_return] per env.
              2. policy2env pairing: shape (P, E_subset) -> each policy gets E_subset envs. (no masking, all combinations are evaluated)
                Values are env indices w.r.t. `env_states`.
                 Returns (P, E_subset, 2) with [solve_rate, avg_return] per (policy, env) pair.

            Args:
                pairing: assignment array (1D or 2D based on the mode).
                env_states: pytree of environment reset states to evaluate on.
                rollout_steps: number of rollout steps.
                params: pytree of policy parameters, indexable by ``policy_indices``.
                hstate: pytree of policy hidden states, indexable by ``policy_indices``.
            """
            n_envs = get_pytree_batch_size(env_states)
            if pairing.ndim == 1:
                assert pairing.shape[0] == n_envs
                # env2policy: (E,) → (E, 2)
                env_indices = jnp.arange(n_envs)
                policy_indices = pairing  # (E,)
                n_pairs = n_envs
                output_shape = (n_envs, 2)
            elif pairing.ndim == 2:
                # policy2env: (P, E_subset) → (P, E_subset, 2)
                P, E_subset = pairing.shape
                policy_indices = jnp.arange(P)[:, None].repeat(E_subset, axis=1).reshape(-1)
                env_indices = pairing.reshape(-1)
                n_pairs = P * E_subset
                output_shape = (P, E_subset, 2)
            else:
                raise ValueError(f"pairing must be 1D or 2D, got shape {pairing.shape}")

            params = jax.tree.map(lambda x: x[policy_indices], params)
            hstate = jax.tree.map(lambda x: x[policy_indices, 0], hstate)

            env_state = jax.tree.map(lambda x: x[env_indices], env_states)
            obs = jax.vmap(self.env.get_obs)(env_state.env_state)
            resets = env_state  # same env set as reset-override targets

            network_apply, env_step = self._make_eval_closures(n_pairs, render_pixels=False)

            self.rng, rollout_rng = jax.random.split(self.rng)
            info = perform_rollout(
                network_apply, env_step, params, hstate, env_state, obs,
                resets, rollout_rng, rollout_steps
            )

            returned_episodes = info.get("returned_episode", jnp.zeros_like(info.get("GoalR", jnp.zeros(()))))
            solved = info.get("returned_episode_solved", info.get("GoalR", jnp.zeros(())))
            returns = info.get("returned_episode_returns", jnp.zeros_like(solved))

            episodes_completed = jnp.sum(returned_episodes, axis=0)
            sum_solved = jnp.sum(jnp.where(returned_episodes, solved, 0), axis=0)
            sum_returns = jnp.sum(jnp.where(returned_episodes, returns, 0), axis=0)

            solve_rate = jnp.where(episodes_completed > 0, sum_solved / episodes_completed, 0.0)
            avg_return = jnp.where(episodes_completed > 0, sum_returns / episodes_completed, 0.0)

            return jnp.stack([solve_rate, avg_return], axis=-1).reshape(output_shape)

        def advance(self):
            """
            Train the policy repository for one step:
            1. Sample environments from env pool.
            2. Compute affinity scores for candidate environments.
            3. Derive assginment mask for envs <-> policy pairings from affinity scores.
            4. Run pre-training diagnostics (frozen and finetune evaluation). Used for analysis of affinity score and perfromance correlation.
            5. Train policies on assigned envs with PPO.
            6. Run post-training diagnostics (frozen evaluation on all candidate envs).
            7. Log and call the optional affinity scoring update hooks.
            """

            # Sample candidate environments (overridable by subclasses if needed)
            # Sampling is mostly done to control the computational cost (scales with num_policies * candidate_pool_size), 
            # although it can also be resembeling real-life scenarios where the agent only encounters a subset of possible environments at a time.
            candidate_env_indices = self.sample_candidate_envs()

            # Compute affinity (logic of scoring is defined by subclass)
            candidate_states = jax.tree.map(
                lambda x: x[candidate_env_indices], self.train_env_reset_states
            )
            affinity = self.compute_affinity(candidate_states)

            # Derive binary active mask, indicating assigmnet between policies and environments for training.
            active_mask = self.derive_assignment_mask(affinity) # Shape: (E_subset, P)

            # Pre-train evaluation of all possible pairs. For post-hoc analysis.
            P = self.num_policies
            E_subset = candidate_env_indices.shape[0]
            all_pairs = jnp.tile(candidate_env_indices[None, :], (P, 1))  # (P, E_subset)
            
            pre_frozen = self.evaluate_pairing(
                all_pairs, self.train_env_reset_states,
                rollout_steps=self.eval_rollout_steps,
                params=self.policy_repo.train_state.params,
                hstate=self.policy_repo.hstate,
            )  # (P, E_subset, 2)

            # Only run periodically (adjustable in config) since being computationally expensive.
            freq = int(self.config.get("diagnostic_finetune_frequency", 1))
            if self.repo_step_counter % freq == 0:
                policy_indices = jnp.repeat(jnp.arange(P), E_subset)
                env_indices = jnp.tile(candidate_env_indices, P)
                pre_finetune = self._finetune_then_eval(
                    policy_indices=policy_indices,
                    env_indices=env_indices,
                    env_states=self.train_env_reset_states,
                    num_steps=int(self.config.get("pretrain_eval_finetune_steps", 20)),
                    output_shape=(P, E_subset, 2),
                )
            else:
                pre_finetune = jnp.full((P, E_subset, 2), jnp.nan) # nan is indicator for "not evaluated" in this step in downstream analysis.

            # choose the env subset from env pool
            batched_candidate_states = jax.tree.map(
                lambda x: x[candidate_env_indices], self.batched_train_env_reset_states
            )
            policy_state = self.train(batched_candidate_states, active_mask)
            self.policy_repo = policy_state

            # to get stable metric of policy performance after training
            post_frozen = self.evaluate_pairing(
                all_pairs, self.train_env_reset_states,
                rollout_steps=self.eval_rollout_steps,
                params=self.policy_repo.train_state.params,
                hstate=self.policy_repo.hstate,
            )  # (P, E_subset, 2)

            self.append_step_log(candidate_env_indices, affinity, active_mask,
                                 post_frozen, pre_frozen, pre_finetune)
            self.update(policy_state, candidate_env_indices, affinity,
                        active_mask, post_frozen)

            self.repo_step_counter += 1

        def sample_candidate_envs(self) -> jax.Array:
            """Samples candidate_pool_size distinct envs randomly from env pool"""
            self.rng, sub_rng = jr.split(self.rng)
            return jr.choice(sub_rng, self.num_distinct_envs,
                             shape=(self.candidate_pool_size,), replace=False)

        @abstractmethod
        def compute_affinity(self, env_reset_states) -> jax.Array:
            """
            Each subclass must implement its own affitnity scoring logic.
            Computation logic for raw affinity scores.
            Affinity scores are calculated per policy - environment pair,
            ideally a proxy of a policy suitability for a certain environment.
            Higher scores indicate a better match. Scores are not required to be normalised in the current setup.

            Args:
                env_reset_states: pytree of environment reset states to score. (N, ...)

            Returns:
                (P, N) float array where N is number of environments.
            """
            pass

        def _greedy_assign(self, eval_env_reset_states) -> tuple[jax.Array, jax.Array]:
            """Compute affinity and derive greedy env → policy assignment for eval step.

            Args:
                eval_env_reset_states: pytree of evaluation env reset states.

            Returns:
                (affinity, env2policy):
                - affinity: (P, E_eval) raw scores from ``compute_affinity``.
                - env2policy: (E_eval,) index of the policy with highest score for each environment.
            """
            affinity = self.compute_affinity(eval_env_reset_states)
            E_eval = affinity.shape[1]
            self.rng, sub_rng = jr.split(self.rng)
            col_rngs = jr.split(sub_rng, E_eval)
            # reason for random tie-breaking:
            # if multiple policies have the same score for an environment, we want to break ties randomly to avoid biasing towards lower-indexed policies.
            env2policy = jax.vmap(_random_argmax)(col_rngs, affinity.T)
            return affinity, env2policy

        def derive_assignment_mask(self, raw_scores: jax.Array) -> jax.Array:
            """Derives (E_subset, P) binary active mask from raw affinity scores.

            Supports three assignment modes controlled by ``config["assignment_mode"]``:

            - ``"greedy"`` (default): each env goes to its highest-scoring policy, with random tie-breaking.
            - ``"softmax"``: column-wise softmax with temperature ``config["assignment_temperature"]``, then categorical sampling.
            - ``"epsilon_greedy"``: greedy choice with probability `1 - ε`, random uniform with probability ε (``config["assignment_epsilon"]``).

            A safety net is applied to ensure that every **policy** gets at least one active env.
            This is to ensure that no policy is left without any training data, which could lead to collapse in the repository by some policies dominating others.
            """
            mode = self.config.get("assignment_mode", "greedy")
            E_subset = raw_scores.shape[1]

            if mode == "greedy":
                self.rng, sub_rng = jr.split(self.rng)
                col_rngs = jr.split(sub_rng, E_subset)
                assignments = jax.vmap(_random_argmax)(
                    col_rngs, raw_scores.T
                )

            elif mode == "softmax":
                tau = self.config["assignment_temperature"]
                probs = jax.nn.softmax(raw_scores / tau, axis=0)
                self.rng, sub_rng = jr.split(self.rng)
                assignments = jax.random.categorical(
                    sub_rng, jnp.log(probs + 1e-12), axis=0
                )

            elif mode == "epsilon_greedy":
                eps = self.config["assignment_epsilon"]
                # Greedy branch: per-column random tie-breaking.
                self.rng, sub_rng = jr.split(self.rng)
                col_rngs = jr.split(sub_rng, E_subset)
                greedy = jax.vmap(_random_argmax)(
                    col_rngs, raw_scores.T
                )
                self.rng, sub_rng = jr.split(self.rng)
                random_ = jr.randint(sub_rng, (E_subset,), 0, self.num_policies)
                self.rng, sub_rng = jr.split(self.rng)
                use_random = jr.uniform(sub_rng, (E_subset,)) < eps
                assignments = jnp.where(use_random, random_, greedy)

            else:
                raise ValueError(
                    f"Unknown assignment_mode '{mode}'; "
                    f"expected 'greedy', 'softmax', or 'epsilon_greedy'"
                )

            mask = jnp.zeros((E_subset, self.num_policies), dtype=bool)
            mask = mask.at[jnp.arange(E_subset), assignments].set(True)

            # Safety net: every policy gets at least one active env (the best-scoring one if none were assigned).
            for p in range(self.num_policies):
                if not mask[:, p].any():
                    self.rng, sub_rng = jr.split(self.rng)
                    env_perm = jr.permutation(sub_rng, E_subset)
                    scores_perm = raw_scores[p, env_perm]
                    best_env = env_perm[jnp.argmax(scores_perm)]
                    mask = mask.at[best_env, p].set(True)

            return mask

        def append_step_log(self, candidate_env_indices, affinity, active_mask,
                             post_frozen_metrics, pre_frozen_metrics,
                             pre_finetune_metrics):
            """Appends a RepoStepLog to self.step_logs."""
            self.step_logs.append(RepoStepLog(
                candidate_env_indices=candidate_env_indices,
                affinity_scores=affinity,
                active_mask=active_mask,
                solve_rate=post_frozen_metrics[:, :, 0],
                avg_return=post_frozen_metrics[:, :, 1],
                pre_frozen_solve_rate=pre_frozen_metrics[:, :, 0],
                pre_frozen_avg_return=pre_frozen_metrics[:, :, 1],
                pre_finetune_solve_rate=pre_finetune_metrics[:, :, 0],
                pre_finetune_avg_return=pre_finetune_metrics[:, :, 1],
            ))

        def evaluate_frozen(self, eval_env_reset_states) -> tuple[jax.Array, jax.Array, jax.Array]:
            """Evaluates the repository on a set of evaluation environments.
            Each eval environment is assigned to its best policy (by affinity), then evaluated with frozen policy parameters.


            Args:
                eval_env_reset_states: pytree of evaluation env reset states.

            Returns:
                Tuple of (affinity, env2policy, eval_metrics):
                - affinity: (P, E_eval) array of affinity scores for each policy-env pair
                - env2policy: (E_eval,) array mapping each eval env to its assigned policy
                - eval_metrics: (E_eval, 2) array with [solve_rate, avg_return] for each eval env under its assigned policy.
            """
            affinity, env2policy = self._greedy_assign(eval_env_reset_states)
            eval_metrics = self.evaluate_pairing(env2policy, eval_env_reset_states,
                                                  rollout_steps=self.eval_rollout_steps,
                                                  params=self.policy_repo.train_state.params,
                                                  hstate=self.policy_repo.hstate)
            return affinity, env2policy, eval_metrics

        def _slice_policy_state(self, indices: jax.Array) -> tuple[TrainState, jax.Array]:
            """Slice the policy repository state to select specific policies.

            Args:
                indices: (P,) int array of policy indices to select.

            Returns:
                (train_state, hstate) sliced to the selected policies.
            """
            ts = self.policy_repo.train_state
            sliced_ts = TrainState(
                step=ts.step[indices],
                apply_fn=ts.apply_fn,
                params=jax.tree.map(lambda x: x[indices], ts.params),
                tx=ts.tx,
                opt_state=jax.tree.map(lambda x: x[indices], ts.opt_state),
            )
            sliced_h = jax.tree.map(lambda x: x[indices], self.policy_repo.hstate)
            return sliced_ts, sliced_h

        def _finetune_then_eval(self, policy_indices, env_indices,
                                    env_states, num_steps, output_shape):
            """Fine-tune temporary policy copies on assigned envs, then evaluate.

            Used by both ``evaluate_finetune`` and per-step diagnostics.
            For each of N assignments, copies the specified policy, fine-tunes it
            on C parallel copies of its assigned env for ``num_steps`` PPO steps,
            then evaluates the fine-tuned policy on that same env (one-to-one).

            Args:
                policy_indices: (N,) int -> which policy to copy for each assignment.
                env_indices: (N,) int -> which env (index into ``env_states``) each
                    copy trains on and is evaluated against.
                env_states: pytree with leading dim >= max(env_indices).
                num_steps: number of PPO update steps for fine-tuning.
                output_shape: shape to reshape final (N, 2) metrics into.

            Returns:
                Metrics array reshaped to ``output_shape``.
            """
            N = policy_indices.shape[0]
            C = self.num_copies_per_env

            ft_train_state, ft_hstate = self._slice_policy_state(policy_indices)

            # Select envs and duplicate: (N, C, ...)
            selected_envs = jax.tree.map(
                lambda x: x[env_indices], env_states
            )
            batched_envs = duplicate_env_states(selected_envs, C)
            obs = jax.vmap(jax.vmap(self.env.get_obs))(batched_envs.env_state)

            # Fine-tuning loop (no mask: each policy trains fully)
            for _ in range(num_steps):
                self.rng, rollout_rng, update_rng = jax.random.split(self.rng, 3)
                rollout_rngs = jax.random.split(rollout_rng, N)
                update_rngs = jax.random.split(update_rng, N)

                ft_train_state, ft_hstate, obs, batched_envs, _, _ = self.ppo_step(
                    ft_train_state,
                    ft_hstate,
                    obs,
                    batched_envs,
                    rollout_rngs,
                    update_rngs,
                    batched_envs,  # reset_state_override: auto-reset to the same envs
                )

            # one-to-one evaluation based on the provided pairing: policy i → selected env i
            metrics = self.evaluate_pairing(
                jnp.arange(N),
                selected_envs,
                rollout_steps=self.eval_rollout_steps,
                params=ft_train_state.params,
                hstate=ft_hstate,
            )  # (N, 2)

            return metrics.reshape(output_shape)

        def evaluate_finetune(self, eval_env_reset_states) -> tuple[jax.Array, jax.Array, jax.Array]:
            """Evaluates with per-env policy fine-tuning before rollout.

            Each eval environment is assigned to its best policy (by affinity),
            then a temporary copy of that policy is fine-tuned on that environment
            for ``evaluate_finetune_steps`` PPO steps before evaluation.
            The fine-tuned parameters are discarded, so that the policy params inside repository remain unchanged.

            Args:
                eval_env_reset_states: pytree of evaluation env reset states.

            Returns:
                Tuple of (affinity, env2policy, finetuned_metrics):
                - affinity: (P, E_eval) array of affinity scores for each policy-env pair
                - env2policy: (E_eval,) array mapping each eval env to its assigned policy
                - finetuned_metrics: (E_eval, 2) array with [solve_rate, avg_return]
                  for each eval env under its fine-tuned assigned policy.
            """
            num_steps = int(self.config.get("evaluate_finetune_steps", 5))
            affinity, env2policy = self._greedy_assign(eval_env_reset_states)
            E_eval = env2policy.shape[0]

            metrics = self._finetune_then_eval(
                policy_indices=env2policy,
                env_indices=jnp.arange(E_eval),
                env_states=eval_env_reset_states,
                num_steps=num_steps,
                output_shape=(E_eval, 2),
            )

            return affinity, env2policy, metrics

        def render_eval_videos(self, env2policy: jax.Array,
                                eval_env_reset_states,
                                video_dir: str) -> None:
            """Render and save MP4 videos of evaluation rollouts.

            Args:
                env2policy: (E,) array mapping each eval env to its assigned policy.
                eval_env_reset_states: pytree of evaluation env reset states.
                video_dir: path to directory where ``env_000.mp4`` … will be saved.
            """
            import mediapy as media
            from pathlib import Path

            video_dir = Path(video_dir)
            video_dir.mkdir(parents=True, exist_ok=True)

            E = env2policy.shape[0]
            policy_indices = env2policy
            env_indices = jnp.arange(E)

            params = jax.tree.map(lambda x: x[policy_indices],
                                   self.policy_repo.train_state.params)
            hstate = jax.tree.map(lambda x: x[policy_indices, 0],
                                   self.policy_repo.hstate)
            env_state = jax.tree.map(lambda x: x[env_indices],
                                      eval_env_reset_states)
            obs = jax.vmap(self.env.get_obs)(env_state.env_state)
            resets = env_state

            network_apply, env_step = self._make_eval_closures(E, render_pixels=True)

            self.rng, rollout_rng = jax.random.split(self.rng)
            pixel_frames = perform_rollout(
                network_apply, env_step, params, hstate, env_state, obs,
                resets, rollout_rng, self.eval_rollout_steps,
            )  # (T, E, H, W, C)

            pixel_frames = np.array(pixel_frames)  # GPU → CPU

            for i in range(E):
                frames_i = pixel_frames[:, i]  # (T, H, W, C)
                out_path = video_dir / f"env_{i:03d}.mp4"
                with media.VideoWriter(
                    str(out_path), shape=frames_i.shape[1:3], fps=30, crf=18
                ) as video:
                    for t in range(frames_i.shape[0]):
                        video.add_image(frames_i[t].transpose(1, 0, 2)[::-1])

        def update(self, policy_state: PolicyState,
                   candidate_env_indices: jax.Array,
                   affinity: jax.Array,
                   active_mask: jax.Array,
                   perf_metrics: jax.Array) -> None:
            """
            Called at the end of each repository training step.
            Subclasses can override this to implement custom behavior, such as updating affinity scoring logic or logging additional metrics.
            Default: no-op.
            """
            pass
  