from abc import ABC, abstractmethod
import jax.numpy as jnp
from apa.utils.network import make_network, init_random_policy, init_network_from_params
from apa.utils.ppo import make_ppo_step, perform_generic_jitted_rollout
from kinetix.render.renderer_pixels import make_render_pixels_rl
from apa.utils.env import make_env_from_config, duplicate_env_states
import jax.random as jr
import jax.numpy as jnp
import jax
import wandb
import numpy as np
from tqdm.auto import tqdm

def get_pytree_element_by_index(pytree, i):
    return jax.tree.map(lambda x: x[i], pytree)

def get_pytree_batch_size(pytree):
    leaf = jax.tree_util.tree_leaves(pytree)[0]
    return leaf.shape[0]

from itertools import chain
def flatten_python_2d_nested_list(nested_list):
    return list(chain.from_iterable(nested_list))

from typing import NamedTuple, Optional
from flax.training.train_state import TrainState

class PolicyState(NamedTuple): # supposed to be vmapped. Skill index is same as the batch index
    train_state: TrainState # includes params
    hstate: jax.Array # sake of compatibility

class RepoStepLog(NamedTuple):
    """Structured log for a single repository step, for later analysis."""
    candidate_env_indices: jnp.ndarray   # (E_subset,)
    affinity_scores: jnp.ndarray         # (P, E_subset)
    active_mask: jnp.ndarray             # (E_subset, P) binary
    solve_rate: jnp.ndarray              # (P, E_subset), NaN where inactive
    avg_return: jnp.ndarray              # (P, E_subset), NaN where inactive


class PolicyRepo(ABC):
        def __init__(self,config,unique_env_reset_states,rng) -> None:
            env, env_params, static_env_params = make_env_from_config(config)
            self.pixel_renderer = make_render_pixels_rl(env_params, static_env_params)
            self.network = make_network(config,env,env_params,static_env_params)

            self.env = env
            self.env_params = env_params
            self.static_env_params = static_env_params
            self.config = config
            self.rng = rng
            self.num_policies = config['num_policies']
            self.num_distinct_envs = int(config['num_train_envs'])
            self.num_copies_per_env = int(config['num_copies_per_env'])
            self.candidate_pool_size = int(config.get('candidate_pool_size', config.get('step_env_pool_size', 1)))
            self.eval_rollout_steps = int(config.get('eval_rollout_steps', 512))
            self.num_ppo_updates_per_repo_step = int(config['num_updates'])

            # Validate: total batch per policy must be divisible by num_minibatches
            batch_per_policy = self.candidate_pool_size * self.num_copies_per_env
            if batch_per_policy % config['num_minibatches'] != 0:
                raise ValueError(
                    f"candidate_pool_size ({self.candidate_pool_size}) * num_copies_per_env "
                    f"({self.num_copies_per_env}) = {batch_per_policy} "
                    f"must be divisible by num_minibatches ({config['num_minibatches']})"
                )

            self.unique_env_reset_states = unique_env_reset_states
            assert get_pytree_batch_size(self.unique_env_reset_states) == self.num_distinct_envs
            self.batched_env_reset_states = duplicate_env_states(
                unique_env_reset_states, self.num_copies_per_env
            )

            self.policy_repo = self.init_policies()
            self.repo_step_counter:int = 0
            self.ppo_step = make_ppo_step(config, env, env_params)
            self.step_logs: list[RepoStepLog] = []

        def init_policies(self):
            self.rng, sub_rng = jr.split(self.rng)

            n_concurrent_envs = self.candidate_pool_size * self.num_copies_per_env

            def make_init_random_policy(rng):
                return init_random_policy(self.config, self.network, self.env, self.env_params, n_concurrent_envs, rng)
            
            network_rngs = jr.split(sub_rng, self.num_policies)
            train_state, hstate = jax.vmap(make_init_random_policy)(network_rngs)
            assert get_pytree_batch_size(train_state) == self.num_policies

            return PolicyState(train_state,hstate)

        def train(self, candidate_env_indices, active_mask):
            """Trains all policies on the shared candidate env batch with per-policy masking.

            All policies share the same E_subset × C environment batch.  The mask
            determines which envs are active for each policy's loss computation.

            Args:
                candidate_env_indices: (E_subset,) — which distinct envs to use.
                active_mask: (E_subset, P) — binary; 1 = this env is active for this policy.

            Returns:
                PolicyState with updated train_state and hstate.
            """
            E_subset = candidate_env_indices.shape[0]
            C = self.num_copies_per_env
            B = E_subset * C

            # All policies share the same envs
            # batched_env_reset_states: (E, C, ...) → [candidate_env_indices] → (E_subset, C, ...)
            reset_states = jax.tree.map(lambda x: x[candidate_env_indices], self.batched_env_reset_states)
            # Flatten to (B, ...)
            reset_states = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]),
                reset_states
            )
            # Tile to all policies: (P, B, ...)
            env_state = jax.tree.map(
                lambda x: jnp.repeat(x[None, ...], self.num_policies, axis=0),
                reset_states
            )
            obs = jax.vmap(jax.vmap(self.env.get_obs))(env_state.env_state)

            # Expand mask: (E_subset, P) → (P, E_subset) → repeat C → (P, B)
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
                    env_state,  # reset_state_override: auto-reset to the same env set
                    env_mask=flat_mask,
                )

            return PolicyState(train_state, hstate)
        
        def log_vids2wandb(self, candidate_env_indices, affinity, active_mask):
            """Logs evaluation videos to wandb.

            For each policy, selects the env with the highest affinity among those
            active for that policy, and runs a frozen-policy rollout for video capture.
            """
            P = self.num_policies
            # For each policy p, pick the active env with highest affinity
            best_env_indices = []
            for p in range(P):
                # Mask out inactive envs for this policy
                masked_aff = jnp.where(active_mask[:, p], affinity[p, :], -jnp.inf)
                best_local_idx = jnp.argmax(masked_aff)
                best_env_indices.append(candidate_env_indices[best_local_idx])
            best_env_indices = jnp.array(best_env_indices)  # (P,)

            # One env per policy (single copy)
            reset_states = jax.tree.map(lambda x: x[best_env_indices, 0], self.batched_env_reset_states)
            env_state = reset_states
            obs = jax.vmap(self.env.get_obs)(env_state.env_state)

            hstate = jax.tree.map(lambda x: x[:, 0], self.policy_repo.hstate)
            params = self.policy_repo.train_state.params

            def network_apply(p, h, o):
                def single_policy_apply(p_, h_, o_):
                    ac_in = (
                        jax.tree.map(lambda x: x[None, None, ...], o_),
                        jnp.zeros((1, 1), dtype=bool)
                    )
                    new_h, pi, _ = self.network.apply(p_, h_, ac_in)
                    return new_h, pi
                return jax.vmap(single_policy_apply)(p, h, o)

            def env_step(rng, states, actions, resets):
                step_rngs = jax.random.split(rng, P)
                next_obs, next_env_state, _, done, info = jax.vmap(self.env.step, in_axes=(0, 0, 0, None, 0))(
                    step_rngs, states, actions, self.env_params, resets
                )
                pixel_obs = jax.vmap(self.pixel_renderer)(next_env_state.env_state).image
                return next_obs, next_env_state, _, done, pixel_obs

            self.rng, rollout_rng = jax.random.split(self.rng)
            video_frames = perform_generic_jitted_rollout(
                network_apply, env_step, params, hstate, env_state, obs,
                reset_states, rollout_rng, self.env_params.max_timesteps
            )

            video_frames = np.array(video_frames)
            data2log = {"repo_step": self.repo_step_counter}

            for i in range(P):
                obs_to_use = video_frames[:, i]
                if obs_to_use.max() <= 1.0:
                    obs_to_use = obs_to_use * 255.0
                vid = obs_to_use.transpose(0, 3, 2, 1)
                vid = vid[:, :, ::-1, :]
                vid = vid.astype(np.uint8)
                env_id = int(best_env_indices[i])
                data2log[f"videos/policy_{i}_env_{env_id}"] = wandb.Video(vid, fps=15, format="gif")

            wandb.log(data2log)

        def log_step2wandb(self, candidate_env_indices, affinity, active_mask, perf_metrics):
            """Logs per-step structured data to wandb.

            Args:
                candidate_env_indices: (E_subset,)
                affinity: (P, E_subset)
                active_mask: (E_subset, P)
                perf_metrics: (P, E_subset, 2) — [solve_rate, avg_return], NaN where inactive.
            """
            data2log = {"repo_step": self.repo_step_counter}
            P = self.num_policies

            solve_rate = perf_metrics[:, :, 0]  # (P, E_subset)
            avg_return = perf_metrics[:, :, 1]  # (P, E_subset)

            for policy_idx in range(P):
                # Per-policy metrics: mean over active envs only (ignoring NaN)
                sr = solve_rate[policy_idx]
                ar = avg_return[policy_idx]
                active = active_mask[:, policy_idx]
                active_sr = sr[active]
                active_ar = ar[active]

                mean_sr = np.nanmean(np.array(active_sr)) if len(active_sr) > 0 else 0.0
                mean_ar = np.nanmean(np.array(active_ar)) if len(active_ar) > 0 else 0.0
                num_active = int(jnp.sum(active))

                data2log.update({
                    f'ep_return/policy_{policy_idx}': mean_ar,
                    f'solve_rate/policy_{policy_idx}': mean_sr,
                    f'num_active_envs/policy_{policy_idx}': num_active,
                })

            wandb.log(data2log)

        def evaluate_pairing(self, pairing: jax.Array, rollout_steps: Optional[int] = None) -> jax.Array:
            """Evaluates a specific pairing of policies to environments.

            Supports two modes based on input shape:
              1. env2policy: shape (E,) — each env gets one policy.
                 Returns (E, 2) with [solve_rate, avg_return] per env.
              2. policy2env:  shape (P, S) — each policy gets S envs.
                 Returns (P, S, 2) with [solve_rate, avg_return] per (policy, env) pair.

            Args:
                pairing: assignment array (1D or 2D).
                rollout_steps: number of rollout steps (default: max_timesteps).
            """
            if rollout_steps is None:
                rollout_steps = self.env_params.max_timesteps
            if pairing.ndim == 1:
                # env2policy: (E,) → (E, 2)
                env_indices = jnp.arange(self.num_distinct_envs)
                policy_indices = pairing  # (E,)
                n_pairs = self.num_distinct_envs
                output_shape = (self.num_distinct_envs, 2)
            elif pairing.ndim == 2:
                # policy2env: (P, S) → (P, S, 2)
                P, S = pairing.shape
                policy_indices = jnp.arange(P)[:, None].repeat(S, axis=1).reshape(-1)
                env_indices = pairing.reshape(-1)
                n_pairs = P * S
                output_shape = (P, S, 2)
            else:
                raise ValueError(f"pairing must be 1D or 2D, got shape {pairing.shape}")

            params = jax.tree.map(lambda x: x[policy_indices], self.policy_repo.train_state.params)
            hstate = jax.tree.map(lambda x: x[policy_indices, 0], self.policy_repo.hstate)

            env_state = jax.tree.map(lambda x: x[env_indices], self.unique_env_reset_states)
            obs = jax.vmap(self.env.get_obs)(env_state.env_state)
            resets = env_state  # same env set as reset-override targets

            def network_apply(p, h, o):
                def single_policy_apply(p_, h_, o_):
                    ac_in = (
                        jax.tree.map(lambda x: x[None, None, ...], o_),
                        jnp.zeros((1, 1), dtype=bool)
                    )
                    new_h, pi, _ = self.network.apply(p_, h_, ac_in)
                    return new_h, pi
                return jax.vmap(single_policy_apply)(p, h, o)

            def env_step(rng, states, actions, resets):
                step_rngs = jax.random.split(rng, n_pairs)
                return jax.vmap(self.env.step, in_axes=(0, 0, 0, None, 0))(
                    step_rngs, states, actions, self.env_params, resets
                )

            self.rng, rollout_rng = jax.random.split(self.rng)
            info = perform_generic_jitted_rollout(
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

        def evaluate_policy_env_matrix(self) -> jax.Array:
            """Evaluates every policy on every distinct environment.

            Returns:
                jax.Array of shape (num_policies, num_distinct_envs, 2)
                with [solve_rate, avg_return] for each (policy, env) pair.
            """
            P, E = self.num_policies, self.num_distinct_envs
            all_pairs = jnp.arange(E)[None, :].repeat(P, axis=0)  # (P, E)
            return self.evaluate_pairing(all_pairs)

        def advance(self):
            """Execute one repository step: sample candidates, compute affinity,
            derive mask, train with masked PPO, evaluate frozen policy, log."""
            # 1. Sample candidate environments (overridable by subclasses)
            candidate_env_indices = self.sample_candidate_envs()

            # 2. Compute affinity (subclass defines the scoring logic)
            affinity = self.compute_affinity(candidate_env_indices)

            # 3. Derive binary active mask
            active_mask = self.derive_assignment_mask(affinity)

            # 4. Masked PPO training
            policy_state = self.train(candidate_env_indices, active_mask)
            self.policy_repo = policy_state

            # 5. Post-training frozen-policy evaluation
            perf_metrics = self.evaluate_frozen_on_candidates(candidate_env_indices, active_mask)

            # 6. Store structured log + allow subclass hooks
            self.append_step_log(candidate_env_indices, affinity, active_mask, perf_metrics)
            self.update(policy_state, candidate_env_indices, affinity, active_mask, perf_metrics)

            # 7. WandB logging
            if self.config['use_wandb']:
                self.log_step2wandb(candidate_env_indices, affinity, active_mask, perf_metrics)
                video_logging_freq = self.config.get('video_frequency', None)
                if video_logging_freq is not None and self.repo_step_counter % video_logging_freq == 0:
                    self.log_vids2wandb(candidate_env_indices, affinity, active_mask)

            self.repo_step_counter += 1

        # ── New shared methods ──────────────────────────────────────────────

        def sample_candidate_envs(self) -> jax.Array:
            """Samples candidate_pool_size distinct envs uniformly at random.
            Override in subclasses for custom sampling (e.g. ConstantMatchingRepo)."""
            self.rng, sub_rng = jr.split(self.rng)
            return jr.choice(sub_rng, self.num_distinct_envs,
                             shape=(self.candidate_pool_size,), replace=False)

        @abstractmethod
        def compute_affinity(self, candidate_env_indices: jax.Array) -> jax.Array:
            """Computes softmax-normalised affinity scores across policies.

            Args:
                candidate_env_indices: (E_subset,) — env indices to score.

            Returns:
                (P, E_subset) float array where for each env e:
                sum_p affinity[p, e] == 1.0 (column-wise softmax).
            """
            pass

        def derive_assignment_mask(self, affinity: jax.Array) -> jax.Array:
            """Derives (E_subset, P) binary active mask by sampling from affinity.

            Each column ``affinity[:, e]`` sums to 1, so it is a valid categorical
            distribution.  We sample one policy per env from that distribution.
            Two safety nets are applied afterwards:

            1. Every **policy** gets at least one active env.
            2. Every **env** gets assigned to at least one policy.

            Override for deterministic (argmax) or more sophisticated assignment
            (e.g. threshold, Hungarian, auction).
            """
            E_subset = affinity.shape[1]
            self.rng, sub_rng = jr.split(self.rng)
            assignments = jax.random.categorical(sub_rng, jnp.log(affinity + 1e-12), axis=0)
            mask = jnp.zeros((E_subset, self.num_policies), dtype=bool)
            mask = mask.at[jnp.arange(E_subset), assignments].set(True)

            # Safety net 1: every policy gets at least one active env
            for p in range(self.num_policies):
                if not mask[:, p].any():
                    best_env = jnp.argmax(affinity[p, :])
                    mask = mask.at[best_env, p].set(True)

            # Safety net 2: every env gets assigned to at least one policy
            for e in range(E_subset):
                if not mask[e, :].any():
                    best_policy = jnp.argmax(affinity[:, e])
                    mask = mask.at[e, best_policy].set(True)

            return mask

        def evaluate_frozen_on_candidates(self, candidate_env_indices, active_mask):
            """Runs a short frozen-policy rollout on all P × E_subset pairs.

            Returns:
                (P, E_subset, 2) array with [solve_rate, avg_return].
                Entries are NaN where active_mask[e, p] == 0.
            """
            E_subset = candidate_env_indices.shape[0]
            all_pairs = jnp.tile(candidate_env_indices[None, :], (self.num_policies, 1))  # (P, E_subset)
            results = self.evaluate_pairing(all_pairs, rollout_steps=self.eval_rollout_steps)  # (P, E_subset, 2)

            # Mask inactive pairs with NaN
            mask_expanded = active_mask.T[:, :, None]  # (P, E_subset, 1)
            results = jnp.where(mask_expanded, results, jnp.nan)
            return results

        def append_step_log(self, candidate_env_indices, affinity, active_mask, perf_metrics):
            """Appends a RepoStepLog to self.step_logs."""
            solve_rate = perf_metrics[:, :, 0]
            avg_return = perf_metrics[:, :, 1]
            self.step_logs.append(RepoStepLog(
                candidate_env_indices=candidate_env_indices,
                affinity_scores=affinity,
                active_mask=active_mask,
                solve_rate=solve_rate,
                avg_return=avg_return,
            ))

        def evaluate(self) -> jax.Array:
            """Evaluates the repository by assigning each env to its best policy
            according to the affinity scores.

            Returns:
                jax.Array of shape (num_distinct_envs, 2) with [solve_rate, avg_return].
            """
            all_envs = jnp.arange(self.num_distinct_envs)
            affinity = self.compute_affinity(all_envs)  # (P, E)
            env2policy = jnp.argmax(affinity, axis=0)   # (E,)
            return self.evaluate_pairing(env2policy)

        # ── Hook (non-abstract, overridable) ─────────────────────────────────

        def update(self, policy_state: PolicyState,
                   candidate_env_indices: jax.Array,
                   affinity: jax.Array,
                   active_mask: jax.Array,
                   perf_metrics: jax.Array) -> None:
            """Hook called after training and evaluation each repo step.

            Default: no-op.  Override in subclasses to maintain extra state
            (e.g. pixel-based success history, eval statistics).
            """
            pass
  