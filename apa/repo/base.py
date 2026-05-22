from abc import ABC, abstractmethod
import jax.numpy as jnp
from apa.utils.network import make_network, init_random_policy, init_network_from_params
from apa.utils.ppo import make_ppo_step, perform_generic_jitted_rollout
from apa.utils.log import append_trajectory_metrics
from kinetix.render.renderer_pixels import make_render_pixels_rl
from apa.utils.env import make_env_from_config
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

from typing import NamedTuple
from flax.training.train_state import TrainState

class PolicyState(NamedTuple): # supposed to be vmapped. Skill index is same as the batch index
    train_state: TrainState # includes params
    hstate: jax.Array # sake of compatibility

class PolicyRepo(ABC):
        def __init__(self,config,batched_env_reset_states,rng) -> None:
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
            self.num_ppo_updates_per_repo_step = int(config['num_updates'])

            self.batched_env_reset_states = batched_env_reset_states
            self.unique_env_reset_states = jax.tree.map(lambda x: x[:, 0], batched_env_reset_states)
            assert get_pytree_batch_size(self.unique_env_reset_states ) == self.num_distinct_envs

            self.policy_repo = self.init_policies()
            self.repo_step_counter:int = 0
            self.ppo_step = make_ppo_step(config, env, env_params)

        def init_policies(self):
            self.rng, sub_rng = jr.split(self.rng)

            def make_init_random_policy(rng):
                return init_random_policy(self.config, self.network, self.env, self.env_params,self.num_copies_per_env,rng)
            
            network_rngs = jr.split(sub_rng, self.num_policies)
            train_state, hstate = jax.vmap(make_init_random_policy)(network_rngs)
            assert get_pytree_batch_size(train_state) == self.num_policies

            return PolicyState(train_state,hstate)

        def train(self, policy2env_pairing):
            # Extract corresponding reset states for each policy
            # This results in a PyTree where each leaf has shape (num_policies, num_copies_per_env, ...)
            reset_states = jax.tree.map(lambda x: x[policy2env_pairing], self.batched_env_reset_states)

            env_state = reset_states
            obs = jax.vmap(jax.vmap(self.env.get_obs))(env_state.env_state)

            train_state = self.policy_repo.train_state
            hstate = self.policy_repo.hstate

            per_env_perf_history = {
                k: [[] for _ in range(self.num_policies)]
                for k in ("length", "return", "start_step", "end_step", "solved")
            }

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
                    reset_states # Pass as override so auto-resets use the correct level
                )
                per_env_perf_history = append_trajectory_metrics(info,per_env_perf_history)

            return PolicyState(train_state, hstate), per_env_perf_history
        
        def log_vids2wandb(self, policy2env_pairing):
            """Runs evaluation rollouts and logs videos to wandb."""
            # FROZEN POLICY RUN

            # Use only one copy per environment for video logging
            # reset_states shape: (num_policies, ...)
            reset_states = jax.tree.map(lambda x: x[policy2env_pairing, 0], self.batched_env_reset_states)
            
            env_state = reset_states
            obs = jax.vmap(self.env.get_obs)(env_state.env_state)
            
            # hstate for each policy (taking the first copy's carry)
            # shape: (num_policies, carry_size...)
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
                step_rngs = jax.random.split(rng, self.num_policies)
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
            
            # VIDEO GENERATION

            # video_frames shape: (T, P, W, H, C) or (T, P, H, W, C)
            video_frames = np.array(video_frames) 
            data2log = {"repo_step": self.repo_step_counter}

            for i in range(self.num_policies):
                # obs_to_use shape: (T, W, H, C)
                obs_to_use = video_frames[:, i]
                
                # 1. Scaling
                if obs_to_use.max() <= 1.0:
                    obs_to_use = obs_to_use * 255.0
                
                # 2. Transpose following backbone.py: (0, 3, 2, 1)
                # This maps (T, W, H, C) -> (T, C, H, W)
                vid = obs_to_use.transpose(0, 3, 2, 1)
                
                # 3. Flip the NEW first spatial dimension (Width-turned-Height)
                # backbone.py uses [:, :, ::-1, :] which targets index 2
                vid = vid[:, :, ::-1, :]
                
                # 4. Final cast to uint8
                vid = vid.astype(np.uint8)
                
                policy_id = i
                data2log[f"videos/policy_{policy_id}"] = wandb.Video(vid, fps=15, format="gif")
            
            wandb.log(data2log)

        def log_step2wandb(self,per_env_perf_history):
            data2log = {}
            for policy_idx in range(self.num_policies):
                mean_ep_return = np.mean(per_env_perf_history['return'][policy_idx])
                mean_ep_length = np.mean(per_env_perf_history['length'][policy_idx])
                mean_ep_solved = np.mean(per_env_perf_history['solved'][policy_idx])

                data2log.update(
                    {f'ep_return/policy_{policy_idx}':mean_ep_return,
                    f'ep_length/policy_{policy_idx}':mean_ep_length,
                    f'solve_rate/policy_{policy_idx}':mean_ep_solved})
                
            data2log.update({'repo_step': self.repo_step_counter})
            wandb.log(data2log)

        def evaluate_pairing(self, env2policy_pairing: jax.Array) -> jax.Array:
            """Evaluates a specific pairing of policies to environments.
            env2policy_pairing: jax.Array of shape (num_distinct_envs,) containing policy indices.
            Returns: jax.Array of shape (num_distinct_envs, 2) containing [solve_rate, avg_return].
            """
            params = jax.tree.map(lambda x: x[env2policy_pairing], self.policy_repo.train_state.params)
            hstate = jax.tree.map(lambda x: x[env2policy_pairing, 0], self.policy_repo.hstate)
            
            env_state = self.unique_env_reset_states
            obs = jax.vmap(self.env.get_obs)(env_state.env_state)
            
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
                step_rngs = jax.random.split(rng, self.num_distinct_envs)
                return jax.vmap(self.env.step, in_axes=(0, 0, 0, None, 0))(
                    step_rngs, states, actions, self.env_params, resets
                )

            self.rng, rollout_rng = jax.random.split(self.rng)
            info = perform_generic_jitted_rollout(
                network_apply, env_step, params, hstate, env_state, obs, 
                self.unique_env_reset_states, rollout_rng, self.env_params.max_timesteps
            )
            
            returned_episodes = info.get("returned_episode", jnp.zeros_like(info.get("GoalR", jnp.zeros(()))))
            solved = info.get("returned_episode_solved", info.get("GoalR", jnp.zeros(())))
            returns = info.get("returned_episode_returns", jnp.zeros_like(solved))
            
            episodes_completed = jnp.sum(returned_episodes, axis=0)
            sum_solved = jnp.sum(jnp.where(returned_episodes, solved, 0), axis=0)
            sum_returns = jnp.sum(jnp.where(returned_episodes, returns, 0), axis=0)
            
            solve_rate = jnp.where(episodes_completed > 0, sum_solved / episodes_completed, 0.0)
            avg_return = jnp.where(episodes_completed > 0, sum_returns / episodes_completed, 0.0)
            
            return jnp.stack([solve_rate, avg_return], axis=-1)

        def advance(self):
            policy2env_pairing = self.match()
            policy_state, per_env_perf_history = self.train(policy2env_pairing)
            self.update(policy_state,per_env_perf_history, policy2env_pairing)

            if self.config['use_wandb']:
                self.log_step2wandb(per_env_perf_history)
                video_logging_freq = self.config.get('video_frequency', None)
                if video_logging_freq is not None and self.repo_step_counter % video_logging_freq == 0:
                    self.log_vids2wandb(policy2env_pairing)

            self.repo_step_counter += 1

        @abstractmethod
        def match(self) -> jax.Array:
            """Determines which environment each policy should be trained on.
            
            Returns:
                jax.Array: An array of shape (num_policies,) containing the index of the 
                environment assigned to each policy.
            """
            pass

        @abstractmethod
        def update(self,policy_state: PolicyState, per_env_perf_history: dict, policy2env_pairing: jax.Array) -> None:
            """Updates the repository's internal state post-training.
            
            Args:
                policy_state (PolicyState): The updated policy networks and hidden states after PPO training.
                per_env_perf_history (dict): Performance metrics (returns, lengths, solve rates) of policies during the training step.
                policy2env_pairing (jax.Array): The environment assignments used for the training session.
            """
            pass

        @abstractmethod
        def evaluate(self) -> jax.Array:
            """Evaluates the current repository set of policies.
            
            Returns:
                jax.Array: An array of shape (num_distinct_envs, 2) returning [solve_rate, avg_return] 
                for each environment, evaluated under the repository's policy-to-environment assignment.
            """
            pass
  