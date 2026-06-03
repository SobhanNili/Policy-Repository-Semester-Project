from kinetix.environment.ued.ued import make_reset_fn_from_config
from kinetix.util import generate_params_from_config, load_evaluation_levels
from kinetix.environment import LogWrapper
from kinetix.environment.wrappers import LogEnvState
from kinetix.environment.env import make_kinetix_env
import jax
import jax.random as jr
import jax.numpy as jnp

def make_env_from_config(config):
    """Create an env template from config. (ONLY ENV DEFINITION)"""
    env_params, static_env_params = generate_params_from_config(config)
    reset_fn = make_reset_fn_from_config(config, env_params, static_env_params)

    env = LogWrapper(
        make_kinetix_env(
            action_type=config["action_type"],
            observation_type=config["observation_type"],
            reset_fn=reset_fn,
            env_params=env_params,
            static_env_params=static_env_params,
        )
    )

    return env, env_params, static_env_params

def get_random_envs(indep_envs_count, envs_in_batch, env, env_params, rng):
    # One RNG per distinct env
    group_reset_rngs = jr.split(rng, indep_envs_count) # (indep_envs_count,) shared reset rng for each group of envs

    # Replicate within each group
    env_reset_rngs = jnp.repeat(group_reset_rngs[:, None, :], envs_in_batch, axis=1)   # (indep_envs_count, envs_in_batch), where the rng is the same within a group

    obs0, fixed_reset_states = jax.vmap(
        jax.vmap(env.reset, in_axes=(0, None)),
        in_axes=(0, None),
    )(env_reset_rngs, env_params)

    return fixed_reset_states

def load_and_replicate_levels(
    level_names: list[str],
    envs_in_batch: int,
    static_env_params_override=None,
):
    """
    Load predefined levels and replicate each one envs_in_batch times.
    
    Args:
        level_names: List of level names (e.g., ["s/h0_weak_thrust", "s/h7_unicycle_left"])
        envs_in_batch: Number of copies per level (ENVS_IN_BATCH)
        static_env_params_override: Optional StaticEnvParams to pad all levels to a fixed capacity.
            Useful for ensuring loaded levels match the env_size used by the network.
    
    Returns:
        reset_states: (NUM_LOADED_LEVELS, ENVS_IN_BATCH) pytree of LogEnvState, 
                      each wrapping an individual env state
    """
    # Load all levels (shape: (NUM_LEVELS, ...))
    # If override is provided, levels are padded to that capacity via expand_env_state
    levels, static_env_params = load_evaluation_levels(
        level_names,
        static_env_params_override=static_env_params_override,
    )
    
    # Replicate each level envs_in_batch times
    # From (NUM_LEVELS, ...) to (NUM_LEVELS, ENVS_IN_BATCH, ...)
    replicated = jax.tree.map(
        lambda x: jnp.repeat(x[:, None, ...], envs_in_batch, axis=1),
        levels,
    )

    def wrap_one(env_state):
        return LogEnvState(
            env_state=env_state,
            episode_returns=0.0,
            episode_lengths=0,
            returned_episode_returns=0.0,
            returned_episode_lengths=0,
            timestep=0,
        )

    # Double vmap to wrap each individual env state independently
    wrapped = jax.vmap(jax.vmap(wrap_one))(replicated)
    
    return wrapped

COMPONENT_NAMES = [
    "n_polygons",       # 0: number of active polygon shapes
    "n_circles",        # 1: number of active circle shapes
    "n_joints",         # 2: number of active joints
    "n_thrusters",      # 3: number of active thrusters
    "n_shapes",         # 4: total active shapes (polygons + circles)
    "n_floor_shapes",   # 5: shapes with role 0 (static floor/background)
    "n_green_shapes",   # 6: shapes with role 1 (agent-controlled green)
    "n_blue_shapes",    # 7: shapes with role 2 (agent-controlled blue)
    "n_red_shapes",     # 8: shapes with role 3 (misc/rare)
]

def count_active_components(env_state) -> jax.Array:
    """Count the number of active (non-padded) components in an EnvState.

    Returns a 1D JAX array where each index corresponds to a component type.
    Use the COMPONENT_NAMES list to map indices to names.

    Args:
        env_state: An EnvState pytree.

    Returns:
        jax.Array of shape (9,) with counts in this order:
            0: n_polygons      — active polygon shapes
            1: n_circles       — active circle shapes
            2: n_joints        — active joints
            3: n_thrusters     — active thrusters
            4: n_shapes        — total active shapes (polygons + circles)
            5: n_floor_shapes  — shapes with role 0 (static floor/background)
            6: n_green_shapes  — shapes with role 1 (agent-controlled green)
            7: n_blue_shapes   — shapes with role 2 (agent-controlled blue)
            8: n_red_shapes    — shapes with role 3 (misc/rare)

    Example — single env:
        stats = count_active_components(single_env)
        print(f"Polygons: {int(stats[0])}, Circles: {int(stats[1])}")

    Example — batch of envs (shape (E, C, ...)):
        all_stats = jax.vmap(jax.vmap(count_active_components))(env_states)
        # all_stats shape: (num_envs, num_copies, 9)
        print(f"Env 0 polys: {int(all_stats[0, 0, 0])}")
    """
    active_polys = env_state.polygon.active
    active_circles = env_state.circle.active
    active_joints = env_state.joint.active
    active_thrusters = env_state.thruster.active

    n_polygons = jnp.sum(active_polys)
    n_circles = jnp.sum(active_circles)

    return jnp.array([
        n_polygons,                                                      # 0
        n_circles,                                                       # 1
        jnp.sum(active_joints),                                          # 2
        jnp.sum(active_thrusters),                                       # 3
        n_polygons + n_circles,                                          # 4
        jnp.sum((env_state.polygon_shape_roles == 0) & active_polys),    # 5
        jnp.sum((env_state.polygon_shape_roles == 1) & active_polys) +   # 6
        jnp.sum((env_state.circle_shape_roles == 1) & active_circles),
        jnp.sum((env_state.polygon_shape_roles == 2) & active_polys) +   # 7
        jnp.sum((env_state.circle_shape_roles == 2) & active_circles),
        jnp.sum((env_state.polygon_shape_roles == 3) & active_polys) +   # 8
        jnp.sum((env_state.circle_shape_roles == 3) & active_circles),
    ])