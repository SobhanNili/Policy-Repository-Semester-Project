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

def get_random_envs(num_envs: int, env, env_params, rng):
    """Generate random environment initial states (single copies).

    Creates ``num_envs`` unique random environments, each with its own RNG key.
    Returns only single copies — use ``duplicate_env_states`` if replication is needed.

    Args:
        num_envs: Number of distinct random environments to create.
        env: The environment object (with a ``reset`` method).
        env_params: Environment parameters passed to ``env.reset``.
        rng: JAX PRNG key.

    Returns:
        A pytree of ``LogEnvState`` with leading shape ``(num_envs, ...)``.
    """
    reset_rngs = jr.split(rng, num_envs)
    obs0, fixed_reset_states = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)
    return fixed_reset_states


def load_levels(
    level_names: list[str],
    static_env_params_override=None,
):
    """Load predefined level initial states (single copy per level).

    Loads each level from ``level_names`` and returns it as a unique ``LogEnvState``
    without replication. Use ``duplicate_env_states`` if copies are needed for training.

    Args:
        level_names: List of level names (e.g., ["s/h0_weak_thrust", "s/h7_unicycle_left"]).
        static_env_params_override: Optional ``StaticEnvParams`` to pad all levels to a
            fixed capacity. Useful for ensuring loaded levels match the ``env_size``
            used by the network.

    Returns:
        A pytree of ``LogEnvState`` with leading shape ``(num_levels, ...)``.
    """
    levels, static_env_params = load_evaluation_levels(
        level_names,
        static_env_params_override=static_env_params_override,
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

    return jax.vmap(wrap_one)(levels)

def duplicate_env_states(unique_env_states, num_copies: int):
    """Duplicate unique environment states along a new copies axis for stable PPO training.

    Takes a pytree of unique env states with leading dimension (num_envs, ...)
    and returns a pytree with shape (num_envs, num_copies, ...) where each row is
    replicated ``num_copies`` times along axis 1.

    This is useful when you want multiple independent copies of each environment
    so that PPO training can average gradients over several copies per unique env.

    Args:
        unique_env_states: A pytree (typically ``LogEnvState`` or ``EnvState``) where each
            leaf has shape ``(num_envs, ...)``.
        num_copies: Number of times to replicate each environment.

    Returns:
        A pytree with the same structure as the input, where each leaf has shape
        ``(num_envs, num_copies, ...)``.
    """
    return jax.tree.map(
        lambda x: jnp.repeat(x[:, None, ...], num_copies, axis=1),
        unique_env_states,
    )


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