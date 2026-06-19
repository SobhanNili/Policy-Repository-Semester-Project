from kinetix.environment.ued.ued import make_reset_fn_from_config
from kinetix.util import generate_params_from_config, load_evaluation_levels
from kinetix.environment import LogWrapper
from kinetix.environment.wrappers import LogEnvState
from kinetix.environment.env import make_kinetix_env
import jax
import jax.random as jr
import jax.numpy as jnp

from kinetix.environment.env_state import EnvParams, StaticEnvParams # tupe checking

def make_env_from_config(config) -> tuple[LogWrapper, EnvParams, StaticEnvParams]:
    """Create an environment template from config. Config has the same structure as in the Kinetix repo.

    Returns:
        Environment template, environment stepping parameters, and static
        environment parameters.
    """
    env_params, static_env_params = generate_params_from_config(config)
    reset_fn = make_reset_fn_from_config(config, env_params, static_env_params)

    # wrapping the env with LogWrapper to get episode returns and lengths in the state for logging
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

def get_random_envs(num_envs: int, env: LogWrapper, env_params: EnvParams, rng) -> LogEnvState:
    """Generate random environment initial states.

    Creates ``num_envs`` unique random environments, each created with its own RNG key.
    Returns single copies, so use ``duplicate_env_states`` if replication is needed.

    Args:
        num_envs: Number of distinct random environments to create.
        env: The environment template.
        env_params: Environment parameters passed to ``env.reset``.
        rng: JAX PRNG key.

    Returns:
        A pytree of ``LogEnvState`` with leading shape ``(num_envs, ...)``.
    """
    reset_rngs = jr.split(rng, num_envs)
    obs0, reset_states = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)
    return reset_states


def load_levels(
    level_names: list[str],
    static_env_params_override=None,
)-> LogEnvState:
    """Load predefined level initial states (single EnvState per level).

    Loads each level from ``level_names`` and returns it as a ``LogEnvState`` pytree.
    Use ``duplicate_env_states`` if copies are needed for training.

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

def duplicate_env_states(env_states, num_copies: int):
    """Duplicate environment states along a new copies axis for stable PPO training.

    Takes a pytree of a number of env states with leading dimension (num_envs, ...)
    and returns a pytree with shape (num_envs, num_copies, ...) where each row is
    replicated ``num_copies`` times along axis 1.

    Useful for PPO training where multiple copies of the same environment are stepped in parallel to reduce variance and improve stability.
    Each copy will experience the same initial state but can diverge due to stochasticity in the environment and policy.

    Args:
        env_states: A pytree (``LogEnvState`` or ``EnvState``) where each leaf has shape ``(num_envs, ...)``.
        num_copies: Number of times to replicate each environment.

    Returns:
        A pytree with the same structure as the input, where each leaf has shape
        ``(num_envs, num_copies, ...)``.
    """
    return jax.tree.map(
        lambda x: jnp.repeat(x[:, None, ...], num_copies, axis=1),
        env_states,
    )


COMPONENT_NAMES = [
    "n_polygons",      
    "n_circles",        
    "n_joints",         
    "n_thrusters",      
    "n_shapes",          
    "n_gray_shapes",   
    "n_green_shapes",  
    "n_blue_shapes",    
    "n_red_shapes",     
]

def count_active_components(env_state) -> jax.Array:
    """Count the number of active (non-padded) components in an EnvState.

    Returns a 1D JAX array where each index corresponds to a component type.
    Use the `utils.env.COMPONENT_NAMES` list to map indices to names.

    A rough metric of environment complexity and scene load. It encompasses number of objects in the scene and also the actuators.
    The color of the shapes also shapes the task: grey objects are neutral, and blue object shall collide with green objects but not the red ones.
    Red objects or certain types of actuators can be absent from environments.

    Args:
        env_state: An EnvState pytree.

    Returns:
        jax.Array of shape (9,) with counts in this order:
            0: n_polygons      - active polygon shapes
            1: n_circles       - active circle shapes
            2: n_joints        - active joints
            3: n_thrusters     - active thrusters
            4: n_shapes        - total active shapes (polygons + circles)
            5: n_gray_shapes   - shapes with role 0 (neutral gray objects)
            6: n_green_shapes  - shapes with role 1 (green objects)
            7: n_blue_shapes   - shapes with role 2 (blue objects)
            8: n_red_shapes    - shapes with role 3 (red objects)

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
        n_polygons,                                                      
        n_circles,                                                       
        jnp.sum(active_joints),                                          
        jnp.sum(active_thrusters),                                       
        n_polygons + n_circles,                                          
        jnp.sum((env_state.polygon_shape_roles == 0) & active_polys) + jnp.sum((env_state.circle_shape_roles == 0) & active_circles),    
        jnp.sum((env_state.polygon_shape_roles == 1) & active_polys) + jnp.sum((env_state.circle_shape_roles == 1) & active_circles),
        jnp.sum((env_state.polygon_shape_roles == 2) & active_polys) + jnp.sum((env_state.circle_shape_roles == 2) & active_circles),
        jnp.sum((env_state.polygon_shape_roles == 3) & active_polys) + jnp.sum((env_state.circle_shape_roles == 3) & active_circles),
    ])