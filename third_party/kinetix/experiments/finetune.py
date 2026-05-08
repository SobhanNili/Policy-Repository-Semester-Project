import bz2
import json
import os
import pickle
import time
from typing import Any, NamedTuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
from omegaconf import OmegaConf

from kinetix.environment import LogWrapper, PixelObservations
from kinetix.environment.env import make_kinetix_env
from kinetix.models import ScannedRNN, make_network_from_config
from kinetix.render.renderer_pixels import make_render_pixels
from kinetix.util import general_eval, generate_params_from_config, init_wandb, load_evaluation_levels, normalise_config
from kinetix.util.saving import load_params

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Any
    info: Any


def resolve_checkpoint_file(checkpoint_path: str) -> str:
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided for finetuning")
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "full_model.pbz2")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def is_head_param(path: tuple[str, ...]) -> bool:
    return (
        len(path) >= 4
        and path[0] == "params"
        and path[1] == "GeneralActorCriticRNN_0"
        and (path[2].startswith("Dense_") or path[2] == "log_std")
    )


def merge_backbone_params(init_params: Any, checkpoint_params: Any) -> Any:
    init_flat = flatten_dict(to_state_dict(init_params))
    checkpoint_flat = flatten_dict(to_state_dict(checkpoint_params))
    merged_flat = dict(init_flat)

    for path, value in checkpoint_flat.items():
        if is_head_param(path):
            continue
        if path not in merged_flat:
            raise KeyError(f"Checkpoint param missing in init tree: {'/'.join(path)}")
        if merged_flat[path].shape != value.shape:
            raise ValueError(
                f"Shape mismatch while loading backbone params for {'/'.join(path)}: {merged_flat[path].shape} != {value.shape}"
            )
        merged_flat[path] = value

    return unflatten_dict(merged_flat)


def make_trainable_mask(params: Any) -> Any:
    flat_params = flatten_dict(to_state_dict(params))
    mask_flat = {path: is_head_param(path) for path in flat_params}
    return unflatten_dict(mask_flat)


def replicate_train_state(train_state: TrainState, num_copies: int) -> TrainState:
    return jax.tree.map(lambda value: jnp.repeat(jnp.asarray(value)[None, ...], num_copies, axis=0), train_state)


def save_level_checkpoint(train_state: TrainState, config: dict[str, Any], level_name: str, update_step: int) -> None:
    checkpoint_root = os.path.join(config["save_path"], f"{config['run_name']}-checkpoints")
    checkpoint_dir = os.path.join(checkpoint_root, f"update_{int(update_step)}", f"level_{level_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    payload = {
        "update_step": int(update_step),
        "level_name": level_name,
        "params": train_state.params,
        "opt_state": train_state.opt_state,
        "step": train_state.step,
    }
    with bz2.BZ2File(os.path.join(checkpoint_dir, "full_model.pbz2"), "w") as f:
        pickle.dump(payload, f)

    with open(os.path.join(checkpoint_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "level_name": level_name,
                "update_step": int(update_step),
                "reuse_mode": config["reuse_mode"],
                "freeze_backbone": bool(config["freeze_backbone"]),
            },
            f,
            indent=2,
            sort_keys=True,
        )


def make_checkpoint_callback(config: dict[str, Any], level_names: list[str]):
    last_saved_update = {idx: -1 for idx in range(len(level_names))}

    def _callback(level_idx: jnp.ndarray, train_state: TrainState, update_step: jnp.ndarray):
        level_idx_int = int(level_idx)
        update_step_int = int(update_step)
        if update_step_int <= 0:
            return
        if update_step_int % int(config["checkpoint_interval_updates"]) != 0:
            return
        if last_saved_update[level_idx_int] == update_step_int:
            return
        save_level_checkpoint(train_state, config, level_names[level_idx_int], update_step_int)
        last_saved_update[level_idx_int] = update_step_int

    return _callback


def make_wandb_logging_callback(config: dict[str, Any], level_names: list[str], time_start: float):
    last_flush_time = time_start
    last_logged_update = 0
    pending_avg_by_update: dict[int, dict[int, dict[str, float]]] = {}
    pending_logs_by_update: dict[int, dict[int, dict[str, Any]]] = {}

    def _callback(
        level_idx: jnp.ndarray,
        metric: Any,
        raw_info: Any,
        loss_info: Any,
        update_step: jnp.ndarray,
        to_log_videos: Any,
        cheap_due: Any,
        expensive_due: Any,
    ):
        del loss_info
        level_idx_int = int(level_idx)
        level_name = level_names[level_idx_int]
        update_step_int = int(update_step)

        if not (bool(cheap_due) or bool(expensive_due)):
            return

        dones = raw_info["returned_episode"]
        num_completed_episodes = int(jnp.sum(dones))

        per_level_metrics = {
            "returned_episode_returns": float(jnp.mean(metric["returned_episode_returns"])),
            "returned_episode_solved": float(jnp.mean(metric["returned_episode_solved"])),
            "returned_episode_lengths": float(jnp.mean(metric["returned_episode_lengths"])),
            "num_completed_episodes": float(num_completed_episodes),
        }

        logs = {
            f"training_metrics/{level_name}/returned_episode_returns": per_level_metrics["returned_episode_returns"],
            f"training_metrics/{level_name}/returned_episode_solved": per_level_metrics["returned_episode_solved"],
            f"training_metrics/{level_name}/returned_episode_lengths": per_level_metrics["returned_episode_lengths"],
            f"training_metrics/{level_name}/num_completed_episodes": per_level_metrics["num_completed_episodes"],
        }

        if bool(expensive_due) and to_log_videos is not None:
            obs_vid, idx_vid = to_log_videos
            clip_len = max(1, int(idx_vid))
            obs_to_use = np.asarray(obs_vid[:clip_len, 0]).transpose(0, 3, 2, 1)[:, :, ::-1, :]
            logs[f"finetune_media/{level_name}/rollout"] = wandb.Video(
                (obs_to_use * 255).astype(np.uint8),
                fps=15,
                format="gif",
            )

        bucket = pending_avg_by_update.setdefault(update_step_int, {})
        bucket[level_idx_int] = per_level_metrics
        logs_bucket = pending_logs_by_update.setdefault(update_step_int, {})
        logs_bucket[level_idx_int] = logs

        if len(bucket) == len(level_names) and len(logs_bucket) == len(level_names):
            nonlocal last_flush_time
            nonlocal last_logged_update
            now = time.time()
            updates_since_last = max(1, update_step_int - last_logged_update)
            env_steps_per_update = int(config["num_steps"]) * int(config["num_train_envs_per_level"]) * len(level_names)
            total_env_steps = update_step_int * env_steps_per_update
            sps = (updates_since_last * env_steps_per_update) / max(now - last_flush_time, 1e-8)
            sps_agg = total_env_steps / max(now - time_start, 1e-8)

            avg_logs = {
                "training_metrics_avg/returned_episode_returns": float(
                    np.mean([entry["returned_episode_returns"] for entry in bucket.values()])
                ),
                "training_metrics_avg/returned_episode_solved": float(
                    np.mean([entry["returned_episode_solved"] for entry in bucket.values()])
                ),
                "training_metrics_avg/returned_episode_lengths": float(
                    np.mean([entry["returned_episode_lengths"] for entry in bucket.values()])
                ),
                "training_metrics_avg/num_completed_episodes": float(
                    np.mean([entry["num_completed_episodes"] for entry in bucket.values()])
                ),
            }

            ordered_logs: dict[str, Any] = {}
            ordered_level_idxs = sorted(logs_bucket.keys())
            ordered_logs["timing/num_updates"] = update_step_int
            ordered_logs["timing/num_env_steps"] = total_env_steps
            ordered_logs["timing/sps"] = float(sps)
            ordered_logs["timing/sps_agg"] = float(sps_agg)
            for level_i in ordered_level_idxs:
                level_logs = logs_bucket[level_i]
                for key, value in level_logs.items():
                    ordered_logs[key] = value

            ordered_logs.update(avg_logs)
            wandb.log(ordered_logs, step=int(ordered_logs["timing/num_env_steps"]))
            last_flush_time = now
            last_logged_update = update_step_int
            pending_avg_by_update.pop(update_step_int, None)
            pending_logs_by_update.pop(update_step_int, None)

    return _callback


def make_train(config: dict[str, Any], env_params, static_env_params):
    if config["num_train_envs_per_level"] % config["num_minibatches"] != 0:
        raise ValueError("num_train_envs_per_level must be divisible by num_minibatches")

    level_states, _ = load_evaluation_levels(config["train_levels_list"])
    level_names = list(config["train_levels_list"])
    num_levels = len(level_names)
    first_level_state = jax.tree.map(lambda x: x[0], level_states)

    env = LogWrapper(
        make_kinetix_env(
            config["action_type"],
            config["observation_type"],
            None,
            env_params,
            static_env_params,
        )
    )

    render_static_env_params = static_env_params.replace(downscale=4)
    # Always build a renderer for eval rollouts, even when training with symbolic observations.
    pixel_renderer = jax.jit(make_render_pixels(env_params, render_static_env_params))
    pixel_render_fn = lambda x: pixel_renderer(x) / 255.0

    network = make_network_from_config(env, env_params, config)

    rng = jax.random.PRNGKey(config["seed"])
    rng, init_rng = jax.random.split(rng)
    init_obs, _ = jax.vmap(env.reset, in_axes=(0, None, None))(
        jax.random.split(init_rng, config["num_train_envs_per_level"]),
        env_params,
        first_level_state,
    )
    init_done = jnp.zeros((config["num_train_envs_per_level"]), dtype=jnp.bool_)
    init_hstate = ScannedRNN.initialize_carry(config["num_train_envs_per_level"])
    init_x = jax.tree.map(lambda x: x[None, ...], (init_obs, init_done))
    network_params = network.init(init_rng, init_hstate, init_x)

    if config["reuse_mode"] == "full_model":
        checkpoint_payload = load_params(config["checkpoint_path"])
        network_params = checkpoint_payload["params"]
    elif config["reuse_mode"] == "backbone_only":
        checkpoint_payload = load_params(config["checkpoint_path"])
        network_params = merge_backbone_params(network_params, checkpoint_payload["params"])
    elif config["reuse_mode"] == "random_init":
        # Keep freshly initialized params for both backbone and heads.
        pass
    else:
        raise ValueError("reuse_mode must be backbone_only, full_model, or random_init")

    tx = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(learning_rate=config["lr"], eps=1e-5),
    )
    if config["freeze_backbone"]:
        tx = optax.masked(tx, make_trainable_mask(network_params))

    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
    batched_train_state = replicate_train_state(train_state, num_levels)

    def _train_level(train_state, rng, level_state, level_idx):
        rng, reset_rng = jax.random.split(rng)
        init_obs, env_state = jax.vmap(env.reset, in_axes=(0, None, None))(
            jax.random.split(reset_rng, config["num_train_envs_per_level"]),
            env_params,
            level_state,
        )
        init_hstate = ScannedRNN.initialize_carry(config["num_train_envs_per_level"])
        runner_state = (
            train_state,
            env_state,
            init_obs,
            jnp.zeros((config["num_train_envs_per_level"]), dtype=jnp.bool_),
            init_hstate,
            rng,
            0,
        )

        checkpoint_callback = make_checkpoint_callback(config, level_names)
        wandb_logging_callback = make_wandb_logging_callback(config, level_names, time_start=time.time())

        def _log_step_zero_rollout(rng, train_state):
            single_level = jax.tree.map(lambda x: x[None, ...], level_state)
            eval_out = general_eval(
                rng,
                env,
                env_params,
                train_state,
                single_level,
                env_params.max_timesteps,
                1,
                keep_states=True,
                return_trajectories=True,
            )
            states = eval_out[0][0]
            done_idxs = eval_out[0][2]
            obs_vid = jax.vmap(jax.vmap(pixel_render_fn))(states.env_state)

            def _log_callback(level_idx: jnp.ndarray, obs_vid: Any, idx_vid: jnp.ndarray):
                level_idx_int = int(level_idx)
                level_name = level_names[level_idx_int]
                clip_len = max(1, int(idx_vid))
                obs_to_use = np.asarray(obs_vid[:clip_len, 0]).transpose(0, 3, 2, 1)[:, :, ::-1, :]
                payload = {
                        "timing/num_updates": 0,
                        "timing/num_env_steps": 0,
                        f"finetune_media/{level_name}/rollout": wandb.Video(
                            (obs_to_use * 255).astype(np.uint8),
                            fps=15,
                            format="gif",
                        ),
                    }
                wandb.log(payload, step=0)

            jax.debug.callback(_log_callback, level_idx, obs_vid, done_idxs[0])

        if config["use_wandb"]:
            rng, eval_rng = jax.random.split(rng)
            _log_step_zero_rollout(eval_rng, train_state)

        def _update_step(runner_state, update_step):
            def _env_step(carry, _):
                train_state, env_state, last_obs, last_done, hstate, rng, _ = carry
                rng, action_rng, env_rng = jax.random.split(rng, 3)

                ac_in = (jax.tree.map(lambda x: x[None, ...], last_obs), last_done[None, :])
                hstate, pi, value = train_state.apply_fn(train_state.params, hstate, ac_in)
                action = pi.sample(seed=action_rng)
                log_prob = pi.log_prob(action)
                value = value.squeeze(0)
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)

                obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))(
                    jax.random.split(env_rng, config["num_train_envs_per_level"]),
                    env_state,
                    action,
                    env_params,
                    level_state,
                )
                transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
                carry = (train_state, env_state, obs, done, hstate, rng, update_step)
                return carry, transition

            initial_hstate = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["num_steps"])

            train_state, env_state, last_obs, last_done, hstate, rng, _ = runner_state
            ac_in = (jax.tree.map(lambda x: x[None, ...], last_obs), last_done[None, :])
            _, _, last_val = train_state.apply_fn(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    delta = transition.reward + config["gamma"] * next_value * (1 - next_done) - transition.value
                    gae = delta + config["gamma"] * config["gae_lambda"] * (1 - next_done) * gae
                    return (gae, transition.value, transition.done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        _, pi, value = train_state.apply_fn(params, init_hstate[0], (traj_batch.obs, traj_batch.done))
                        log_prob = pi.log_prob(traj_batch.action)
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["clip_eps"], config["clip_eps"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["clip_eps"], 1.0 + config["clip_eps"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, permutation_rng = jax.random.split(rng)
                permutation = jax.random.permutation(permutation_rng, config["num_train_envs_per_level"])
                batch = (init_hstate, traj_batch, advantages, targets)
                batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                batch = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(x, [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:])),
                        1,
                        0,
                    ),
                    batch,
                )

                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, batch)
                return (train_state, init_hstate, traj_batch, advantages, targets, rng), total_loss

            init_hstate = initial_hstate[None, :]
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["update_epochs"])
            train_state = update_state[0]

            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum() / jnp.maximum(1, traj_batch.info["returned_episode"].sum()),
                traj_batch.info,
            )
            rng = update_state[-1]

            cheap_due = jnp.logical_and(
                config["cheap_log_every_updates"] > 0, update_step % config["cheap_log_every_updates"] == 0
            )
            expensive_due = jnp.logical_and(
                config["expensive_log_every_updates"] > 0, update_step % config["expensive_log_every_updates"] == 0
            )
            checkpoint_due = jnp.logical_and(
                config["checkpoint_interval_updates"] > 0, update_step % config["checkpoint_interval_updates"] == 0
            )

            def _fake_video():
                return jnp.zeros(
                    (
                        env_params.max_timesteps,
                        1,
                        *PixelObservations(env_params, render_static_env_params).observation_space(env_params).shape,
                    )
                )

            def _real_eval(rng):
                single_level = jax.tree.map(lambda x: x[None, ...], level_state)
                eval_out = general_eval(
                    rng,
                    env,
                    env_params,
                    train_state,
                    single_level,
                    env_params.max_timesteps,
                    1,
                    keep_states=True,
                    return_trajectories=True,
                )
                states = eval_out[0][0]
                done_idxs = eval_out[0][2]
                obs_vid = jax.vmap(jax.vmap(pixel_render_fn))(states.env_state)
                return obs_vid, done_idxs[0]

            rng, eval_rng = jax.random.split(rng)
            to_log_videos = jax.lax.cond(
                expensive_due,
                _real_eval,
                lambda _: (_fake_video(), jnp.array(0, dtype=jnp.int32)),
                eval_rng,
            )

            def _log_callback(_):
                if config["use_wandb"]:
                    jax.debug.callback(
                        wandb_logging_callback,
                        level_idx,
                        metric,
                        traj_batch.info,
                        loss_info,
                        update_step,
                        to_log_videos,
                        cheap_due,
                        expensive_due,
                    )
                return 0

            def _checkpoint_callback(_):
                jax.debug.callback(checkpoint_callback, level_idx, train_state, update_step)
                return 0

            jax.lax.cond(jnp.logical_or(cheap_due, expensive_due), _log_callback, lambda _: 0, operand=0)
            jax.lax.cond(checkpoint_due, _checkpoint_callback, lambda _: 0, operand=0)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng, update_step + 1)
            return runner_state, metric

        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(1, config["num_updates_per_level"] + 1)
        )
        return {"runner_state": runner_state, "metric": metric}

    rngs = jax.random.split(rng, num_levels)
    level_indices = jnp.arange(num_levels)
    train = jax.vmap(_train_level, in_axes=(0, 0, 0, 0))
    return train, batched_train_state, rngs, level_states, level_indices, level_names


@hydra.main(config_path="../configs", config_name="finetune", version_base=None)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    config["learning"]["num_train_envs"] = config["num_train_envs_per_level"]
    config = normalise_config(config, "FINETUNE")

    derived_total_timesteps = config["num_updates_per_level"] * config["num_steps"] * config["num_train_envs_per_level"]
    if config.get("runname") is not None:
        config["run_name"] = config["runname"]
    else:
        nsteps = int(derived_total_timesteps // 1_000_000)
        letter = "M"
        if nsteps >= 1000:
            nsteps = nsteps // 1000
            letter = "B"
        config["run_name"] = config["env_name"] + "-FINETUNE-" + str(nsteps) + letter + "-" + str(
            config["num_train_envs"]
        )

    if config["reuse_mode"] in {"backbone_only", "full_model"}:
        if config["checkpoint_path"] is None:
            raise ValueError("finetune requires checkpoint_path for backbone_only/full_model modes")
        # Validate checkpoint existence early to fail fast before JAX/env initialization.
        config["checkpoint_path"] = resolve_checkpoint_file(config["checkpoint_path"])
    if config["train_level_mode"] != "list":
        raise ValueError("finetune expects train_levels to use train_level_mode: list")
    if config["reuse_mode"] not in {"backbone_only", "full_model", "random_init"}:
        raise ValueError("reuse_mode must be backbone_only, full_model, or random_init")
    if config["reuse_mode"] == "random_init" and config["freeze_backbone"]:
        print(
            "[WARNING] reuse_mode=random_init with freeze_backbone=true would keep a random backbone frozen. "
            "Overriding freeze_backbone=false so the backbone is trainable."
        )
        config["freeze_backbone"] = False
    if config["num_train_envs_per_level"] % config["num_minibatches"] != 0:
        raise ValueError("num_train_envs_per_level must be divisible by num_minibatches")

    env_params, static_env_params = generate_params_from_config(config)
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)

    if config["use_wandb"]:
        init_wandb(config, "FINETUNE")

    train, batched_train_state, batched_rngs, level_states, level_indices, level_names = make_train(
        config,
        env_params,
        static_env_params,
    )

    out = train(batched_train_state, batched_rngs, level_states, level_indices)
    final_train_states = out["runner_state"][0]

    for level_idx, level_name in enumerate(level_names):
        level_train_state = jax.tree.map(lambda x: x[level_idx], final_train_states)
        save_level_checkpoint(level_train_state, config, level_name, config["num_updates_per_level"])

    if config["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    main()