"""
train_ppo_refactor.py

Refactored, robust training script for a RailEnv-based environment using Stable Baselines 3 (PPO).

Key features:
- Dynamic observation-space verification (uses env.observation_space.shape)
- Auto-resume from latest checkpoint
- Chunked training loop (train in chunks and save after each chunk)
- Safe handling of gym vs gymnasium reset() return signatures
- Automatic policy network sizing based on observation dimensionality
- Correct use of torch activation callables (nn.ReLU, not string)
- Robust test loop and TensorBoard logging callback

Drop this next to your `rail_env7.py` (or update the import to match your environment module).
"""

import os
import glob
from datetime import datetime
from typing import Tuple, Optional
import numpy as np
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.logger import configure

# 
from rail_env7 import RailEnv


# ----------------------------- Helper utilities -----------------------------

def safe_reset(env):
    """Call env.reset() and return only the observation (handles gym vs gymnasium).

    Returns

    obs
        The raw observation array returned by the environment.
    """
    res = env.reset()
    # gymnasium (obs, info) style or older gym (obs)
    if isinstance(res, tuple):
        return res[0]
    return res


def unwrap_obs_if_tuple(x):
    return x[0] if isinstance(x, tuple) else x


def get_latest_checkpoint(model_dir: str, pattern: str = "rail_ppo_chunk*.zip") -> Tuple[Optional[str], int]:
    ckpts = glob.glob(os.path.join(model_dir, pattern))
    if not ckpts:
        return None, 0
    # sort by numeric chunk id if present
    def _key(p):
        # try to parse the trailing number
        base = os.path.basename(p)
        parts = base.replace('.zip', '').split('chunk')
        if len(parts) > 1:
            try:
                return int(parts[-1])
            except Exception:
                return 0
        return 0
    ckpts.sort(key=_key)
    latest = ckpts[-1]
    chunk_idx = _key(latest)
    return latest, chunk_idx


def verify_observation_space(env) -> Tuple[int]:
    """Verify the environment's observation space by comparing env.reset() with env.observation_space.

    Raises a ValueError if they mismatch.
    Returns the observed shape if matched.
    """
    obs = safe_reset(env)
    expected_shape = env.observation_space.shape
    if obs.shape != expected_shape:
        raise ValueError(f"Observation space mismatch: expected {expected_shape}, got {obs.shape}")
    print(f"✓ Observation space verified: {obs.shape}")
    return obs.shape


def auto_policy_kwargs_from_obs(obs_dim: int):
    """Create reasonable policy_kwargs depending on observation dimensionality.

    Returns a dict for `policy_kwargs` suitable for PPO.
    """
    # Heuristics for network widths
    if obs_dim <= 64:
        h1, h2 = 128, 64
    else:
        h1 = min(512, max(128, int(obs_dim * 1)))
        h2 = min(256, max(64, int(obs_dim * 0.5)))

    net_arch = [dict(pi=[h1, h2], vf=[h1, h2])]
    return {"net_arch": net_arch, "activation_fn": nn.ReLU}


class TensorboardInfoCallback(BaseCallback):
    """Custom callback to push useful info to tensorboard from `info` dicts returned by the env."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or self.locals.get("info")
        if not infos:
            return True

        # Handle vectorized info (list) or single env info (dict)
        if isinstance(infos, list):
            collision_counts = []
            active_trains = []
            for info in infos:
                if not info:
                    continue
                if "collision_count" in info:
                    collision_counts.append(info.get("collision_count", 0))
                if "active_trains" in info:
                    active_trains.append(info.get("active_trains", 0))
                if "episode_summary" in info:
                    summary = info["episode_summary"]
                    for k, v in summary.items():
                        try:
                            self.logger.record(f"episode/{k}", float(v))
                        except Exception:
                            pass
            if collision_counts:
                self.logger.record("env/collision_count_mean", float(np.mean(collision_counts)))
            if active_trains:
                self.logger.record("env/active_trains_mean", float(np.mean(active_trains)))

        else:
            info = infos
            if "collision_count" in info:
                self.logger.record("env/collision_count", float(info.get("collision_count", 0)))
            if "active_trains" in info:
                self.logger.record("env/active_trains", float(info.get("active_trains", 0)))
            if "episode_summary" in info:
                summary = info["episode_summary"]
                for k, v in summary.items():
                    try:
                        self.logger.record(f"episode/{k}", float(v))
                    except Exception:
                        pass

        return True


# ----------------------------- Main training flow -----------------------------


def main():
    RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = os.path.join("runs", RUN_ID)
    TB_LOG_DIR = os.path.join("tb_logs", RUN_ID)
    CHECKPOINT_DIR = os.path.join("checkpoints", RUN_ID)
    BEST_MODEL_DIR = os.path.join("best_model", RUN_ID)
    EVAL_LOG_DIR = os.path.join("eval_logs", RUN_ID)
    MODEL_DIR = os.path.join("models", "latest")  # where checkpoints are stored/retrieved

    for d in [LOG_DIR, TB_LOG_DIR, CHECKPOINT_DIR, BEST_MODEL_DIR, EVAL_LOG_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)

    # --- Example env config. Replace or load from file as you prefer ---
    env_config = {
        "n_tracks": 4,
        # Keep `max_trains` in config so obs/action spaces are consistent regardless of curriculum stage
        "max_trains": 10,
        "n_trains": 10,
        "track_length": 1200,
        "train_length": 25,
        "station_halt_time": 12,
        "max_speed": 4,
        "max_steps": 1800,
        "spawn_points": [50, 320, 680,1450],
        "log_dir": LOG_DIR,
    }

    print(f"Creating base env with config: {env_config}")
    base_env = RailEnv(config=env_config)

    # Sanity check environment using SB3's checker
    try:
        check_env(base_env, warn=True)
    except Exception as e:
        print("Warning: env check failed (this may be normal if the env uses randomness during reset/step).", e)

    # Wrap with Monitor for logging
    env = Monitor(base_env, LOG_DIR)

    # Verify observation shape matches the declared observation_space
    obs_shape = verify_observation_space(env)
    obs_dim = int(np.prod(obs_shape))

    print(f"Observation space: {env.observation_space}, dim={obs_dim}")
    print(f"Action space: {env.action_space}")

    # Auto compute sensible policy kwargs
    policy_kwargs = auto_policy_kwargs_from_obs(obs_dim)
    print("Using policy_kwargs:", policy_kwargs)

    # Setup tensorboard logger
    new_logger = configure(TB_LOG_DIR, ["stdout", "tensorboard"])

    # Try to resume from latest checkpoint
    latest_ckpt, chunk_idx = get_latest_checkpoint(MODEL_DIR)
    model = None

    if latest_ckpt:
        print(f"Found latest checkpoint: {latest_ckpt} (chunk {chunk_idx})")
        try:
            model = PPO.load(latest_ckpt, env=env)
            model.set_logger(new_logger)
            print("✓ Successfully loaded model from checkpoint")
        except Exception as e:
            print("Failed to load checkpoint. Starting new model. Error:", e)
            model = None

    if model is None:
        print("Creating new PPO model")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=TB_LOG_DIR,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
        )
        model.set_logger(new_logger)

    # Create eval env (fresh instance)
    eval_env = Monitor(RailEnv(config=env_config), EVAL_LOG_DIR)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="rail_ppo_chunk",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=EVAL_LOG_DIR,
        eval_freq=20_000,
        deterministic=True,
    )

    info_callback = TensorboardInfoCallback()

    # Training hyperparams
    CHUNK_STEPS = 100_000
    MAX_TOTAL_STEPS = 2_500_000

    total_steps_trained = chunk_idx * CHUNK_STEPS

    print(f"Starting training loop (already trained steps: {total_steps_trained})")

    try:
        while total_steps_trained < MAX_TOTAL_STEPS:
            chunk_idx += 1
            print(f"\n=== Training chunk {chunk_idx} — total_steps_trained={total_steps_trained} ===")

            model.learn(
                total_timesteps=CHUNK_STEPS,
                reset_num_timesteps=False,
                callback=[checkpoint_callback, eval_callback, info_callback],
                tb_log_name=f"PPO_chunk_{chunk_idx}",
            )

            # Save model after each chunk
            save_path = os.path.join(MODEL_DIR, f"rail_ppo_chunk{chunk_idx}")
            model.save(save_path)
            print(f"Saved checkpoint: {save_path}")

            total_steps_trained += CHUNK_STEPS

            # Quick deterministic test run
            test_obs = safe_reset(eval_env)
            for _ in range(10):
                action, _ = model.predict(test_obs, deterministic=True)
                test_obs, reward, terminated, truncated, info = eval_env.step(action)
                test_obs = unwrap_obs_if_tuple(test_obs)
                if terminated or truncated:
                    break
            print("Test run info:", info)

    except KeyboardInterrupt:
        print("Training interrupted by user — saving current model...")
        final_path = os.path.join(MODEL_DIR, f"rail_ppo_interrupted_chunk{chunk_idx}")
        model.save(final_path)
        print(f"Saved interrupted model to {final_path}")

    except Exception as e:
        print("Training error occurred — attempting to save model before re-raising:", e)
        err_save = os.path.join(MODEL_DIR, f"rail_ppo_error_chunk{chunk_idx}")
        try:
            model.save(err_save)
            print(f"Saved model to {err_save}")
        except Exception as se:
            print("Failed to save model during exception handling:", se)
        raise

    finally:
        env.close()
        eval_env.close()
        print("Training finished / cleaned up")


if __name__ == "__main__":
    main()
