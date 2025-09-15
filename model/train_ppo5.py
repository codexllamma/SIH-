"""
train_ppo_fresh_start.py

Modified training script that ALWAYS starts with a fresh model, ignoring any previous checkpoints.
This ensures the model is trained from scratch with your latest rail_env7.py configuration changes.

Key changes from original:
- Removes checkpoint loading/resuming functionality
- Always creates a new model from scratch
- Simplified directory structure (no need for checkpoint management)
- Focus on training with latest environment configuration
"""

import os
from datetime import datetime
from typing import Tuple
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

# Import your latest environment
from rail_env7 import RailEnv


# ----------------------------- Helper utilities -----------------------------

def safe_reset(env):
    """Call env.reset() and return only the observation (handles gym vs gymnasium)."""
    res = env.reset()
    # gymnasium (obs, info) style or older gym (obs)
    if isinstance(res, tuple):
        return res[0]
    return res


def unwrap_obs_if_tuple(x):
    return x[0] if isinstance(x, tuple) else x


def verify_observation_space(env) -> Tuple[int]:
    """Verify the environment's observation space by comparing env.reset() with env.observation_space."""
    obs = safe_reset(env)
    expected_shape = env.observation_space.shape
    if obs.shape != expected_shape:
        raise ValueError(f"Observation space mismatch: expected {expected_shape}, got {obs.shape}")
    print(f"✓ Observation space verified: {obs.shape}")
    return obs.shape


def auto_policy_kwargs_from_obs(obs_dim: int):
    """Create reasonable policy_kwargs depending on observation dimensionality."""
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
    # Create fresh run ID for this training session
    RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = os.path.join("runs", RUN_ID)
    TB_LOG_DIR = os.path.join("tb_logs", RUN_ID)
    BEST_MODEL_DIR = os.path.join("best_model", RUN_ID)
    EVAL_LOG_DIR = os.path.join("eval_logs", RUN_ID)
    FINAL_MODEL_DIR = os.path.join("models", RUN_ID)

    for d in [LOG_DIR, TB_LOG_DIR, BEST_MODEL_DIR, EVAL_LOG_DIR, FINAL_MODEL_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"🆕 STARTING FRESH TRAINING SESSION: {RUN_ID}")
    print("📁 No previous checkpoints will be loaded - training from scratch")

    # --- Update this config to match your latest rail_env7.py changes ---
    env_config = {
        "n_tracks": 3,
        "max_trains": 5,
        "n_trains": 5,
        "track_length": 900,
        "train_length": 25,
        "station_halt_time": 12,
        "max_speed": 4,
        "max_steps": 1200,
        "spawn_points": [50, 320, 680],
        "log_dir": LOG_DIR,
        # Add any new config parameters you've added to rail_env7.py here
    }

    print(f"🚂 Creating environment with latest config: {env_config}")
    base_env = RailEnv(config=env_config)

    # Sanity check environment using SB3's checker
    try:
        check_env(base_env, warn=True)
        print("✅ Environment check passed")
    except Exception as e:
        print(f"⚠️  Environment check warning (may be normal): {e}")

    # Wrap with Monitor for logging
    env = Monitor(base_env, LOG_DIR)

    # Verify observation shape matches the declared observation_space
    obs_shape = verify_observation_space(env)
    obs_dim = int(np.prod(obs_shape))

    print(f"Observation space: {env.observation_space}, dim={obs_dim}")
    print(f"Action space: {env.action_space}")

    # Auto compute sensible policy kwargs based on latest observation space
    policy_kwargs = auto_policy_kwargs_from_obs(obs_dim)
    print(f"Using policy_kwargs: {policy_kwargs}")

    # Setup tensorboard logger
    new_logger = configure(TB_LOG_DIR, ["stdout", "tensorboard"])

    # 🆕 ALWAYS CREATE NEW MODEL - no checkpoint loading
    print("Creating brand new PPO model (ignoring any previous checkpoints)")
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

    # Create eval env (fresh instance with same config)
    eval_env = Monitor(RailEnv(config=env_config), EVAL_LOG_DIR)

    # Callbacks for this fresh training run
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=FINAL_MODEL_DIR,
        name_prefix="fresh_rail_ppo",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=EVAL_LOG_DIR,
        eval_freq=20_000,
        deterministic=True,
    )

    info_callback = TensorboardInfoCallback()

    # Training parameters
    TOTAL_TRAINING_STEPS = 1_000_000  # Adjust as needed
    
    print(f"🚀 Starting fresh training for {TOTAL_TRAINING_STEPS} steps")
    print(f"📈 Monitor training: tensorboard --logdir {TB_LOG_DIR}")

    try:
        model.learn(
            total_timesteps=TOTAL_TRAINING_STEPS,
            reset_num_timesteps=True,  # Fresh start
            callback=[checkpoint_callback, eval_callback, info_callback],
            tb_log_name="FreshPPO",
        )

        # Save final model
        final_model_path = os.path.join(FINAL_MODEL_DIR, "fresh_rail_ppo_final")
        model.save(final_model_path)
        print(f"💾 Saved final model: {final_model_path}")

        # Quick test run with the trained model
        print("🧪 Running quick test with trained model...")
        test_obs = safe_reset(eval_env)
        total_reward = 0
        for step in range(20):
            action, _ = model.predict(test_obs, deterministic=True)
            test_obs, reward, terminated, truncated, info = eval_env.step(action)
            test_obs = unwrap_obs_if_tuple(test_obs)
            total_reward += reward
            if terminated or truncated:
                break
        
        print(f"🏁 Test run completed: {step+1} steps, total reward: {total_reward:.2f}")
        print(f"📊 Final info: {info}")

    except KeyboardInterrupt:
        print("\n⛔ Training interrupted by user")
        interrupted_path = os.path.join(FINAL_MODEL_DIR, "fresh_rail_ppo_interrupted")
        model.save(interrupted_path)
        print(f"💾 Saved interrupted model: {interrupted_path}")

    except Exception as e:
        print(f"\n💥 Training error occurred: {e}")
        error_path = os.path.join(FINAL_MODEL_DIR, "fresh_rail_ppo_error")
        try:
            model.save(error_path)
            print(f"💾 Saved model before crash: {error_path}")
        except Exception as se:
            print(f"❌ Failed to save model during error: {se}")
        raise

    finally:
        env.close()
        eval_env.close()
        print("🧹 Training cleanup completed")


if __name__ == "__main__":
    main()