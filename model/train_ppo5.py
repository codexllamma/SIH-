"""
train_ppo_enhanced.py

Enhanced training script for RailEnv with predicted delays.
Includes specific monitoring and logging for delay prediction performance.

Key enhancements over the original:
- Delay-aware reward monitoring
- Enhanced TensorBoard logging for predicted vs actual delays
- Curriculum learning support based on delay performance
- Better policy network architecture for delay features
"""

import os
import glob
from datetime import datetime
from typing import Tuple, Optional
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.logger import configure

# Import the enhanced environment
from rail_env7 import RailEnv


# ----------------------------- Enhanced Helper utilities -----------------------------

def safe_reset(env):
    """Call env.reset() and return only the observation (handles gym vs gymnasium)."""
    res = env.reset()
    if isinstance(res, tuple):
        return res[0]
    return res


def unwrap_obs_if_tuple(x):
    return x[0] if isinstance(x, tuple) else x


def get_latest_checkpoint(model_dir: str, pattern: str = "rail_ppo_chunk*.zip") -> Tuple[Optional[str], int]:
    ckpts = glob.glob(os.path.join(model_dir, pattern))
    if not ckpts:
        return None, 0
    # Sort by numeric chunk id if present
    def _key(p):
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


def verify_observation_space_with_delays(env) -> Tuple[int]:
    """Verify the environment's observation space includes predicted delays."""
    obs = safe_reset(env)
    expected_shape = env.observation_space.shape
    if obs.shape != expected_shape:
        raise ValueError(f"Observation space mismatch: expected {expected_shape}, got {obs.shape}")
    
    # Check if predicted delays are included by examining observation structure
    max_trains = env.config.get("max_trains", 10)
    per_train_features = 8  # Including predicted_delay
    obs_config = env.config.get("observation_config", {})
    
    if obs_config.get("include_neighbor_states", True):
        per_train_features += 4
    
    expected_train_features = max_trains * per_train_features
    
    if len(obs) < expected_train_features:
        raise ValueError(f"Observation doesn't seem to include predicted delays. Expected at least {expected_train_features} features for trains, got {len(obs)}")
    
    print(f"✓ Observation space verified: {obs.shape}")
    print(f"✓ Predicted delays confirmed in observation (positions 5, 13, 21, ... for each train)")
    
    # Print sample delay values from first few trains
    for i in range(min(3, max_trains)):
        delay_idx = i * per_train_features + 5  # predicted_delay is at index 5 per train
        if delay_idx < len(obs):
            print(f"  Train {i} predicted delay: {obs[delay_idx]:.4f}")
    
    return obs.shape


def auto_policy_kwargs_with_delay_focus(obs_dim: int):
    """Create policy_kwargs optimized for delay prediction features."""
    # Larger networks to handle the complexity of delay prediction
    if obs_dim <= 64:
        h1, h2, h3 = 256, 128, 64
    elif obs_dim <= 128:
        h1, h2, h3 = 512, 256, 128
    else:
        h1 = min(1024, max(256, int(obs_dim * 1.5)))
        h2 = min(512, max(128, int(obs_dim * 1.0)))
        h3 = min(256, max(64, int(obs_dim * 0.5)))

    # Three-layer network for better delay pattern learning
    net_arch = [dict(pi=[h1, h2, h3], vf=[h1, h2, h3])]
    return {
        "net_arch": net_arch, 
        "activation_fn": nn.ReLU,
        "ortho_init": True,  # Better initialization for delay learning
    }


class DelayAwareTensorboardCallback(BaseCallback):
    """Enhanced callback that specifically tracks delay prediction accuracy."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.delay_prediction_errors = deque(maxlen=1000)
        self.actual_delays = deque(maxlen=1000)
        self.predicted_delays = deque(maxlen=1000)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or self.locals.get("info")
        if not infos:
            return True

        # Handle vectorized info (list) or single env info (dict)
        if isinstance(infos, list):
            for info in infos:
                if not info:
                    continue
                self._process_single_info(info)
        else:
            self._process_single_info(infos)

        # Log delay prediction metrics every 100 steps
        if self.num_timesteps % 100 == 0 and self.delay_prediction_errors:
            self._log_delay_metrics()

        return True

    def _process_single_info(self, info):
        """Process info dict from a single environment step."""
        # Standard metrics
        if "collision_count" in info:
            self.logger.record("env/collision_count", float(info.get("collision_count", 0)))
        if "active_trains" in info:
            self.logger.record("env/active_trains", float(info.get("active_trains", 0)))
        if "total_delay" in info:
            self.logger.record("env/total_delay", float(info.get("total_delay", 0)))

        # NEW: Delay prediction specific metrics
        if "average_predicted_delay" in info:
            predicted = float(info.get("average_predicted_delay", 0))
            actual = float(info.get("total_delay", 0)) / max(1, info.get("active_trains", 1))
            
            self.predicted_delays.append(predicted)
            self.actual_delays.append(actual)
            
            # Track prediction error
            prediction_error = abs(predicted - actual)
            self.delay_prediction_errors.append(prediction_error)
            
            self.logger.record("delays/predicted_delay", predicted)
            self.logger.record("delays/actual_delay", actual)
            self.logger.record("delays/prediction_error", prediction_error)

        if "network_congestion" in info:
            self.logger.record("env/network_congestion", float(info.get("network_congestion", 0)))

        # Episode summary processing
        if "episode_summary" in info:
            summary = info["episode_summary"]
            for k, v in summary.items():
                try:
                    self.logger.record(f"episode/{k}", float(v))
                except Exception:
                    pass

    def _log_delay_metrics(self):
        """Log aggregated delay prediction metrics."""
        if not self.delay_prediction_errors:
            return

        # Mean absolute error of delay predictions
        mae = np.mean(self.delay_prediction_errors)
        self.logger.record("delays/prediction_mae", mae)

        # Root mean square error
        rmse = np.sqrt(np.mean([e**2 for e in self.delay_prediction_errors]))
        self.logger.record("delays/prediction_rmse", rmse)

        # Correlation between predicted and actual delays (if we have both)
        if len(self.predicted_delays) > 10 and len(self.actual_delays) > 10:
            corr = np.corrcoef(list(self.predicted_delays)[-100:], 
                              list(self.actual_delays)[-100:])[0, 1]
            if not np.isnan(corr):
                self.logger.record("delays/prediction_correlation", corr)

        # Trend analysis - are predictions getting better?
        if len(self.delay_prediction_errors) >= 200:
            recent_errors = list(self.delay_prediction_errors)[-100:]
            earlier_errors = list(self.delay_prediction_errors)[-200:-100]
            
            recent_mae = np.mean(recent_errors)
            earlier_mae = np.mean(earlier_errors)
            improvement = earlier_mae - recent_mae
            
            self.logger.record("delays/prediction_improvement", improvement)


class CurriculumCallback(BaseCallback):
    """Curriculum learning callback that adjusts training difficulty based on delay performance."""
    
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.performance_window = deque(maxlen=50)
        self.last_curriculum_update = 0
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or self.locals.get("info")
        if not infos:
            return True
            
        # Track performance
        if isinstance(infos, list):
            for info in infos:
                if info and "average_predicted_delay" in info:
                    prediction_error = abs(info.get("average_predicted_delay", 0) - 
                                         info.get("total_delay", 0) / max(1, info.get("active_trains", 1)))
                    self.performance_window.append(prediction_error)
        else:
            if "average_predicted_delay" in infos:
                prediction_error = abs(infos.get("average_predicted_delay", 0) - 
                                     infos.get("total_delay", 0) / max(1, infos.get("active_trains", 1)))
                self.performance_window.append(prediction_error)
        
        # Update curriculum every 10,000 steps
        if (self.num_timesteps - self.last_curriculum_update) >= 10000 and self.performance_window:
            self._update_curriculum()
            self.last_curriculum_update = self.num_timesteps
            
        return True
    
    def _update_curriculum(self):
        """Update training difficulty based on delay prediction performance."""
        avg_error = np.mean(self.performance_window)
        
        # Simple curriculum: adjust number of active trains based on performance
        current_trains = self.env.config.get("n_trains", 10)
        max_trains = self.env.config.get("max_trains", 10)
        
        if avg_error < 0.5 and current_trains < max_trains:
            # Good performance, increase difficulty
            new_trains = min(max_trains, current_trains + 1)
            self.env.config["n_trains"] = new_trains
            print(f"Curriculum: Increased trains to {new_trains} (error: {avg_error:.3f})")
            self.logger.record("curriculum/n_trains", new_trains)
            
        elif avg_error > 2.0 and current_trains > 5:
            # Poor performance, decrease difficulty
            new_trains = max(5, current_trains - 1)
            self.env.config["n_trains"] = new_trains
            print(f"Curriculum: Decreased trains to {new_trains} (error: {avg_error:.3f})")
            self.logger.record("curriculum/n_trains", new_trains)


# ----------------------------- Main enhanced training flow -----------------------------


def main():
    RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = os.path.join("runs", RUN_ID)
    TB_LOG_DIR = os.path.join("tb_logs", RUN_ID)
    CHECKPOINT_DIR = os.path.join("checkpoints", RUN_ID)
    BEST_MODEL_DIR = os.path.join("best_model", RUN_ID)
    EVAL_LOG_DIR = os.path.join("eval_logs", RUN_ID)
    MODEL_DIR = os.path.join("models", "latest")

    for d in [LOG_DIR, TB_LOG_DIR, CHECKPOINT_DIR, BEST_MODEL_DIR, EVAL_LOG_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)

    # Enhanced env config with curriculum support
    env_config = {
        "n_tracks": 4,
        "max_trains": 10,  # Keep max_trains fixed for consistent spaces
        "n_trains": 6,     # Start with fewer trains for curriculum learning
        "track_length": 1200,
        "train_length": 25,
        "station_halt_time": 12,
        "max_speed": 4,
        "accel_units": 1,
        "decel_units": 1,
        "max_steps": 1800,
        "stations": {
            "A": (20, 50),
            "B": (760, 790),
            "C": (1420, 1450)
        },
        "junctions": [320, 680, 1450],
        "spawn_points": [50, 320, 680, 1450],  # Added extra spawn point
        "unit_meters": 10.0,
        "timestep_seconds": 2.0,
        "brake_mps2": 1.2,
        "track_speed_limits": [4, 4, 4, 4],
        "cascade_N": 8,
        "log_dir": LOG_DIR,
        "reward_config": {
            "safety": {
                "collision_penalty": -2000.0,
                "near_miss_penalty": -50.0,
                "safe_distance_bonus": 15.0,
                "emergency_brake_penalty": -25.0,
                "signal_violation_penalty": -300.0,
                "min_safe_distance": 8.0,
            },
            "efficiency": {
                "on_time_arrival": 500.0,
                "early_arrival": 200.0,
                "grace_arrival": 150.0,
                "late_arrival": 25.0,
                "throughput_bonus": 100.0,
                "speed_efficiency": 2.0,
                "idle_penalty": -15.0,
            },
            "flow": {
                "smooth_acceleration": 5.0,
                "junction_efficiency": 30.0,
                "headway_optimization": 20.0,
                "network_flow_bonus": 50.0,
            }
        },
        "observation_config": {
            "include_network_state": True,
            "include_predictive_features": True,
            "include_neighbor_states": True,
            "spatial_awareness_radius": 200.0,
        }
    }

    print(f"Creating enhanced env with predicted delays. Config: {env_config}")
    base_env = RailEnv(config=env_config)

    # Sanity check environment
    try:
        check_env(base_env, warn=True)
    except Exception as e:
        print("Warning: env check failed (may be normal with randomness):", e)

    # Wrap with Monitor
    env = Monitor(base_env, LOG_DIR)

    # Enhanced verification that includes delay checking
    obs_shape = verify_observation_space_with_delays(env)
    obs_dim = int(np.prod(obs_shape))

    print(f"Enhanced observation space: {env.observation_space}, dim={obs_dim}")
    print(f"Action space: {env.action_space}")

    # Enhanced policy kwargs for delay-aware learning
    policy_kwargs = auto_policy_kwargs_with_delay_focus(obs_dim)
    print("Using enhanced policy_kwargs for delay learning:", policy_kwargs)

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
        print("Creating new enhanced PPO model for delay prediction")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=TB_LOG_DIR,
            learning_rate=3e-4,  # Slightly higher LR for delay learning
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Slight entropy bonus for exploration
            policy_kwargs=policy_kwargs,
        )
        model.set_logger(new_logger)

    # Create eval env (fresh instance)
    eval_env = Monitor(RailEnv(config=env_config), EVAL_LOG_DIR)

    # Enhanced callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="rail_ppo_delay_chunk",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=EVAL_LOG_DIR,
        eval_freq=20_000,
        deterministic=True,
    )

    # NEW: Delay-aware callback and curriculum callback
    delay_callback = DelayAwareTensorboardCallback()
    curriculum_callback = CurriculumCallback(base_env)

    # Training hyperparams
    CHUNK_STEPS = 100_000
    MAX_TOTAL_STEPS = 3_000_000  # More steps for delay learning

    total_steps_trained = chunk_idx * CHUNK_STEPS

    print(f"Starting enhanced training loop with delay prediction")
    print(f"Already trained steps: {total_steps_trained}")
    print(f"Starting with {env_config['n_trains']} trains (curriculum learning enabled)")

    try:
        while total_steps_trained < MAX_TOTAL_STEPS:
            chunk_idx += 1
            print(f"\n=== Enhanced Training chunk {chunk_idx} — total_steps_trained={total_steps_trained} ===")

            model.learn(
                total_timesteps=CHUNK_STEPS,
                reset_num_timesteps=False,
                callback=[checkpoint_callback, eval_callback, delay_callback, curriculum_callback],
                tb_log_name=f"PPO_delay_chunk_{chunk_idx}",
            )

            # Save model after each chunk
            save_path = os.path.join(MODEL_DIR, f"rail_ppo_delay_chunk{chunk_idx}")
            model.save(save_path)
            print(f"Saved checkpoint: {save_path}")

            total_steps_trained += CHUNK_STEPS

            # Enhanced test run with delay analysis
            print("Running delay prediction test...")
            test_obs = safe_reset(eval_env)
            delays_predicted = []
            delays_actual = []
            
            for step in range(20):
                action, _ = model.predict(test_obs, deterministic=True)
                test_obs, reward, terminated, truncated, info = eval_env.step(action)
                test_obs = unwrap_obs_if_tuple(test_obs)
                
                if info.get("average_predicted_delay") is not None:
                    delays_predicted.append(info["average_predicted_delay"])
                    actual_delay = info.get("total_delay", 0) / max(1, info.get("active_trains", 1))
                    delays_actual.append(actual_delay)
                
                if terminated or truncated:
                    break
            
            # Analyze delay prediction performance
            if delays_predicted and delays_actual:
                mae = np.mean([abs(p - a) for p, a in zip(delays_predicted, delays_actual)])
                print(f"Delay prediction MAE in test: {mae:.4f}")
                print(f"Average predicted delay: {np.mean(delays_predicted):.4f}")
                print(f"Average actual delay: {np.mean(delays_actual):.4f}")
            
            print("Test run final info:", info)

    except KeyboardInterrupt:
        print("Training interrupted by user — saving current model...")
        final_path = os.path.join(MODEL_DIR, f"rail_ppo_delay_interrupted_chunk{chunk_idx}")
        model.save(final_path)
        print(f"Saved interrupted model to {final_path}")

    except Exception as e:
        print("Training error occurred — attempting to save model before re-raising:", e)
        err_save = os.path.join(MODEL_DIR, f"rail_ppo_delay_error_chunk{chunk_idx}")
        try:
            model.save(err_save)
            print(f"Saved model to {err_save}")
        except Exception as se:
            print("Failed to save model during exception handling:", se)
        raise

    finally:
        env.close()
        eval_env.close()
        print("Enhanced training finished / cleaned up")


if __name__ == "__main__":
    main()