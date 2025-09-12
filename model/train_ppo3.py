"""
train_ppo_resume_fixed.py

PPO training for RailEnv with auto-resume and correct observation space:
- Uses the corrected RailEnv that matches the original training configuration.
- Observation space: (36,) to match the saved model.
- Resumes from the latest saved checkpoint if found.
- Continues training in 200k-step chunks (safe for overnight jobs).
- Saves checkpoints and best models along the way.
"""

import os
import glob
from datetime import datetime
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

# Import the corrected RailEnv
from rail_env7 import RailEnv


# Custom TensorBoard info logger
class TensorboardInfoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or self.locals.get("info")
        if infos:
            if isinstance(infos, list):
                collision_counts, delta_pos, active_trains = [], [], []
                for info in infos:
                    if not info:
                        continue
                    if "collision_count" in info:
                        collision_counts.append(info.get("collision_count", 0))
                    if "delta_pos" in info:
                        delta_pos.append(info.get("delta_pos", 0.0))
                    if "active_trains" in info:
                        active_trains.append(info.get("active_trains", 0))
                    
                    # Log episode summary if available
                    if "episode_summary" in info:
                        summary = info["episode_summary"]
                        self.logger.record("episode/completion_rate", summary.get("completion_rate", 0))
                        self.logger.record("episode/on_time_rate", summary.get("on_time_rate", 0))
                        self.logger.record("episode/collision_free", summary.get("collision_free", False))
                        self.logger.record("episode/safety_score", summary.get("safety_score", 0))
                        self.logger.record("episode/efficiency_score", summary.get("efficiency_score", 0))
                        self.logger.record("episode/avg_delay", summary.get("avg_delay", 0))
                
                if collision_counts:
                    self.logger.record("env/collision_count_mean", float(np.mean(collision_counts)))
                if delta_pos:
                    self.logger.record("env/delta_pos_mean", float(np.mean(delta_pos)))
                if active_trains:
                    self.logger.record("env/active_trains_mean", float(np.mean(active_trains)))
                

            else:
                info = infos
                if "collision_count" in info:
                    self.logger.record("env/collision_count", float(info["collision_count"]))
                if "delta_pos" in info:
                    self.logger.record("env/delta_pos", float(info["delta_pos"]))
                if "active_trains" in info:
                    self.logger.record("env/active_trains", float(info["active_trains"]))
                
                # Log episode summary if available
                if "episode_summary" in info:
                    summary = info["episode_summary"]
                    self.logger.record("episode/completion_rate", summary.get("completion_rate", 0))
                    self.logger.record("episode/on_time_rate", summary.get("on_time_rate", 0))
                    self.logger.record("episode/collision_free", summary.get("collision_free", False))
                    self.logger.record("episode/safety_score", summary.get("safety_score", 0))
                    self.logger.record("episode/efficiency_score", summary.get("efficiency_score", 0))
                    self.logger.record("episode/avg_delay", summary.get("avg_delay", 0))
        return True


def get_latest_checkpoint(model_dir: str):
    """Find latest checkpoint if it exists."""
    ckpts = glob.glob(os.path.join(model_dir, "rail_ppo_chunk*.zip"))
    if not ckpts:
        return None, 0
    ckpts.sort(key=lambda x: int(x.split("chunk")[-1].split(".zip")[0]))
    latest = ckpts[-1]
    chunk_idx = int(latest.split("chunk")[-1].split(".zip")[0])
    return latest, chunk_idx


def verify_observation_space(env):
    """Verify the observation space matches expected dimensions."""
    obs, _ = env.reset()
    expected_shape = (264,)  # 6 trains × 6 features per train
    if obs.shape != expected_shape:
        raise ValueError(f"Observation space mismatch: expected {expected_shape}, got {obs.shape}")
    print(f"✓ Observation space verified: {obs.shape}")
    return obs.shape


def main():
    RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = os.path.join("runs", RUN_ID)
    TB_LOG_DIR = os.path.join("tb_logs", RUN_ID)
    CHECKPOINT_DIR = os.path.join("checkpoints", RUN_ID)
    BEST_MODEL_DIR = os.path.join("best_model", RUN_ID)
    EVAL_LOG_DIR = os.path.join("eval_logs", RUN_ID)
    MODEL_DIR = os.path.join("models", "20250905-005856")  # fixed folder for reuse

    for d in [LOG_DIR, TB_LOG_DIR, CHECKPOINT_DIR, BEST_MODEL_DIR, EVAL_LOG_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)

    # Configuration that matches the original training setup
    env_config = {
        "n_tracks": 2,
        "n_trains": 6,
        "track_length": 140,
        "train_length": 10,
        "max_speed": 3,
        "max_steps": 500,
        "stationA_range": (10, 20),
        "switch1_range": (85, 88),
        "switch2_range": (97, 100),
        "stationB_range": (120, 130),
        "spawn_points": [0, 10],
        "log_dir": LOG_DIR,
    }

    print(f"Creating env with config: {env_config}")
    base_env = RailEnv(config=env_config)
    env = Monitor(base_env, LOG_DIR)
    
    # Verify observation space before proceeding
    obs_shape = verify_observation_space(env)
    print(f"Environment observation space: {env.observation_space}")
    print(f"Environment action space: {env.action_space}")
    
    # Check environment
    check_env(env, warn=True)

    new_logger = configure(TB_LOG_DIR, ["stdout", "tensorboard"])

    # Try to resume from checkpoint
    latest_ckpt, chunk_idx = get_latest_checkpoint(MODEL_DIR)

    if latest_ckpt:
        print(f"Found checkpoint: {latest_ckpt}")
        print(f"Resuming training from chunk {chunk_idx}")
        
        try:
            # Load the model and verify compatibility
            model = PPO.load(latest_ckpt, env=env)
            model.set_logger(new_logger)
            print(f"✓ Successfully loaded model from {latest_ckpt}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Creating new model instead...")
            model = None
    else:
        print("No checkpoint found")
        model = None

    # Create new model if loading failed or no checkpoint exists
    if model is None:
        print("Starting fresh training with new model")
        chunk_idx = 0
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=TB_LOG_DIR,
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs={
                "net_arch": [dict(pi=[256, 128], vf=[256, 128])],
                "activation_fn": "relu",
            }
        )
        model.set_logger(new_logger)

    # Create evaluation environment
    eval_env = Monitor(RailEnv(config=env_config), EVAL_LOG_DIR)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000, 
        save_path=CHECKPOINT_DIR, 
        name_prefix="rail_ppo"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=EVAL_LOG_DIR,
        eval_freq=20_000,
        deterministic=True,
        render=False,
    )
    
    info_callback = TensorboardInfoCallback()

    # Training loop parameters
    CHUNK_STEPS = 200_000
    MAX_TOTAL_STEPS = 2_000_000  # Limit total training to prevent infinite loops
    
    print(f"Starting training from chunk {chunk_idx + 1}")
    print(f"Training configuration:")
    print(f"  - Chunk size: {CHUNK_STEPS:,} steps")
    print(f"  - Max total steps: {MAX_TOTAL_STEPS:,}")
    print(f"  - Observation space: {obs_shape}")
    print(f"  - Action space: {env.action_space}")

    total_steps_trained = chunk_idx * CHUNK_STEPS
    
    try:
        while total_steps_trained < MAX_TOTAL_STEPS:
            chunk_idx += 1
            print(f"\n{'='*60}")
            print(f"Training chunk {chunk_idx} ({CHUNK_STEPS:,} steps)")
            print(f"Total steps so far: {total_steps_trained:,}")
            print(f"{'='*60}")
            
            model.learn(
                total_timesteps=CHUNK_STEPS,
                reset_num_timesteps=False,  # Keep cumulative timestep count
                callback=[checkpoint_callback, eval_callback, info_callback],
                tb_log_name=f"PPO_chunk_{chunk_idx}",
            )
            
            # Save checkpoint after each chunk
            save_path = os.path.join(MODEL_DIR, f"rail_ppo_chunk{chunk_idx}")
            model.save(save_path)
            print(f"✓ Saved checkpoint: {save_path}")
            
            total_steps_trained += CHUNK_STEPS
            
            # Test the model briefly after each chunk
            print("Testing current model...")
            test_obs, _ = eval_env.reset()
            for _ in range(10):
                action, _ = model.predict(test_obs, deterministic=True)
                test_obs, reward, terminated, truncated, info = eval_env.step(action)
                if terminated or truncated:
                    break
            print(f"Test episode info: {info}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        final_save_path = os.path.join(MODEL_DIR, f"rail_ppo_interrupted_chunk{chunk_idx}")
        model.save(final_save_path)
        print(f"✓ Saved interrupted model: {final_save_path}")
    
    except Exception as e:
        print(f"\nTraining error: {e}")
        error_save_path = os.path.join(MODEL_DIR, f"rail_ppo_error_chunk{chunk_idx}")
        model.save(error_save_path)
        print(f"✓ Saved model before error: {error_save_path}")
        raise
    
    finally:
        # Cleanup
        env.close()
        eval_env.close()
        print("Training completed or stopped")


if __name__ == "__main__":
    main()