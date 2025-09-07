"""
train_ppo_endless.py

Endless PPO training for RailEnv:
- Runs forever in chunks (safe for overnight / long jobs).
- Keeps continuous learning (no re-init of PPO).
- Saves checkpoints and best models along the way.
"""

import os
from datetime import datetime
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

from rail_env5 import RailEnv


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
        return True


def main():
    RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = os.path.join("runs", RUN_ID)
    TB_LOG_DIR = os.path.join("tb_logs", RUN_ID)
    CHECKPOINT_DIR = os.path.join("checkpoints", RUN_ID)
    BEST_MODEL_DIR = os.path.join("best_model", RUN_ID)
    EVAL_LOG_DIR = os.path.join("eval_logs", RUN_ID)
    MODEL_DIR = os.path.join("models", RUN_ID)

    for d in [LOG_DIR, TB_LOG_DIR, CHECKPOINT_DIR, BEST_MODEL_DIR, EVAL_LOG_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)

    env_config = {
        "n_tracks": 2,
        "n_trains": 4,
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
    check_env(env, warn=True)

    # Logger
    new_logger = configure(TB_LOG_DIR, ["stdout", "tensorboard"])

    # PPO model
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
    )
    model.set_logger(new_logger)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000, save_path=CHECKPOINT_DIR, name_prefix="rail_ppo"
    )
    eval_env = Monitor(RailEnv(config=env_config), EVAL_LOG_DIR)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=EVAL_LOG_DIR,
        eval_freq=20_000,
        deterministic=True,
        render=False,
    )
    info_callback = TensorboardInfoCallback()

    # Endless chunked training
    CHUNK_STEPS = 200_000
    chunk_idx = 0

    while True:
        chunk_idx += 1
        print(f"=== Training chunk {chunk_idx} ({CHUNK_STEPS} steps) ===")
        model.learn(
            total_timesteps=CHUNK_STEPS,
            reset_num_timesteps=False,
            callback=[checkpoint_callback, eval_callback, info_callback],
        )
        save_path = os.path.join(MODEL_DIR, f"rail_ppo_chunk{chunk_idx}")
        model.save(save_path)
        print(f"Saved checkpoint: {save_path}")


if __name__ == "__main__":
    main()
