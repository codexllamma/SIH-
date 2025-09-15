import os
from stable_baselines3 import PPO

# Import your custom environment and its 5-train configuration
# Make sure the SIMPLIFIED_CONFIG in this file is set to 5 trains!
from model.rail_env7 import RailEnv, SIMPLIFIED_CONFIG

# --- Configuration ---
# 1. Path to the model you want to CONTINUE training
MODEL_PATH = "./best_model/20250915-144229/best_model.zip"

# 2. Path to save the NEW, improved model
NEW_MODEL_SAVE_PATH = "./best_model/railppo_resume" 

# 3. Number of ADDITIONAL timesteps to train for
TRAINING_TIMESTEPS = 800_000

def continue_training():
    """
    Loads a pre-trained PPO model and continues training it.
    """
    print("--- Starting Continued Training ---")
    
    # 1. Create the 5-train environment
    print("Creating the 5-train RailEnv...")
    env = RailEnv(config=SIMPLIFIED_CONFIG)
    
    # 2. Load the pre-trained PPO model
    try:
        print(f"Loading model from: {MODEL_PATH}")
        # We pass `env=env` to connect the model to the environment for training
        model = PPO.load(MODEL_PATH, env=env)
    except FileNotFoundError:
        print(f"ERROR: Model not found at {MODEL_PATH}.")
        print("Please make sure the MODEL_PATH variable is set correctly.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # 3. Continue training the model
    print(f"\n--- Continuing training for an additional {TRAINING_TIMESTEPS} timesteps... ---")
    
    # The `learn` method will pick up where the model left off.
    # `reset_num_timesteps=False` is important for continuing logs and learning rate schedules.
    model.learn(total_timesteps=TRAINING_TIMESTEPS, reset_num_timesteps=False)
            
    # 4. Save the newly improved model
    print(f"\n--- Training Complete ---")
    print(f"Saving new model to: {NEW_MODEL_SAVE_PATH}.zip")
    model.save(NEW_MODEL_SAVE_PATH)
    
    print("\nModel has been updated and saved successfully.")
    env.close()

if __name__ == "__main__":
    continue_training()