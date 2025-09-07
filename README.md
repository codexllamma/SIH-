
Project Structure
rail-ai          # Konva.js frontend UI
backend           # Flask/Node backend endpoints
/model/
   rail_env5.py     # Current RL environment (collision/switch focus)
   rail_env6.py     # Work-in-progress (schedule adherence, cascading delays)
   train_ppo2.py     # Standard PPO training script
   eval_model.py    # Evaluate a trained model
   models/
       best_model/
           rail_ppo_best.zip   # Best trained PPO model
README.md
requirements.txt

**Installation**

Pull from the repo and install Python requirements:

pip install -r requirements.txt

For GPU acceleration, install the correct torch version from PyTorch
.

**Training**


python rl/train_ppo2.py
Train continuously 
All logs are saved to tb_logs/. View training progress with:
tensorboard --logdir tb_logs --port 6006

**Evaluation**

Run a trained model deterministically in the environment:
python rl/eval_model.py --model models/best_model


This will:
Load the environment (rail_env5 by default).
Run the policy for N steps.
Print metrics: collisions, delays, successful arrivals.

**Inputs & Outputs**

Input to model:
Observation vector with train positions, speeds, signals, block states, schedule info.

Output from model:
Action vector (per train):

0 → no-op
1 → accelerate
2 → decelerate
3 → stop (station halt if in station)
4 → switch left
5 → switch right

**Roadmap**

Collision-free routing with PPO.
Switch management.
Schedule adherence & cascading delay handling (in rail_env6).
Full integration with frontend dispatcher UI.

**Key Notes**

Long training runs (10M–30M timesteps) are necessary for stable policies.
Use EvalCallback to track best-performing models during training.
Only keep one final model (rail_ppo_best.zip) in repo. Logs/checkpoints are excluded from GitHub to avoid bloat.

Dont try to train on your laptop as computation is heavy for the model the training logs and other stuff are stuff that make the repo bloated and not something you need to have in your systems. The best and most trained model as well as the rail environment used for training and ppo pipleine for training has been added to the repo so you can figure out the stuff for integration as for the modles, rail_env5.py is the best model yet and has been trained fo roughly 28M timesteps. I am workgin on an even better model that can take inputs, use Abhinav's delay prediction system to predict delays and deliver an ouput that will be integrated into the frontend to reflect the changes. The other models are previosu versions of the model that have been refactored for btter use now and given birth to rail_env5.
