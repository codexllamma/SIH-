import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time
import copy
import torch

# --- Import your existing, powerful RailEnv and its dependencies ---
from rail_env7 import RailEnv, SIMPLIFIED_CONFIG
from stable_baselines3 import PPO

# --- 1. Centralized State Management ---
# Using a dataclass to ensure all components share a consistent view of the world.
@dataclass
class SystemState:
    """A snapshot of the entire railway system at a single point in time."""
    tick: int
    train_positions: np.ndarray
    train_speeds: np.ndarray
    train_tracks: np.ndarray
    
    # This data will be enriched by the components
    predicted_delays: Dict[int, float] = field(default_factory=dict)
    conflicts: List[Dict] = field(default_factory=list)
    risk_score: float = 0.0
    conflict_severity: float = 0.0

# --- 2. Component 1: Lightweight Delay Predictor ---
# This is an optimized version of your Monte Carlo model, designed for speed.

class LightweightDelayPredictor:
    """Runs continuously to provide a fast, preliminary risk assessment."""
    def __init__(self, historical_data_path="trainsmall.csv"):
        self.delay_models = self._build_models(historical_data_path)
        print("Lightweight Delay Predictor initialized.")

    def _build_models(self, path: str) -> Dict[str, Any]:
        """Analyzes historical data to create a simple delay model for each train type."""
        try:
            df = pd.read_csv(path)
            models = {}
            for train_id, group in df.groupby('train_id'):
                clean_delays = self._remove_outliers(group['delay'].values)
                models[train_id] = {
                    'mean': np.mean(clean_delays),
                    'std': np.std(clean_delays)
                }
            return models
        except FileNotFoundError:
            print(f"Warning: Historical data '{path}' not found. Using default delay models.")
            return {'a': {'mean': 2.0, 'std': 1.5}, 'b': {'mean': 3.5, 'std': 2.0}, 'c': {'mean': 1.0, 'std': 0.8}}

    def _remove_outliers(self, data: np.ndarray, z_threshold=2.0) -> np.ndarray:
        if len(data) == 0: return data
        z_scores = np.abs(stats.zscore(data))
        return data[z_scores < z_threshold]

    def predict(self, state: SystemState) -> SystemState:
        """
        Enriches the system state with delay predictions and a general risk score.
        For simplicity, we'll assign train types 'a', 'b', 'c' cyclically.
        """
        train_types = ['a', 'b', 'c']
        total_predicted_delay = 0
        
        for i in range(len(state.train_positions)):
            train_type = train_types[i % len(train_types)]
            model = self.delay_models.get(train_type, {'mean': 1.0, 'std': 1.0})
            
            # Generate a single, fast prediction
            predicted_delay = max(0, np.random.normal(model['mean'], model['std']))
            state.predicted_delays[i] = predicted_delay
            total_predicted_delay += predicted_delay

        # The risk score is a simple heuristic based on total predicted delay.
        state.risk_score = min(1.0, total_predicted_delay / (len(state.train_positions) * 3.0))
        return state

# --- 3. Component 2: Conflict Detection Engine ---
# This component is only triggered if the initial risk score is high enough.

class ConflictDetectionEngine:
    """Analyzes predicted delays to identify specific, high-severity conflicts."""
    def __init__(self, time_buffer_minutes: float = 5.0, severity_threshold: float = 0.5):
        self.time_buffer = time_buffer_minutes
        self.severity_threshold = severity_threshold
        print("Conflict Detection Engine initialized.")

    def detect(self, state: SystemState) -> SystemState:
        """
        Identifies conflicts based on shared track sections and timing.
        This is a simplified version focusing on track proximity.
        """
        n_trains = len(state.train_positions)
        max_severity = 0.0

        for i in range(n_trains):
            for j in range(i + 1, n_trains):
                if state.train_tracks[i] == state.train_tracks[j]:
                    dist = np.abs(state.train_positions[i] - state.train_positions[j])
                    
                    # Predict future positions based on current speed and predicted delay
                    # A negative time_to_conflict means they are already too close
                    time_to_conflict = (dist - 25) / max(1, state.train_speeds[i] + state.train_speeds[j]) 
                    time_to_conflict -= (state.predicted_delays.get(i, 0) + state.predicted_delays.get(j, 0)) * 10 # Convert minutes to seconds

                    if time_to_conflict < self.time_buffer * 60:
                        severity = 1.0 - (time_to_conflict / (self.time_buffer * 60))
                        severity = min(1.0, max(0.0, severity))
                        max_severity = max(max_severity, severity)

                        if severity > self.severity_threshold:
                            state.conflicts.append({
                                'trains': [i, j],
                                'track': state.train_tracks[i],
                                'time_to_conflict': time_to_conflict,
                                'severity': severity
                            })
        
        state.conflict_severity = max_severity
        return state

# --- 4. Component 3: The Heavy Machinery - RL Resolution Engine ---
# This is your highly sophisticated, now delay-aware, PPO model.

class RLResolutionEngine:
    """The sophisticated PPO model, invoked only for severe, confirmed conflicts."""
    ACTION_MAP = {
        0: "Maintain Speed", 1: "Accelerate", 2: "Decelerate",
        3: "Emergency Brake", 4: "Hold", 5: "Switch Left",
        6: "Switch Right", 7: "Skip Station", 8: "Request Priority"
    }

    def __init__(self, model_path: str, config: Dict):
        print(f"Loading RL model from {model_path}...")
        
        # --- KEY ENHANCEMENT: Modify the RL environment's observation space ---
        self.original_features_per_train = self._count_features(config)
        config["observation_config"]["features_per_train"] = self.original_features_per_train + 3 # Add our new features
        
        self.env = RailEnv(config=config)
        self.model = PPO.load(model_path, env=self.env, device='cpu')
        print("RL Resolution Engine initialized and model is loaded.")

    def _count_features(self, config: Dict) -> int:
        """Helper to count original features based on config."""
        obs_config = config["observation_config"]
        base = 20
        if obs_config.get("include_network_state", False): base += 8
        if obs_config.get("include_predictive_features", False): base += 6
        if obs_config.get("include_neighbor_states", False): base += 10
        return base

    def _enhance_observation(self, obs: np.ndarray, state: SystemState) -> np.ndarray:
        """
        Injects delay and conflict data directly into the RL agent's observation.
        This makes the RL agent "delay-aware".
        """
        n_trains = self.env.n_trains
        
        # Reshape the flat observation array to (n_trains, features_per_train)
        obs_reshaped = obs.reshape((n_trains, self.original_features_per_train))
        
        # Create an array for our new features
        new_features = np.zeros((n_trains, 3))
        
        for i in range(n_trains):
            # Feature 1: Normalized predicted delay for this train
            new_features[i, 0] = min(1.0, state.predicted_delays.get(i, 0) / 10.0) # Normalize by 10 minutes
            # Feature 2: Overall system risk score
            new_features[i, 1] = state.risk_score
            # Feature 3: Max conflict severity involving this train
            max_sev = 0.0
            for conflict in state.conflicts:
                if i in conflict['trains']:
                    max_sev = max(max_sev, conflict['severity'])
            new_features[i, 2] = max_sev
            
        # Append new features and flatten back into a single vector
        enhanced_obs_reshaped = np.hstack([obs_reshaped, new_features])
        return enhanced_obs_reshaped.flatten().astype(np.float32)

    def resolve(self, state: SystemState, num_suggestions: int = 3) -> List[Dict]:
        """Uses the RL model to generate multiple, high-quality resolution plans."""
        # Set the internal environment to match the current system state
        self.env.tracks = state.train_tracks
        self.env.positions = state.train_positions
        self.env.speeds = state.train_speeds
        # You can set other env attributes here if needed (e.g., started, halt_remaining)

        # Get the base observation from the environment
        base_obs = self.env._get_enhanced_obs()

        # --- THIS IS THE CRITICAL INTEGRATION POINT ---
        enhanced_obs = self._enhance_observation(base_obs, state)

        # Use the model's probability distribution to get multiple suggestions (Beam Search)
        obs_tensor = torch.as_tensor([enhanced_obs], device=self.model.device)
        dist = self.model.policy.get_distribution(obs_tensor)
        action_probs = dist.distribution.probs.squeeze(0)

        beams = [(0.0, [])]
        for train_idx in range(self.env.n_trains):
            new_beams = []
            for log_prob, actions in beams:
                train_action_probs = action_probs[train_idx]
                top_k_probs, top_k_indices = torch.topk(train_action_probs, num_suggestions)
                for i in range(num_suggestions):
                    action = top_k_indices[i].item()
                    prob = top_k_probs[i].item()
                    new_beams.append((log_prob + np.log(prob), actions + [action]))
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:num_suggestions]

        # Annotate and return the best plans
        suggestions = []
        for i, (log_prob, plan) in enumerate(beams):
            suggestions.append({
                "plan_id": f"RL-Plan-{chr(65+i)}",
                "annotation": f"AI-generated resolution. Confidence: {np.exp(log_prob):.1%}",
                "actions": {f"train_{j}": self.ACTION_MAP.get(act, "Unknown") for j, act in enumerate(plan)}
            })
        return suggestions

# --- 5. The Gatekeeper: Intelligent Dispatcher ---
# This class orchestrates the entire process, implementing your architectural vision.

class IntelligentDispatcher:
    def __init__(self, rl_model_path: str, risk_threshold: float = 0.3, severity_threshold: float = 0.6):
        self.risk_threshold = risk_threshold
        self.severity_threshold = severity_threshold
        
        # Initialize all components
        self.predictor = LightweightDelayPredictor()
        self.conflict_detector = ConflictDetectionEngine()
        self.rl_engine = RLResolutionEngine(rl_model_path, copy.deepcopy(SIMPLIFIED_CONFIG))

        print("\n--- Intelligent Dispatcher is online. System ready. ---")
        
    def get_resolution_suggestions(self, current_state: SystemState) -> List[Dict]:
        """
        The main entry point. Implements the gatekeeper pattern to get suggestions.
        """
        print(f"\nTick {current_state.tick}: Analyzing system state...")
        
        # Stage 1: Always run the lightweight predictor
        state_with_delays = self.predictor.predict(current_state)
        print(f" -> Lightweight Predictor: Risk score = {state_with_delays.risk_score:.2f}")

        # Gate 1: Check if risk score exceeds the first threshold
        if state_with_delays.risk_score < self.risk_threshold:
            print(" -> Status: All clear. No escalation needed.")
            return [{"plan_id": "Nominal-A", "annotation": "System nominal. No action required.", "actions": {}}]
            
        # Stage 2: Escalate to the conflict detection engine
        print(" -> Risk threshold exceeded. Escalating to Conflict Detection Engine...")
        state_with_conflicts = self.conflict_detector.detect(state_with_delays)
        print(f" -> Conflict Detector: Max severity = {state_with_conflicts.conflict_severity:.2f}")
        
        # Gate 2: Check if conflict severity exceeds the second threshold
        if state_with_conflicts.conflict_severity < self.severity_threshold:
            print(" -> Status: Minor conflicts detected. Monitoring. No RL intervention needed.")
            return [{"plan_id": "Monitor-A", "annotation": "Low-severity conflict detected. Monitor situation.", "actions": {}}]

        # Stage 3: Escalate to the heavy machinery - the RL Resolution Engine
        print(" -> Severity threshold exceeded. ENGAGING RL RESOLUTION ENGINE...")
        suggestions = self.rl_engine.resolve(state_with_conflicts)
        print(f" -> RL Engine: Generated {len(suggestions)} resolution plans.")
        
        return suggestions

# --- Example Usage ---
if __name__ == '__main__':
    # Assume your best trained model is at this path
    FINAL_MODEL_PATH = "./best_model/20250912-112601/best_model.zip"

    # 1. Initialize the dispatcher. This loads all models and takes a few seconds.
    dispatcher = IntelligentDispatcher(FINAL_MODEL_PATH)
    
    # 2. In a real application, this state would come from your Konva.js frontend
    # Let's create a sample state for demonstration
    
    # SCENARIO 1: Everything is fine
    print("\n" + "="*50 + "\nSCENARIO 1: LOW RISK\n" + "="*50)
    low_risk_state = SystemState(
        tick=1,
        train_positions=np.array([100, 500, 800], dtype=np.float32),
        train_speeds=np.array([4, 4, 3], dtype=np.int32),
        train_tracks=np.array([0, 1, 2], dtype=np.int32)
    )
    suggestions_1 = dispatcher.get_resolution_suggestions(low_risk_state)
    print("Final Suggestions:", suggestions_1)

    # SCENARIO 2: High risk, but not yet severe conflict -> Monitor
    print("\n" + "="*50 + "\nSCENARIO 2: MEDIUM RISK\n" + "="*50)
    medium_risk_state = SystemState(
        tick=2,
        train_positions=np.array([100, 150, 800], dtype=np.float32), # Two trains are close
        train_speeds=np.array([4, 3, 3], dtype=np.int32),
        train_tracks=np.array([0, 0, 1], dtype=np.int32) # On the same track
    )
    # Manually set a high risk score to bypass the first gate for the demo
    medium_risk_state.risk_score = 0.4 
    suggestions_2 = dispatcher.get_resolution_suggestions(medium_risk_state)
    print("Final Suggestions:", suggestions_2)

    # SCENARIO 3: High risk and severe conflict -> Engage RL
    print("\n" + "="*50 + "\nSCENARIO 3: HIGH RISK (RL ENGAGED)\n" + "="*50)
    high_risk_state = SystemState(
        tick=3,
        train_positions=np.array([280, 310, 800, 820, 400, 500, 600, 700, 900, 1000], dtype=np.float32),
        train_speeds=np.array([4, 0, 4, 0, 3, 3, 3, 3, 3, 3], dtype=np.int32),
        train_tracks=np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 1], dtype=np.int32)
    )
    suggestions_3 = dispatcher.get_resolution_suggestions(high_risk_state)
    print("Final Suggestions:", suggestions_3)
