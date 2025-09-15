"""
Rail System Dispatcher - Main Orchestrator
Coordinates delay predictions, RL decisions, and frontend communication
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from collections import deque
import yaml
import os
from datetime import datetime, timedelta

# Import your existing modules
import sys
sys.path.append('./models')
from model.rail_env7 import RailEnv, SIMPLIFIED_CONFIG
# Updated import to use the new delay analyzer
sys.path.append(os.path.dirname(__file__))
from delaymodel import analyze_train_delays

from backend.state_manager import state_manager

# Import stable-baselines3 for model loading
from stable_baselines3 import PPO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainState:
    """Train state for frontend synchronization"""
    id: int
    track: int
    position: float
    speed: int
    status: str  # 'active', 'stopped', 'arrived', 'disabled'
    delay: float
    collision_risk: float
    next_station: Optional[str] = None
    signal_ahead: Optional[bool] = None
    
    def to_frontend_format(self) -> Dict:
        """Convert to frontend coordinate system"""
        # Map track to Y coordinate
        track_y_map = {0: 80, 1: 160, 2: 240, 3: 320}
        # Scale position from 0-1200 to 0-800px
        x_pos = (self.position / 1200.0) * 800
        
        return {
            "id": f"T{self.id}",
            "x": x_pos,
            "y": track_y_map.get(self.track, 80),
            "speed": self.speed * 30,  # Convert to px/s
            "status": self.status,
            "delay": self.delay,
            "riskLevel": self._get_risk_level(),
            "color": self._get_train_color()
        }
    
    def _get_risk_level(self) -> str:
        if self.collision_risk > 0.7:
            return "high"
        elif self.collision_risk > 0.3:
            return "medium"
        return "low"
    
    def _get_train_color(self) -> str:
        colors = ["#00bfff", "#ff7f50", "#ad8cf7", "#ffd24d", "#7fff00",
                  "#ff69b4", "#40e0d0", "#ffa500", "#9370db", "#87ceeb"]
        return colors[self.id % len(colors)]

@dataclass
class ActionSuggestion:
    """Action suggestion for frontend"""
    id: str
    train_id: int
    action_type: str
    reason: str
    priority: str  # 'critical', 'high', 'medium', 'low'
    predicted_outcome: str
    timestamp: float
    
    def to_frontend_format(self) -> Dict:
        action_names = {
            0: "Maintain Speed",
            1: "Accelerate",
            2: "Decelerate",
            3: "Emergency Brake",
            4: "Hold Position",
            5: "Switch Left Track",
            6: "Switch Right Track",
            7: "Skip Station",
            8: "Request Priority"
        }
        
        return {
            "id": self.id,
            "trainId": f"T{self.train_id}",
            "title": action_names.get(int(self.action_type.split('_')[1]), "Unknown Action"),
            "message": self.reason,
            "priority": self.priority,
            "outcome": self.predicted_outcome,
            "timestamp": self.timestamp
        }

class RailSystemDispatcher:
    """Main dispatcher coordinating all components"""
    
    def __init__(self, config_path: str = "./config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        self.last_results = None
        # Initialize environment
        self.env = RailEnv(config=SIMPLIFIED_CONFIG)
        self.obs, _ = self.env.reset()
        
        # Load trained model
        self.model = self._load_model()
        
        # State management
        self.train_states: List[TrainState] = []
        self.suggestions_queue: deque = deque(maxlen=20)
        self.pending_actions: Dict[str, Any] = {}
        self.accepted_actions: List[int] = []
        
        # Performance metrics
        self.metrics = {
            "conflicts_prevented": 0,
            "delays_mitigated": 0,
            "on_time_arrivals": 0,
            "total_delays": 0.0,
            "system_efficiency": 1.0,
            "last_update": time.time()
        }
        
        # Updated delay predictions structure to store full analyzer results
        self.delay_predictions: Dict[str, Any] = {}
        self.last_delay_update = 0
        
        # WebSocket connections
        self.ws_connections = set()
        
        logger.info("Dispatcher initialized successfully")
    

    def _load_config(self, path="./config.yaml"):
        """Load system configuration and normalize paths relative to project root."""
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            config = {
                "model_path": "../model/best_model/20250912-112601/best_model.zip",
                "delay_update_interval": 30,
                "suggestion_threshold": 0.3,
                "ws_port": 8765,
                "api_port": 8000,
                "update_rate": 5,
            }

        #Always resolve model_path relative to this file's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = config.get("model_path")
        if model_path:
            config["model_path"] = os.path.abspath(os.path.join(base_dir, model_path))

        return config

    def _load_model(self):
        try:
            model_path = self.config["model_path"]
            logger.info(f"Trying to load RL model from absolute path: {model_path}")

            if not os.path.exists(model_path):
                logger.warning(f"No RL model found at {model_path}, using heuristics")
                return None

            model = PPO.load(model_path, env=self.env)
            logger.info(f"Successfully loaded model from {model_path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Running without RL model - using heuristics")
            return None

    
    
    def _collect_current_schedule_data(self) -> List[Dict[str, Any]]:
        """
        Collect current train schedule/state data for delay analysis.
        
        Returns:
            List of dicts with train_id and delay information
        """
        schedule_data = []
        
        for i in range(self.env.n_trains):
            # Only include active trains (started but not arrived)
            if self.env.started[i] and not self.env.arrived[i] and not self.env.disabled[i]:
                current_delay = self.env._compute_current_delay(i)
                
                schedule_data.append({
                    'train_id': f"train_{i}",  # Use consistent train naming
                    'delay': max(0.0, float(current_delay)),  # Ensure non-negative delay
                    'position': float(self.env.positions[i]),
                    'speed': int(self.env.speeds[i]),
                    'track': int(self.env.tracks[i])
                })
        
        return schedule_data

    def _update_delay_predictions_with_analyzer(self):
        """
        Update delay predictions using the new delay analyzer after each simulation step.
        """
        try:
            # Collect current schedule data
            schedule_data = self._collect_current_schedule_data()
            
            if not schedule_data:
                logger.debug("No active trains found for delay analysis")
                self.delay_predictions = {}
                return
            
            # Analyze delays using the new analyzer
            analysis_results = analyze_train_delays(
                schedule_data,
                outlier_threshold=2.0,
                delay_threshold=1.0
            )
            
            # Store the full results for frontend
            self.delay_predictions = analysis_results
            
            # Log predictions for debugging
            print("=" * 50)
            print("DELAY ANALYSIS UPDATE:")
            print(json.dumps(self.delay_predictions, indent=2))
            print("=" * 50)
            
            logger.info(f"Delay predictions updated - Mean: {analysis_results.get('mean_delay', 0):.2f}min, "
                       f"Active trains: {analysis_results.get('total_trains', 0)}, "
                       f"Delayed: {analysis_results.get('delayed_trains', 0)}")
            
        except Exception as e:
            logger.error(f"Error updating delay predictions with analyzer: {e}")
            # Fallback to empty predictions on error
            self.delay_predictions = {}

    async def update_delay_predictions(self):
        """Update delay predictions using Monte Carlo model (legacy method - keeping for compatibility)"""
        current_time = time.time()
        
        # Update every 30 seconds
        if current_time - self.last_delay_update < self.config.get("delay_update_interval", 30):
            return
        
        try:
            # Use the new analyzer-based approach
            self._update_delay_predictions_with_analyzer()
            
            # Check for potential conflicts based on delay predictions
            conflicts = self._check_future_conflicts()
            if conflicts:
                await self._generate_conflict_avoidance_suggestions(conflicts)
            
            self.last_delay_update = current_time
            
        except Exception as e:
            logger.error(f"Error in legacy delay prediction update: {e}")
    
    def _prepare_schedule_data(self) -> Dict:
        """Prepare current schedule data for delay model (legacy method)"""
        schedule = []
        for i in range(self.env.n_trains):
            if self.env.started[i] and not self.env.arrived[i]:
                schedule.append({
                    'train_id': chr(ord('a') + (i % 3)),
                    'delay': max(0, self.env._compute_current_delay(i))
                })
        return schedule
    
    def _check_future_conflicts(self) -> List[Dict]:
        """Check for potential future conflicts based on predictions"""
        conflicts = []
        
        # Use delay probabilities from the analyzer if available
        delay_probs = self.delay_predictions.get('delay_probabilities', {})
        
        for i in range(self.env.n_trains):
            if not self.env.started[i] or self.env.arrived[i] or self.env.disabled[i]:
                continue
            
            # Get predicted delay probability for this train
            train_key = f"train_{i}"
            delay_prob = delay_probs.get(train_key, 0.0)
            future_position = self.env.positions[i] + (self.env.speeds[i] * 10)  # 10 steps ahead
            
            for j in range(i + 1, self.env.n_trains):
                if not self.env.started[j] or self.env.arrived[j] or self.env.disabled[j]:
                    continue
                if self.env.tracks[i] != self.env.tracks[j]:
                    continue
                
                future_j = self.env.positions[j] + (self.env.speeds[j] * 10)
                distance = abs(future_position - future_j)
                
                # Factor in delay probability for conflict severity
                conflict_severity = "high" if (distance < 50 and delay_prob > 0.5) else "medium"
                
                if distance < self.env.cfg["reward_config"]["safety"]["min_safe_distance"] * 2:
                    conflicts.append({
                        "train1": i,
                        "train2": j,
                        "distance": distance,
                        "time_to_conflict": 10,
                        "severity": conflict_severity,
                        "delay_factor": delay_prob
                    })
        
        return conflicts
    
    async def _generate_conflict_avoidance_suggestions(self, conflicts: List[Dict]):
        """Generate action suggestions to avoid conflicts"""
        for conflict in conflicts:
            train1, train2 = conflict["train1"], conflict["train2"]
            
            # Determine which train should take action
            if self.env.speeds[train1] > self.env.speeds[train2]:
                action_train = train1
                action = 2  # Decelerate
                reason = f"Slow down to avoid conflict with Train {train2}"
            else:
                action_train = train2
                action = 1  # Accelerate if safe
                reason = f"Speed up to clear path for Train {train1}"
            
            # Check if switch is better option
            if self._can_switch_track(action_train):
                action = 5 if self.env.tracks[action_train] > 0 else 6
                reason = f"Switch track to avoid conflict with Train {train1 if action_train == train2 else train2}"
            
            suggestion = ActionSuggestion(
                id=f"sug_{int(time.time()*1000)}_{action_train}",
                train_id=action_train,
                action_type=f"action_{action}",
                reason=reason,
                priority=conflict["severity"],
                predicted_outcome=f"Prevents collision in {conflict['time_to_conflict']} steps",
                timestamp=time.time()
            )
            
            self.suggestions_queue.append(suggestion)
    
    def _can_switch_track(self, train_idx: int) -> bool:
        """Check if train can safely switch tracks"""
        return self.env._is_in_switch_zone(train_idx) and self.env.speeds[train_idx] <= 1
    
    async def get_rl_actions(self) -> np.ndarray:
        """Get actions from RL model with delay-aware observations"""
        if self.model is None:
            # Fallback to heuristic actions
            return self._get_heuristic_actions()
        
        # Enhance observations with delay predictions
        enhanced_obs = self._enhance_observations_with_delays()
        
        # Get model predictions
        actions, _ = self.model.predict(enhanced_obs, deterministic=True)
        
        return actions
    
    def _enhance_observations_with_delays(self) -> np.ndarray:
        """Add delay predictions to observations"""
        obs = self.obs.copy()
        
        # Inject delay predictions into relevant observation indices
        # This assumes the observation structure from rail_env7
        features_per_train = self.env.features_per_train
        delay_probs = self.delay_predictions.get('delay_probabilities', {})
        
        for i in range(self.env.n_trains):
            train_key = f"train_{i}"
            if train_key in delay_probs:
                # Find delay feature index for this train
                delay_idx = i * features_per_train + 13  # Delay tracking index
                if delay_idx < len(obs):
                    # Use delay probability as normalized feature
                    obs[delay_idx] = min(1.0, delay_probs[train_key])
        
        return obs
    
    def _get_heuristic_actions(self) -> np.ndarray:
        """Fallback heuristic action generation"""
        actions = np.zeros(self.env.n_trains, dtype=np.int32)
        
        for i in range(self.env.n_trains):
            if not self.env.started[i] or self.env.arrived[i] or self.env.disabled[i]:
                actions[i] = 0  # No-op
                continue
            
            # Simple heuristics
            collision_risk = self.env._compute_collision_risk(i)
            
            if collision_risk > 0.7:
                actions[i] = 3  # Emergency brake
            elif collision_risk > 0.3:
                actions[i] = 2  # Decelerate
            elif self.env._is_in_any_station(i):
                actions[i] = 4  # Hold
            elif self.env.speeds[i] < 2:
                actions[i] = 1  # Accelerate
            else:
                actions[i] = 0  # Maintain
        
        return actions
    
    async def step_simulation(self):
        """Main simulation step with integrated decision making"""
        # Update delay predictions (legacy periodic update)
        await self.update_delay_predictions()
        
        # Get RL actions
        base_actions = await self.get_rl_actions()
        
        # Apply user-accepted actions (override RL if user accepted)
        final_actions = self._merge_user_actions(base_actions)
        
        # Step environment
        self.obs, reward, terminated, truncated, info = self.env.step(final_actions)
        
        # Update train states
        self._update_train_states()
        
        # *** NEW: Update delay predictions after each simulation step ***
        self._update_delay_predictions_with_analyzer()
        
        # Generate new suggestions if needed
        await self._generate_suggestions(final_actions, info)
        
        # Update metrics
        self._update_metrics(info)
        
        # Broadcast state to frontend (includes delay predictions)
        self._broadcast_state_update()
        
        return terminated or truncated
    
    def _merge_user_actions(self, base_actions: np.ndarray) -> np.ndarray:
        """Merge user-accepted actions with RL recommendations"""
        final_actions = base_actions.copy()
        
        # Apply accepted actions
        for action_data in self.accepted_actions:
            train_id = action_data.get("train_id")
            action = action_data.get("action")
            if train_id < len(final_actions):
                final_actions[train_id] = action
        
        # Clear processed actions
        self.accepted_actions.clear()
        
        return final_actions
    
    def _update_train_states(self):
        """Update train states for frontend"""
        self.train_states.clear()
        
        for i in range(self.env.n_trains):
            status = "active"
            if self.env.disabled[i]:
                status = "disabled"
            elif self.env.arrived[i]:
                status = "arrived"
            elif not self.env.started[i]:
                status = "waiting"
            elif self.env.speeds[i] == 0:
                status = "stopped"
            
            state = TrainState(
                id=i,
                track=int(self.env.tracks[i]),
                position=float(self.env.positions[i]),
                speed=int(self.env.speeds[i]),
                status=status,
                delay=float(self.env._compute_current_delay(i)),
                collision_risk=float(self.env.collision_risks[i])
            )
            self.train_states.append(state)
    
    async def _generate_suggestions(self, actions: np.ndarray, info: Dict):
        """Generate action suggestions for critical situations"""
        threshold = self.config.get("suggestion_threshold", 0.3)
        
        for i in range(self.env.n_trains):
            if not self.env.started[i] or self.env.arrived[i] or self.env.disabled[i]:
                continue
            
            # Check if intervention needed
            risk = self.env.collision_risks[i]
            delay = self.env._compute_current_delay(i)
            
            if risk > threshold or delay > 50:
                # Generate suggestion
                if risk > 0.7:
                    priority = "critical"
                    action = 3  # Emergency brake
                    reason = "Imminent collision risk detected"
                elif risk > threshold:
                    priority = "high"
                    action = 2  # Decelerate
                    reason = "High collision risk - recommend slowing down"
                elif delay > 100:
                    priority = "medium"
                    action = 8  # Request priority
                    reason = f"Train is {delay} steps delayed - request priority"
                else:
                    continue
                
                suggestion = ActionSuggestion(
                    id=f"sug_{int(time.time()*1000)}_{i}",
                    train_id=i,
                    action_type=f"action_{action}",
                    reason=reason,
                    priority=priority,
                    predicted_outcome=self._predict_outcome(i, action),
                    timestamp=time.time()
                )
                
                # Only add if not duplicate
                if not self._is_duplicate_suggestion(suggestion):
                    self.suggestions_queue.append(suggestion)
    
    def _predict_outcome(self, train_id: int, action: int) -> str:
        """Predict outcome of an action"""
        outcomes = {
            1: "Will increase speed and reduce delay",
            2: "Will reduce collision risk by 40%",
            3: "Will prevent imminent collision",
            4: "Will maintain safe distance",
            5: "Will switch to less congested track",
            6: "Will switch to alternative track",
            8: "Will get priority at next junction"
        }
        return outcomes.get(action, "Will improve situation")
    
    def _is_duplicate_suggestion(self, suggestion: ActionSuggestion) -> bool:
        """Check if suggestion is duplicate"""
        for existing in self.suggestions_queue:
            if (existing.train_id == suggestion.train_id and 
                existing.action_type == suggestion.action_type and
                time.time() - existing.timestamp < 10):  # Within 10 seconds
                return True
        return False
    
   
    
    def _update_metrics(self, info: Dict):
        """Update performance metrics"""
        if "collision_count" in info:
            self.metrics["conflicts_prevented"] = max(0, 10 - info["collision_count"])
        
        if "episode_summary" in info:
            summary = info["episode_summary"]
            self.metrics["on_time_arrivals"] = summary.get("on_time_rate", 0) * 100
            self.metrics["total_delays"] = summary.get("total_delay", 0)
            self.metrics["system_efficiency"] = summary.get("efficiency_score", 0)
        
        self.metrics["last_update"] = time.time()
    
    async def _broadcast_state_update(self):
        """Broadcast current state to all WebSocket connections"""
        state_update = {
            "type": "state_update",
            "data": {
                "trains": [t.to_frontend_format() for t in self.train_states],
                "signals": self._get_signal_states(),
                "suggestions": [s.to_frontend_format() for s in list(self.suggestions_queue)[-5:]],
                "metrics": self.metrics,
                "timestamp": time.time()
            }
        }
        
        # This will be sent via WebSocket server
        return state_update
    
    def _get_signal_states(self) -> Dict:
        """Get signal states for frontend"""
        signals = {}
        for i, (pos, state) in enumerate(self.env.signal_states.items()):
            # Map to frontend signal IDs
            signals[i + 1] = state  # True = red, False = green
        return signals
    
    def _broadcast_state_update(self):
        """Push current state to the StateManager for WebSocket broadcast."""
        state_manager.bulk_update({
            "step": self.env.current_step,
            "trains": [t.to_frontend_format() for t in self.train_states],
            "signals": self.env.signal_states,
            "metrics": self.metrics,
            "suggestions": [s.to_frontend_format() for s in self.suggestions_queue],
            # *** UPDATED: Include full delay predictions from analyzer ***
            "delay_predictions": self.delay_predictions,
            "network": {
                "congestion": self.env.network_congestion,
                "throughput": self.env.system_throughput,
                "cascade_factor": self.env.cascade_delay_factor
            }
        })
    
    def accept_action(self, suggestion_id: str, train_id: str):
        """Accept a suggested action from frontend"""
        # Find the suggestion
        for suggestion in self.suggestions_queue:
            if suggestion.id == suggestion_id:
                action_type = int(suggestion.action_type.split('_')[1])
                train_num = int(train_id[1:])  # Remove 'T' prefix
                
                self.accepted_actions.append({
                    "train_id": train_num,
                    "action": action_type,
                    "timestamp": time.time()
                })
                
                logger.info(f"Action accepted: Train {train_num} -> Action {action_type}")
                return True
        
        return False
    
    def reject_action(self, suggestion_id: str):
        """Reject a suggested action"""
        # Remove from queue
        self.suggestions_queue = deque(
            [s for s in self.suggestions_queue if s.id != suggestion_id],
            maxlen=20
        )
        logger.info(f"Action rejected: {suggestion_id}")

    
    def convert_schedule_to_delays(self, schedule):
        """
        Convert schedule data (arrival_time + actual_arrival) 
        into the format expected by analyze_train_delays().
        """
        converted = []
        for record in schedule:
            planned = record.get("arrival_time")
            actual = record.get("actual_arrival")
            if planned is None or actual is None:
                continue  # skip incomplete records
            delay = max(0, actual - planned)  # negative means early, set to 0
            converted.append({
                "train_id": record.get("train_id", "UNKNOWN"),
                "delay": delay
            })
        return converted
    
    def get_schedule(self):
        """Dummy schedule — replace later with real API/DB fetch."""
        return [
            {"train_id": "T1", "arrival_time": 10, "actual_arrival": 12},
            {"train_id": "T2", "arrival_time": 15, "actual_arrival": 15},
            {"train_id": "T3", "arrival_time": 20, "actual_arrival": 23},
            {"train_id": "T4", "arrival_time": 25, "actual_arrival": 24},
        ]

    def log_results(self, results):
        print("\n=== Updated Delay Analysis ===")
        print(json.dumps(results, indent=2))
    
    async def run(self):
        refresh_interval = self.config.get("refresh_interval", 5.0)

        while True:
            try:
                # 1. Fetch schedule (dummy for now)
                schedule_data = self.get_schedule()

                # 2. Convert -> analyze
                converted = self.convert_schedule_to_delays(schedule_data)
                results = analyze_train_delays(converted)

                # 3. Only log if something changed
                results_json = json.dumps(results, sort_keys=True)
                if results_json != self.last_results:
                    self.log_results(results)
                    self.last_results = results_json

            except Exception as e:
                print(f"[Dispatcher] Error: {e}")

            await asyncio.sleep(refresh_interval)

    

# Entry point
if __name__ == "__main__":
    dispatcher = RailSystemDispatcher()
    asyncio.run(dispatcher.run())
    