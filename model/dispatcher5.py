"""
Enhanced Rail System Dispatcher - Pre-trained Model Version
Modified to use a pre-trained delay prediction model instead of integrated delay analysis
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from collections import deque
import yaml
import os
from datetime import datetime, timedelta
import joblib  # For loading pre-trained models
import pickle

# Import your existing modules
import sys
sys.path.append('./models')
from model.rail_env7 import RailEnv, SIMPLIFIED_CONFIG
sys.path.append(os.path.dirname(__file__))

from backend.state_manager import state_manager

# Import stable-baselines3 for RL model loading
from stable_baselines3 import PPO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DelayAwareTrainState:
    """Enhanced train state with comprehensive delay information"""
    id: int
    track: int
    position: float
    speed: int
    status: str  # 'active', 'stopped', 'arrived', 'disabled'
    
    # Delay-specific fields
    current_delay: float
    predicted_delay: float
    delay_probability: float
    delay_trend: str  # 'improving', 'stable', 'worsening'
    
    # Safety fields
    collision_risk: float
    time_to_collision: float
    
    # Schedule fields
    next_station: Optional[str] = None
    signal_ahead: Optional[bool] = None
    eta_deviation: float = 0.0
    schedule_pressure: float = 0.0
    
    def to_frontend_format(self) -> Dict:
        """Convert to frontend coordinate system with delay information"""
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
            
            # Delay information
            "delay": self.current_delay,
            "predictedDelay": self.predicted_delay,
            "delayProbability": self.delay_probability,
            "delayTrend": self.delay_trend,
            
            # Risk information
            "riskLevel": self._get_risk_level(),
            "timeToCollision": self.time_to_collision,
            
            # Schedule information
            "etaDeviation": self.eta_deviation,
            "schedulePressure": self.schedule_pressure,
            
            # Visual
            "color": self._get_train_color(),
            "delayColor": self._get_delay_color()
        }
    
    def _get_risk_level(self) -> str:
        if self.collision_risk > 0.7:
            return "critical"
        elif self.collision_risk > 0.5:
            return "high"
        elif self.collision_risk > 0.3:
            return "medium"
        return "low"
    
    def _get_train_color(self) -> str:
        colors = ["#00bfff", "#ff7f50", "#ad8cf7", "#ffd24d", "#7fff00",
                  "#ff69b4", "#40e0d0", "#ffa500", "#9370db", "#87ceeb"]
        return colors[self.id % len(colors)]
    
    def _get_delay_color(self) -> str:
        """Color based on delay status"""
        if self.current_delay > 10:
            return "#ff4444"  # Red for high delay
        elif self.current_delay > 5:
            return "#ffaa44"  # Orange for moderate delay
        elif self.current_delay > 0:
            return "#ffff44"  # Yellow for minor delay
        else:
            return "#44ff44"  # Green for on-time

@dataclass
class DelayAwareActionSuggestion:
    """Enhanced action suggestion with delay-aware reasoning"""
    id: str
    train_id: int
    action_type: str
    reason: str
    priority: str  # 'critical', 'high', 'medium', 'low'
    predicted_outcome: str
    timestamp: float
    
    # Delay-specific fields
    delay_impact: float  # How much this action affects delay
    cascade_prevention: bool  # Whether this prevents cascade delays
    schedule_recovery: bool  # Whether this helps recover schedule
    
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
            "timestamp": self.timestamp,
            
            # Delay-specific frontend fields
            "delayImpact": self.delay_impact,
            "cascadePrevention": self.cascade_prevention,
            "scheduleRecovery": self.schedule_recovery,
            "delayReduction": max(0, -self.delay_impact)  # Positive if reduces delay
        }

class PretrainedDelayPredictor:
    """Wrapper for pre-trained delay prediction model"""
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained delay prediction model"""
        try:
            # Try different formats for loading the model
            if self.model_path.endswith('.pkl') or self.model_path.endswith('.pickle'):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            elif self.model_path.endswith('.joblib'):
                self.model = joblib.load(self.model_path)
            else:
                # Try joblib first, then pickle
                try:
                    self.model = joblib.load(self.model_path)
                except:
                    with open(self.model_path, 'rb') as f:
                        self.model = pickle.load(f)
            
            # Load scaler if provided
            if self.scaler_path and os.path.exists(self.scaler_path):
                if self.scaler_path.endswith('.joblib'):
                    self.scaler = joblib.load(self.scaler_path)
                else:
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
            
            logger.info(f"Successfully loaded delay prediction model from {self.model_path}")
            if self.scaler:
                logger.info(f"Successfully loaded scaler from {self.scaler_path}")
                
        except Exception as e:
            logger.error(f"Failed to load delay prediction model: {e}")
            self.model = None
            self.scaler = None
    
    def predict(self, train_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make delay predictions using the pre-trained model
        
        Args:
            train_features: List of dictionaries containing train features
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            logger.warning("No model loaded, returning default predictions")
            return self._get_default_predictions(train_features)
        
        try:
            # Convert features to the format expected by your model
            feature_matrix = self._prepare_features(train_features)
            
            # Make predictions
            if self.scaler:
                feature_matrix = self.scaler.transform(feature_matrix)
            
            predictions = self.model.predict(feature_matrix)
            
            # Convert predictions to the expected format
            return self._format_predictions(predictions, train_features)
            
        except Exception as e:
            logger.error(f"Error making delay predictions: {e}")
            return self._get_default_predictions(train_features)
    
    def _prepare_features(self, train_features: List[Dict[str, Any]]) -> np.ndarray:
        """Convert train features to model input format"""
        # This depends on how your model was trained
        # Adjust feature extraction based on your model's expected input
        
        feature_vectors = []
        for train in train_features:
            # Extract relevant features - modify based on your model
            features = [
                train.get('delay', 0.0),
                train.get('position', 0.0),
                train.get('speed', 0.0),
                train.get('track', 0.0),
                train.get('collision_risk', 0.0),
                train.get('eta_deviation', 0.0),
                train.get('schedule_pressure', 0.0)
            ]
            feature_vectors.append(features)
        
        return np.array(feature_vectors)
    
    def _format_predictions(self, predictions: np.ndarray, train_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format model predictions into expected output format"""
        delay_probabilities = {}
        future_delays = {}
        delay_trends = {}
        
        for i, (pred, train) in enumerate(zip(predictions, train_features)):
            train_id = train.get('train_id', f'train_{i}')
            
            # Interpret predictions based on your model's output
            if isinstance(pred, (list, np.ndarray)):
                # Multi-output model
                delay_prob = float(pred[0]) if len(pred) > 0 else 0.0
                future_delay = float(pred[1]) if len(pred) > 1 else train.get('delay', 0.0)
            else:
                # Single output (delay probability or future delay)
                delay_prob = float(pred)
                current_delay = train.get('delay', 0.0)
                future_delay = current_delay + (delay_prob * 10)  # Simple mapping
            
            delay_probabilities[train_id] = min(1.0, max(0.0, delay_prob))
            future_delays[train_id] = max(0.0, future_delay)
            
            # Determine trend based on current vs predicted delay
            current_delay = train.get('delay', 0.0)
            if future_delay > current_delay + 2:
                delay_trends[train_id] = "worsening"
            elif future_delay < current_delay - 2:
                delay_trends[train_id] = "improving"
            else:
                delay_trends[train_id] = "stable"
        
        # Calculate aggregate metrics
        delays = [train.get('delay', 0.0) for train in train_features]
        mean_delay = np.mean(delays) if delays else 0.0
        delay_std = np.std(delays) if len(delays) > 1 else 0.0
        
        return {
            'delay_probabilities': delay_probabilities,
            'future_delays': future_delays,
            'delay_trends': delay_trends,
            'mean_delay': float(mean_delay),
            'delay_std': float(delay_std),
            'prediction_timestamp': time.time()
        }
    
    def _get_default_predictions(self, train_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return default predictions when model is not available"""
        delay_probabilities = {}
        future_delays = {}
        delay_trends = {}
        
        for i, train in enumerate(train_features):
            train_id = train.get('train_id', f'train_{i}')
            current_delay = train.get('delay', 0.0)
            collision_risk = train.get('collision_risk', 0.0)
            
            # Simple heuristic-based predictions
            delay_prob = min(0.8, collision_risk + (current_delay / 30))
            future_delay = current_delay + (collision_risk * 5)
            
            delay_probabilities[train_id] = delay_prob
            future_delays[train_id] = future_delay
            delay_trends[train_id] = "stable"
        
        delays = [train.get('delay', 0.0) for train in train_features]
        return {
            'delay_probabilities': delay_probabilities,
            'future_delays': future_delays,
            'delay_trends': delay_trends,
            'mean_delay': float(np.mean(delays)) if delays else 0.0,
            'delay_std': float(np.std(delays)) if len(delays) > 1 else 0.0,
            'prediction_timestamp': time.time()
        }

class DelayAwareDispatcher:
    """Enhanced dispatcher using pre-trained delay prediction model"""
    
    def __init__(self, config_path: str = "./config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize environment
        self.env = RailEnv(config=SIMPLIFIED_CONFIG)
        self.obs, _ = self.env.reset()
        
        # Load trained RL model
        self.model = self._load_model()
        
        # Initialize pre-trained delay predictor
        self.delay_predictor = self._load_delay_predictor()
        
        # Enhanced state management
        self.train_states: List[DelayAwareTrainState] = []
        self.suggestions_queue: deque = deque(maxlen=20)
        self.pending_actions: Dict[str, Any] = {}
        self.accepted_actions: List[int] = []
        
        # Delay tracking (simplified - no internal analysis)
        self.delay_predictions: Dict[str, Any] = {}
        self.delay_history: Dict[int, deque] = {i: deque(maxlen=10) for i in range(self.env.n_trains)}
        self.cascade_delays: Dict[int, float] = {}
        self.last_delay_update = 0
        
        # Performance metrics with delay awareness
        self.metrics = {
            "conflicts_prevented": 0,
            "delays_mitigated": 0,
            "cascade_delays_prevented": 0,
            "schedule_recovery_actions": 0,
            "on_time_arrivals": 0,
            "total_delays": 0.0,
            "avg_delay_per_train": 0.0,
            "system_efficiency": 1.0,
            "delay_prediction_accuracy": 0.0,
            "last_update": time.time()
        }
        
        # WebSocket connections
        self.ws_connections = set()
        
        logger.info("Delay-aware dispatcher with pre-trained model initialized successfully")
    
    def _load_config(self, path="./config.yaml"):
        """Load system configuration"""
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            config = {
                "model_path": "../model/best_model/20250912-112601/best_model.zip",
                "delay_model_path": "../model/delaymodel.joblib",  # NEW: Path to delay model
                "delay_scaler_path": "../model/delay_scaler.joblib",  # NEW: Path to feature scaler
                "delay_update_interval": 15,  # Less frequent since using pre-trained model
                "suggestion_threshold": 0.2,
                "delay_prediction_horizon": 50,
                "cascade_sensitivity": 0.3,
                "ws_port": 8765,
                "api_port": 8000,
                "update_rate": 2,
            }

        # Resolve paths relative to this file's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for path_key in ["model_path", "delay_model_path", "delay_scaler_path"]:
            if path_key in config and config[path_key]:
                config[path_key] = os.path.abspath(os.path.join(base_dir, config[path_key]))

        return config
    
    def _load_model(self):
        """Load RL model"""
        try:
            model_path = self.config["model_path"]
            logger.info(f"Loading RL model from: {model_path}")

            if not os.path.exists(model_path):
                logger.warning(f"No RL model found at {model_path}, using heuristics")
                return None

            model = PPO.load(model_path, env=self.env)
            logger.info(f"Successfully loaded RL model from {model_path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
            return None
    
    def _load_delay_predictor(self) -> PretrainedDelayPredictor:
        """Load pre-trained delay prediction model"""
        delay_model_path = self.config.get("delay_model_path")
        delay_scaler_path = self.config.get("delay_scaler_path")
        
        if not delay_model_path:
            logger.warning("No delay model path specified, using default predictions")
            return PretrainedDelayPredictor("", None)  # Will use defaults
        
        return PretrainedDelayPredictor(delay_model_path, delay_scaler_path)
    
    def _collect_train_features_for_prediction(self) -> List[Dict[str, Any]]:
        """
        Collect train features for delay prediction model
        """
        train_features = []
        
        for i in range(self.env.n_trains):
            # Only include active trains
            if self.env.started[i] and not self.env.disabled[i]:
                current_delay = self.env._compute_current_delay(i)
                
                # Update delay history
                self.delay_history[i].append(current_delay)
                
                train_features.append({
                    'train_id': f"train_{i}",
                    'delay': max(0.0, float(current_delay)),
                    'position': float(self.env.positions[i]),
                    'speed': int(self.env.speeds[i]),
                    'track': int(self.env.tracks[i]),
                    'collision_risk': float(self.env.collision_risks[i]),
                    'eta_deviation': self._calculate_eta_deviation(i),
                    'schedule_pressure': self._calculate_schedule_pressure(i)
                })
        
        return train_features
    
    def _update_delay_predictions_with_pretrained_model(self):
        """
        Update delay predictions using the pre-trained model
        """
        try:
            # Collect train features
            train_features = self._collect_train_features_for_prediction()
            
            if not train_features:
                logger.debug("No active trains found for delay prediction")
                self.delay_predictions = {}
                return
            
            # Use pre-trained model for predictions
            predictions = self.delay_predictor.predict(train_features)
            
            # Enhance with additional system-level analysis
            enhanced_predictions = self._enhance_with_system_analysis(predictions, train_features)
            
            # Store results
            self.delay_predictions = enhanced_predictions
            
            # Update cascade delay tracking
            self._update_cascade_delays(enhanced_predictions)
            
            logger.info(f"Delay predictions updated using pre-trained model - "
                       f"Mean: {enhanced_predictions.get('mean_delay', 0):.2f}min")
            
        except Exception as e:
            logger.error(f"Error in pre-trained model delay prediction: {e}")
            self.delay_predictions = {}
    
    def _enhance_with_system_analysis(self, base_predictions: Dict, train_features: List[Dict]) -> Dict:
        """Add system-level analysis to model predictions"""
        enhanced = base_predictions.copy()
        
        # Calculate system-level metrics
        enhanced.update({
            'system_delay_pressure': self._calculate_system_delay_pressure(train_features),
            'cascade_risk_level': self._assess_cascade_risk(train_features),
            'prediction_horizon_steps': self.config.get("delay_prediction_horizon", 50)
        })
        
        return enhanced
    
    def _calculate_eta_deviation(self, train_id: int) -> float:
        """Calculate how far off the train is from expected arrival time"""
        if self.env.arrived[train_id]:
            return 0.0
        
        remaining_dist = self.env.destinations[train_id] - self.env.positions[train_id]
        if remaining_dist <= 0:
            return 0.0
        
        current_speed = max(1, self.env.speeds[train_id])
        estimated_remaining_time = remaining_dist / current_speed
        planned_remaining = max(0, self.env.planned_arrival[train_id] - self.env.current_step)
        
        return float(estimated_remaining_time - planned_remaining)
    
    def _calculate_schedule_pressure(self, train_id: int) -> float:
        """Calculate pressure from schedule constraints (0-1 scale)"""
        if self.env.arrived[train_id]:
            return 0.0
        
        current_delay = self.env._compute_current_delay(train_id)
        time_remaining = max(1, self.env.planned_arrival[train_id] - self.env.current_step)
        
        pressure = min(1.0, current_delay / time_remaining)
        return float(pressure)
    
    def _calculate_system_delay_pressure(self, train_features: List[Dict]) -> float:
        """Calculate overall system pressure from delays"""
        if not train_features:
            return 0.0
        
        total_pressure = sum(feature.get('schedule_pressure', 0) for feature in train_features)
        return min(1.0, total_pressure / len(train_features))
    
    def _assess_cascade_risk(self, train_features: List[Dict]) -> str:
        """Assess risk level of cascade delays"""
        high_delay_trains = sum(1 for feature in train_features if feature['delay'] > 10)
        total_trains = len(train_features)
        
        if total_trains == 0:
            return "none"
        
        risk_ratio = high_delay_trains / total_trains
        
        if risk_ratio > 0.5:
            return "high"
        elif risk_ratio > 0.25:
            return "medium"
        elif risk_ratio > 0:
            return "low"
        else:
            return "none"
    
    def _update_cascade_delays(self, predictions: Dict):
        """Track and update cascade delay effects"""
        cascade_sensitivity = self.config.get("cascade_sensitivity", 0.3)
        
        delay_probs = predictions.get('delay_probabilities', {})
        future_delays = predictions.get('future_delays', {})
        
        self.cascade_delays.clear()
        
        # Calculate cascade effects using predictions
        for i in range(self.env.n_trains):
            train_key = f"train_{i}"
            
            if train_key in delay_probs and delay_probs[train_key] > cascade_sensitivity:
                cascade_factor = delay_probs[train_key] * future_delays.get(train_key, 0)
                
                # Affect downstream trains
                for j in range(i + 1, min(i + 4, self.env.n_trains)):
                    if self.env.started[j] and not self.env.arrived[j]:
                        existing_cascade = self.cascade_delays.get(j, 0)
                        new_cascade = cascade_factor * 0.3 * (1 / (j - i))
                        self.cascade_delays[j] = existing_cascade + new_cascade

    # REST OF THE CLASS METHODS REMAIN THE SAME
    # (get_delay_aware_actions, _generate_delay_aware_suggestions, step, etc.)
    # Just update the delay prediction call in the step method:
    
    async def step(self):
        """Enhanced main simulation step using pre-trained model"""
        try:
            # Update delay predictions using pre-trained model
            self.suggestions_queue.clear()
            current_time = time.time()
            if current_time - self.last_delay_update >= self.config.get("delay_update_interval", 15):
                self._update_delay_predictions_with_pretrained_model()  # CHANGED THIS LINE
                self.last_delay_update = current_time
            
            # Rest of the step method remains the same...
            actions = await self.get_delay_aware_actions()
            final_actions = self._apply_accepted_actions(actions)
            self.obs, rewards, dones, truncated, info = self.env.step(final_actions)
            self._update_enhanced_train_states()
            await self._generate_delay_aware_suggestions(final_actions, info)
            self._update_enhanced_metrics()
            self._update_state_manager()
            await self._broadcast_delay_aware_updates()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in simulation step: {e}")
            return False
    
    # All other methods remain exactly the same...
    # (Just copying the method signatures to show they're unchanged)
    def _update_state_manager(self):
      """Push current dispatcher state into StateManager"""
      data = {
          "step": self.env.current_step,
          "trains": [train.to_frontend_format() for train in self.train_states],
          "metrics": self.metrics,
          "suggestions": [s.to_frontend_format() for s in self.suggestions_queue],
          "delay_predictions": self.delay_predictions,
      }
      state_manager.bulk_update(data)

    
    async def get_delay_aware_actions(self) -> np.ndarray:
        """Get actions from RL model enhanced with delay awareness"""
        # Method unchanged
        pass
    
    def _update_enhanced_train_states(self):
        """Update train states with comprehensive delay information"""
        # Method unchanged
        pass
    
    async def _generate_delay_aware_suggestions(self, actions: np.ndarray, info: Dict):
        """Generate action suggestions with delay-aware reasoning"""
        # Method unchanged - uses self.delay_predictions which now comes from pre-trained model
        pass
    
    # ... all other methods remain unchanged

# Factory function remains the same
def create_delay_aware_dispatcher(config_path: str = "./config.yaml") -> DelayAwareDispatcher:
    """Create and initialize a delay-aware rail dispatcher with pre-trained model"""
    return DelayAwareDispatcher(config_path)