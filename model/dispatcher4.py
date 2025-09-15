"""
Enhanced Rail System Dispatcher - Fully Delay-Aware Version
Improved integration of delay predictions with RL actions and frontend communication
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

# Import your existing modules
import sys
sys.path.append('./models')
from model.rail_env7 import RailEnv, SIMPLIFIED_CONFIG
sys.path.append(os.path.dirname(__file__))
from delaymodel import analyze_train_delays

from backend.state_manager import state_manager

# Import stable-baselines3 for model loading
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

class DelayAwareDispatcher:
    """Enhanced dispatcher with full delay integration"""
    
    def __init__(self, config_path: str = "./config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize environment
        self.env = RailEnv(config=SIMPLIFIED_CONFIG)
        self.obs, _ = self.env.reset()
        
        # Load trained model
        self.model = self._load_model()
        
        # Enhanced state management
        self.train_states: List[DelayAwareTrainState] = []
        self.suggestions_queue: deque = deque(maxlen=20)
        self.pending_actions: Dict[str, Any] = {}
        self.accepted_actions: List[int] = []
        
        # Delay tracking
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
        
        logger.info("Delay-aware dispatcher initialized successfully")
    
    def _load_config(self, path="./config.yaml"):
        """Load system configuration"""
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            config = {
                "model_path": "../model/best_model/20250912-112601/best_model.zip",
                "delay_update_interval": 10,  # More frequent updates for delay awareness
                "suggestion_threshold": 0.2,  # Lower threshold for delay-based suggestions
                "delay_prediction_horizon": 50,  # Steps to predict ahead
                "cascade_sensitivity": 0.3,  # Sensitivity to cascade effects
                "ws_port": 8765,
                "api_port": 8000,
                "update_rate": 2,  # Faster updates for real-time delay tracking
            }

        # Always resolve model_path relative to this file's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = config.get("model_path")
        if model_path:
            config["model_path"] = os.path.abspath(os.path.join(base_dir, model_path))

        return config
    
    def _load_model(self):
        """Load RL model with delay-aware enhancements"""
        try:
            model_path = self.config["model_path"]
            logger.info(f"Loading delay-aware RL model from: {model_path}")

            if not os.path.exists(model_path):
                logger.warning(f"No RL model found at {model_path}, using delay-aware heuristics")
                return None

            model = PPO.load(model_path, env=self.env)
            logger.info(f"Successfully loaded model from {model_path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Running without RL model - using delay-aware heuristics")
            return None
    
    def _collect_enhanced_schedule_data(self) -> List[Dict[str, Any]]:
        """
        Collect comprehensive schedule data with delay trend analysis
        """
        schedule_data = []
        
        for i in range(self.env.n_trains):
            # Only include trains that have meaningful state
            if self.env.started[i] and not self.env.disabled[i]:
                current_delay = self.env._compute_current_delay(i)
                
                # Update delay history
                self.delay_history[i].append(current_delay)
                
                # Calculate delay trend
                delay_trend = "stable"
                if len(self.delay_history[i]) >= 3:
                    recent = list(self.delay_history[i])[-3:]
                    if recent[-1] > recent[0] + 2:
                        delay_trend = "worsening"
                    elif recent[-1] < recent[0] - 2:
                        delay_trend = "improving"
                
                schedule_data.append({
                    'train_id': f"train_{i}",
                    'delay': max(0.0, float(current_delay)),
                    'position': float(self.env.positions[i]),
                    'speed': int(self.env.speeds[i]),
                    'track': int(self.env.tracks[i]),
                    'collision_risk': float(self.env.collision_risks[i]),
                    'delay_trend': delay_trend,
                    'eta_deviation': self._calculate_eta_deviation(i),
                    'schedule_pressure': self._calculate_schedule_pressure(i)
                })
        
        return schedule_data
    
    def _calculate_eta_deviation(self, train_id: int) -> float:
        """Calculate how far off the train is from expected arrival time"""
        if self.env.arrived[train_id]:
            return 0.0
        
        # Estimate remaining time based on current speed and distance
        remaining_dist = self.env.destinations[train_id] - self.env.positions[train_id]
        if remaining_dist <= 0:
            return 0.0
        
        current_speed = max(1, self.env.speeds[train_id])
        estimated_remaining_time = remaining_dist / current_speed
        
        # Compare with planned remaining time
        planned_remaining = max(0, self.env.planned_arrival[train_id] - self.env.current_step)
        
        return float(estimated_remaining_time - planned_remaining)
    
    def _calculate_schedule_pressure(self, train_id: int) -> float:
        """Calculate pressure from schedule constraints (0-1 scale)"""
        if self.env.arrived[train_id]:
            return 0.0
        
        current_delay = self.env._compute_current_delay(train_id)
        time_remaining = max(1, self.env.planned_arrival[train_id] - self.env.current_step)
        
        # Higher pressure means less time buffer
        pressure = min(1.0, current_delay / time_remaining)
        return float(pressure)
    
    def _update_delay_predictions_enhanced(self):
        """
        Enhanced delay prediction update with trend analysis and cascade detection
        """
        try:
            # Collect enhanced schedule data
            schedule_data = self._collect_enhanced_schedule_data()
            
            if not schedule_data:
                logger.debug("No active trains found for delay analysis")
                self.delay_predictions = {}
                return
            
            # Analyze delays using the delay model
            analysis_results = analyze_train_delays(
                schedule_data,
                outlier_threshold=2.0,
                delay_threshold=1.0
            )
            
            # Enhance with prediction and trend analysis
            enhanced_results = self._enhance_delay_predictions(analysis_results, schedule_data)
            
            # Store enhanced results
            self.delay_predictions = enhanced_results
            
            # Update cascade delay tracking
            self._update_cascade_delays(enhanced_results)
            
            # Update prediction accuracy
            self._update_prediction_accuracy()
            
            logger.info(f"Enhanced delay predictions updated - Mean: {enhanced_results.get('mean_delay', 0):.2f}min, "
                       f"Cascade risks: {len(self.cascade_delays)} trains")
            
        except Exception as e:
            logger.error(f"Error in enhanced delay prediction update: {e}")
            self.delay_predictions = {}
    
    def _enhance_delay_predictions(self, base_results: Dict, schedule_data: List[Dict]) -> Dict:
        """Enhance base delay analysis with additional predictions"""
        enhanced = base_results.copy()
        
        # Add prediction horizon analysis
        prediction_horizon = self.config.get("delay_prediction_horizon", 50)
        
        # Calculate future delay predictions for each train
        future_delays = {}
        delay_trends = {}
        
        for train_data in schedule_data:
            train_id = train_data['train_id']
            current_delay = train_data['delay']
            trend = train_data['delay_trend']
            collision_risk = train_data['collision_risk']
            
            # Predict future delay based on current trend and risk factors
            if trend == "worsening":
                future_delay = current_delay * 1.5 + collision_risk * 10
            elif trend == "improving":
                future_delay = max(0, current_delay * 0.7 - 2)
            else:  # stable
                future_delay = current_delay + collision_risk * 5
            
            future_delays[train_id] = min(100, future_delay)  # Cap at 100 minutes
            delay_trends[train_id] = trend
        
        enhanced.update({
            'future_delays': future_delays,
            'delay_trends': delay_trends,
            'prediction_horizon_steps': prediction_horizon,
            'system_delay_pressure': self._calculate_system_delay_pressure(schedule_data),
            'cascade_risk_level': self._assess_cascade_risk(schedule_data)
        })
        
        return enhanced
    
    def _calculate_system_delay_pressure(self, schedule_data: List[Dict]) -> float:
        """Calculate overall system pressure from delays"""
        if not schedule_data:
            return 0.0
        
        total_pressure = sum(data.get('schedule_pressure', 0) for data in schedule_data)
        return min(1.0, total_pressure / len(schedule_data))
    
    def _assess_cascade_risk(self, schedule_data: List[Dict]) -> str:
        """Assess risk level of cascade delays"""
        high_delay_trains = sum(1 for data in schedule_data if data['delay'] > 10)
        total_trains = len(schedule_data)
        
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
        
        # Clear old cascade delays
        self.cascade_delays.clear()
        
        # Calculate cascade effects
        for i in range(self.env.n_trains):
            train_key = f"train_{i}"
            
            if train_key in delay_probs and delay_probs[train_key] > cascade_sensitivity:
                # This train might cause cascade delays
                cascade_factor = delay_probs[train_key] * future_delays.get(train_key, 0)
                
                # Affect downstream trains (simple model)
                for j in range(i + 1, min(i + 4, self.env.n_trains)):  # Next 3 trains
                    if self.env.started[j] and not self.env.arrived[j]:
                        existing_cascade = self.cascade_delays.get(j, 0)
                        new_cascade = cascade_factor * 0.3 * (1 / (j - i))  # Decay with distance
                        self.cascade_delays[j] = existing_cascade + new_cascade
    
    def _update_prediction_accuracy(self):
        """Update accuracy metrics for delay predictions"""
        # Simple accuracy tracking based on prediction vs actual
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(self.env.n_trains):
            if len(self.delay_history[i]) >= 2:
                # Compare last prediction with actual
                total_predictions += 1
                # Simplified accuracy check (within 20% of actual)
                if abs(list(self.delay_history[i])[-1] - list(self.delay_history[i])[-2]) < 2:
                    correct_predictions += 1
        
        if total_predictions > 0:
            self.metrics["delay_prediction_accuracy"] = correct_predictions / total_predictions
    
    async def get_delay_aware_actions(self) -> np.ndarray:
        """Get actions from RL model enhanced with delay awareness"""
        if self.model is None:
            return self._get_delay_aware_heuristics()
        
        # Enhance observations with comprehensive delay information
        enhanced_obs = self._create_delay_enhanced_observations()
        
        # Get model predictions
        actions, _ = self.model.predict(enhanced_obs, deterministic=True)
        
        # Post-process actions with delay considerations
        refined_actions = self._refine_actions_for_delays(actions)
        
        return refined_actions
    
    def _create_delay_enhanced_observations(self) -> np.ndarray:
        """Create comprehensive delay-enhanced observations"""
        obs = self.obs.copy()
        features_per_train = self.env.features_per_train
        
        # Get current delay predictions
        delay_probs = self.delay_predictions.get('delay_probabilities', {})
        future_delays = self.delay_predictions.get('future_delays', {})
        system_pressure = self.delay_predictions.get('system_delay_pressure', 0.0)
        
        for i in range(self.env.n_trains):
            train_key = f"train_{i}"
            base_idx = i * features_per_train
            
            # Enhance existing delay feature (index 13)
            delay_idx = base_idx + 13
            if delay_idx < len(obs):
                current_delay = self.env._compute_current_delay(i)
                delay_prob = delay_probs.get(train_key, 0.0)
                future_delay = future_delays.get(train_key, 0.0)
                cascade_delay = self.cascade_delays.get(i, 0.0)
                
                # Combine multiple delay factors
                enhanced_delay_feature = (
                    current_delay * 0.4 + 
                    delay_prob * 30 * 0.3 + 
                    future_delay * 0.2 + 
                    cascade_delay * 0.1
                ) / 30  # Normalize
                
                obs[delay_idx] = min(1.0, enhanced_delay_feature)
            
            # Add system pressure to network state features if available
            if len(obs) > base_idx + 20:  # Ensure network features exist
                pressure_idx = base_idx + 21  # Network congestion + 1
                if pressure_idx < len(obs):
                    obs[pressure_idx] = system_pressure
        
        return obs
    
    def _refine_actions_for_delays(self, base_actions: np.ndarray) -> np.ndarray:
        """Refine RL actions based on delay-specific logic"""
        refined_actions = base_actions.copy()
        
        delay_probs = self.delay_predictions.get('delay_probabilities', {})
        future_delays = self.delay_predictions.get('future_delays', {})
        
        for i in range(self.env.n_trains):
            if not self.env.started[i] or self.env.arrived[i] or self.env.disabled[i]:
                continue
            
            train_key = f"train_{i}"
            current_delay = self.env._compute_current_delay(i)
            delay_prob = delay_probs.get(train_key, 0.0)
            future_delay = future_delays.get(train_key, 0.0)
            cascade_risk = self.cascade_delays.get(i, 0.0)
            
            # Override RL action if delay situation is critical
            if current_delay > 20 and delay_prob > 0.7:
                # Critical delay - prioritize speed recovery
                if self.env.speeds[i] < self.env.max_speed and self.env.collision_risks[i] < 0.3:
                    refined_actions[i] = 1  # Accelerate
                elif self._can_request_priority(i):
                    refined_actions[i] = 8  # Request priority
            
            elif cascade_risk > 5 and self._can_switch_track(i):
                # High cascade risk - try to switch tracks
                refined_actions[i] = 5 if self.env.tracks[i] > 0 else 6
            
            elif future_delay > current_delay + 5:
                # Worsening delay prediction - take preventive action
                if self.env.speeds[i] == 0 and not self.env._is_in_any_station(i):
                    refined_actions[i] = 1  # Start moving
        
        return refined_actions
    
    def _get_delay_aware_heuristics(self) -> np.ndarray:
        """Enhanced heuristic actions with delay awareness"""
        actions = np.zeros(self.env.n_trains, dtype=np.int32)
        
        delay_probs = self.delay_predictions.get('delay_probabilities', {})
        future_delays = self.delay_predictions.get('future_delays', {})
        
        for i in range(self.env.n_trains):
            if not self.env.started[i] or self.env.arrived[i] or self.env.disabled[i]:
                actions[i] = 0  # No-op
                continue
            
            # Get delay information
            train_key = f"train_{i}"
            current_delay = self.env._compute_current_delay(i)
            delay_prob = delay_probs.get(train_key, 0.0)
            future_delay = future_delays.get(train_key, 0.0)
            cascade_risk = self.cascade_delays.get(i, 0.0)
            collision_risk = self.env.collision_risks[i]
            
            # Prioritize safety first
            if collision_risk > 0.7:
                actions[i] = 3  # Emergency brake
            elif collision_risk > 0.4:
                actions[i] = 2  # Decelerate
            
            # Then consider delay mitigation
            elif current_delay > 15 or delay_prob > 0.6:
                if self._can_request_priority(i):
                    actions[i] = 8  # Request priority
                elif self.env.speeds[i] < self.env.max_speed and collision_risk < 0.2:
                    actions[i] = 1  # Accelerate
                elif self._can_switch_track(i):
                    actions[i] = 5 if self.env.tracks[i] > 0 else 6
                else:
                    actions[i] = 1 if self.env.speeds[i] < 3 else 0
            
            # Handle cascade prevention
            elif cascade_risk > 3:
                if self._can_switch_track(i):
                    actions[i] = 5 if self.env.tracks[i] > 0 else 6
                elif self.env.speeds[i] < 2:
                    actions[i] = 1  # Accelerate to clear path
            
            # Normal operations with delay awareness
            elif self.env._is_in_any_station(i):
                actions[i] = 4  # Hold at station
            elif self.env.speeds[i] < 2 and future_delay > current_delay:
                actions[i] = 1  # Accelerate to prevent worsening
            else:
                actions[i] = 0  # Maintain current state
        
        return actions
    
    def _can_request_priority(self, train_idx: int) -> bool:
        """Check if train can request priority"""
        # Simple logic - can request if near junction and delayed
        return (self.env._distance_to_next_junction(train_idx) < 50 and
                self.env._compute_current_delay(train_idx) > 5)
    
    def _can_switch_track(self, train_idx: int) -> bool:
        """Check if train can safely switch tracks"""
        return (self.env._is_in_switch_zone(train_idx) and 
                self.env.speeds[train_idx] <= 1 and
                self.env.collision_risks[train_idx] < 0.3)
    
    def _update_enhanced_train_states(self):
        """Update train states with comprehensive delay information"""
        self.train_states.clear()
        
        delay_probs = self.delay_predictions.get('delay_probabilities', {})
        future_delays = self.delay_predictions.get('future_delays', {})
        delay_trends = self.delay_predictions.get('delay_trends', {})
        
        for i in range(self.env.n_trains):
            # Determine status
            status = "active"
            if self.env.disabled[i]:
                status = "disabled"
            elif self.env.arrived[i]:
                status = "arrived"
            elif not self.env.started[i]:
                status = "waiting"
            elif self.env.speeds[i] == 0:
                status = "stopped"
            
            # Get delay information
            train_key = f"train_{i}"
            current_delay = self.env._compute_current_delay(i)
            predicted_delay = future_delays.get(train_key, current_delay)
            delay_probability = delay_probs.get(train_key, 0.0)
            delay_trend = delay_trends.get(train_key, "stable")
            
            # Calculate time to collision
            time_to_collision = self.env._time_to_collision(i)
            if time_to_collision == float('inf'):
                time_to_collision = 999.0  # Max display value
            
            state = DelayAwareTrainState(
                id=i,
                track=int(self.env.tracks[i]),
                position=float(self.env.positions[i]),
                speed=int(self.env.speeds[i]),
                status=status,
                current_delay=float(current_delay),
                predicted_delay=float(predicted_delay),
                delay_probability=float(delay_probability),
                delay_trend=delay_trend,
                collision_risk=float(self.env.collision_risks[i]),
                time_to_collision=float(time_to_collision),
                eta_deviation=self._calculate_eta_deviation(i),
                schedule_pressure=self._calculate_schedule_pressure(i)
            )
            self.train_states.append(state)
    
    async def _generate_delay_aware_suggestions(self, actions: np.ndarray, info: Dict):
        """Generate action suggestions with delay-aware reasoning"""
        threshold = self.config.get("suggestion_threshold", 0.2)
        
        delay_probs = self.delay_predictions.get('delay_probabilities', {})
        future_delays = self.delay_predictions.get('future_delays', {})
        
        for i in range(self.env.n_trains):
            if not self.env.started[i] or self.env.arrived[i] or self.env.disabled[i]:
                continue
            
            # Get comprehensive state
            train_key = f"train_{i}"
            current_delay = self.env._compute_current_delay(i)
            predicted_delay = future_delays.get(train_key, current_delay)
            delay_prob = delay_probs.get(train_key, 0.0)
            collision_risk = self.env.collision_risks[i]
            cascade_risk = self.cascade_delays.get(i, 0.0)
            
            # Determine if intervention needed and generate appropriate suggestion
            suggestion = None
            
            # Critical situations (safety + severe delays)
            if collision_risk > 0.7 and current_delay > 10:
                suggestion = DelayAwareActionSuggestion(
                    id=f"sug_{int(time.time()*1000)}_{i}",
                    train_id=i,
                    action_type="action_3",
                    reason=f"Critical: Collision risk {collision_risk:.2f} + {current_delay:.1f}min delay",
                    priority="critical",
                    predicted_outcome="Prevents collision and limits delay cascade",
                    timestamp=time.time(),
                    delay_impact=-2.0,
                    cascade_prevention=True,
                    schedule_recovery=False
                )
            
            # High delay situations
            elif current_delay > 15 or delay_prob > 0.6:
                if self._can_request_priority(i):
                    suggestion = DelayAwareActionSuggestion(
                        id=f"sug_{int(time.time()*1000)}_{i}",
                        train_id=i,
                        action_type="action_8",
                        reason=f"High delay ({current_delay:.1f}min) - requesting priority",
                        priority="high",
                        predicted_outcome="Reduces delay by 3-5 minutes through priority routing",
                        timestamp=time.time(),
                        delay_impact=-4.0,
                        cascade_prevention=True,
                        schedule_recovery=True
                    )
                elif self.env.speeds[i] < self.env.max_speed and collision_risk < 0.2:
                    suggestion = DelayAwareActionSuggestion(
                        id=f"sug_{int(time.time()*1000)}_{i}",
                        train_id=i,
                        action_type="action_1",
                        reason=f"Accelerate to recover from {current_delay:.1f}min delay",
                        priority="high",
                        predicted_outcome="Reduces delay by 1-3 minutes",
                        timestamp=time.time(),
                        delay_impact=-2.5,
                        cascade_prevention=False,
                        schedule_recovery=True
                    )
            
            # Cascade prevention
            elif cascade_risk > 3 and self._can_switch_track(i):
                suggestion = DelayAwareActionSuggestion(
                    id=f"sug_{int(time.time()*1000)}_{i}",
                    train_id=i,
                    action_type="action_5" if self.env.tracks[i] > 0 else "action_6",
                    reason=f"Switch tracks to prevent cascade delays (risk: {cascade_risk:.1f})",
                    priority="medium",
                    predicted_outcome="Prevents up to 5 minutes of cascade delays",
                    timestamp=time.time(),
                    delay_impact=-3.0,
                    cascade_prevention=True,
                    schedule_recovery=False
                )
            
            # Predictive delay prevention
            elif predicted_delay > current_delay + 3 and delay_prob > threshold:
                if self.env.speeds[i] == 0 and not self.env._is_in_any_station(i):
                    suggestion = DelayAwareActionSuggestion(
                        id=f"sug_{int(time.time()*1000)}_{i}",
                        train_id=i,
                        action_type="action_1",
                        reason=f"Predicted delay increase: {predicted_delay:.1f}min (currently {current_delay:.1f}min)",
                        priority="medium",
                        predicted_outcome="Prevents predicted delay increase",
                        timestamp=time.time(),
                        delay_impact=-(predicted_delay - current_delay),
                        cascade_prevention=False,
                        schedule_recovery=True
                    )
            
            # Station optimization for schedule recovery
            elif self.env._is_in_any_station(i) and current_delay > 5:
                suggestion = DelayAwareActionSuggestion(
                    id=f"sug_{int(time.time()*1000)}_{i}",
                    train_id=i,
                    action_type="action_7",
                    reason=f"Skip station stop to recover {current_delay:.1f}min delay",
                    priority="low",
                    predicted_outcome="Recovers 2-4 minutes of delay",
                    timestamp=time.time(),
                    delay_impact=-3.0,
                    cascade_prevention=False,
                    schedule_recovery=True
                )
            
            if suggestion:
                self.suggestions_queue.append(suggestion)
    
    def _update_enhanced_metrics(self):
        """Update comprehensive performance metrics including delay-specific metrics"""
        current_time = time.time()
        
        # Basic counts
        active_trains = sum(1 for i in range(self.env.n_trains) 
                           if self.env.started[i] and not self.env.arrived[i] and not self.env.disabled[i])
        arrived_trains = sum(1 for i in range(self.env.n_trains) if self.env.arrived[i])
        
        # Delay metrics
        total_current_delays = 0.0
        on_time_arrivals = 0
        delayed_arrivals = 0
        
        for i in range(self.env.n_trains):
            current_delay = self.env._compute_current_delay(i)
            total_current_delays += max(0, current_delay)
            
            if self.env.arrived[i]:
                if current_delay <= 2:  # Within 2 minutes is considered on-time
                    on_time_arrivals += 1
                else:
                    delayed_arrivals += 1
        
        # Update metrics
        self.metrics.update({
            "active_trains": active_trains,
            "arrived_trains": arrived_trains,
            "on_time_arrivals": on_time_arrivals,
            "delayed_arrivals": delayed_arrivals,
            "total_delays": total_current_delays,
            "avg_delay_per_train": total_current_delays / max(1, self.env.n_trains) if active_trains > 0 else 0.0,
            "system_efficiency": max(0.0, 1.0 - (total_current_delays / (self.env.n_trains * 30))),  # Normalized efficiency
            "cascade_delays_at_risk": len(self.cascade_delays),
            "system_delay_pressure": self.delay_predictions.get('system_delay_pressure', 0.0),
            "cascade_risk_level": self.delay_predictions.get('cascade_risk_level', 'none'),
            "last_update": current_time
        })
        
        # Performance tracking for delay-aware actions
        suggestions_in_last_minute = len([s for s in self.suggestions_queue 
                                        if current_time - s.timestamp < 60])
        self.metrics["active_suggestions"] = suggestions_in_last_minute
    
    async def step(self):
        """Enhanced main simulation step with comprehensive delay integration"""
        try:
            # Update delay predictions more frequently
            self.suggestions_queue.clear()
            current_time = time.time()
            if current_time - self.last_delay_update >= self.config.get("delay_update_interval", 10):
                self._update_delay_predictions_enhanced()
                self.last_delay_update = current_time
            
            # Get delay-aware actions
            actions = await self.get_delay_aware_actions()
            
            # Apply accepted actions from frontend
            final_actions = self._apply_accepted_actions(actions)
            
            # Execute environment step
            self.obs, rewards, dones, truncated, info = self.env.step(final_actions)
            
            # Update comprehensive train states
            self._update_enhanced_train_states()
            
            # Generate delay-aware suggestions
            await self._generate_delay_aware_suggestions(final_actions, info)
            
            # Update all metrics
            self._update_enhanced_metrics()
            
            # Update state manager with delay information
            self._update_state_manager()
            
            # Broadcast updates to all connected clients
            await self._broadcast_delay_aware_updates()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in enhanced simulation step: {e}")
            return False
    
    def _apply_accepted_actions(self, suggested_actions: np.ndarray) -> np.ndarray:
        """Apply user-accepted actions, falling back to suggested actions"""
        final_actions = suggested_actions.copy()
        
        current_time = time.time()
        expired_actions = []
        
        for action_id, action_data in self.pending_actions.items():
            # Check if action has expired (30 seconds timeout)
            if current_time - action_data.get('timestamp', 0) > 30:
                expired_actions.append(action_id)
                continue
            
            train_id = action_data.get('train_id')
            action_type = action_data.get('action')
            
            if train_id is not None and 0 <= train_id < len(final_actions):
                final_actions[train_id] = action_type
        
        # Clean up expired actions
        for action_id in expired_actions:
            del self.pending_actions[action_id]
        
        return final_actions
    
    def _update_state_manager(self):
        """Update state manager with comprehensive delay-aware information"""
        try:
            # Prepare enhanced state data
            state_data = {
                "trains": [state.to_frontend_format() for state in self.train_states],
                "suggestions": [sug.to_frontend_format() for sug in list(self.suggestions_queue)[-10:]],
                "metrics": self.metrics.copy(),
                "delay_analysis": self.delay_predictions,
                "system_status": {
                    "step": int(self.env.current_step),
                    "total_trains": self.env.n_trains,
                    "system_delay_pressure": self.delay_predictions.get('system_delay_pressure', 0.0),
                    "cascade_risk_level": self.delay_predictions.get('cascade_risk_level', 'none'),
                    "avg_delay": self.metrics.get('avg_delay_per_train', 0.0),
                    "on_time_percentage": (self.metrics.get('on_time_arrivals', 0) / 
                                         max(1, self.metrics.get('arrived_trains', 1))) * 100
                }
            }
            
            # Update state manager
            state_manager.bulk_update(state_data)
            
        except Exception as e:
            logger.error(f"Error updating state manager: {e}")
    
    async def _broadcast_delay_aware_updates(self):
        """Broadcast comprehensive updates to all WebSocket connections"""
        if not self.ws_connections:
            return
        
        try:
            # Prepare comprehensive update message
            update_message = {
                "type": "delay_aware_update",
                "data": {
                    "trains": [state.to_frontend_format() for state in self.train_states],
                    "suggestions": [sug.to_frontend_format() for sug in list(self.suggestions_queue)[-5:]],
                    "metrics": self.metrics,
                    "delay_overview": {
                        "mean_delay": self.delay_predictions.get('mean_delay', 0.0),
                        "delay_std": self.delay_predictions.get('delay_std', 0.0),
                        "high_delay_trains": len([s for s in self.train_states if s.current_delay > 10]),
                        "system_pressure": self.delay_predictions.get('system_delay_pressure', 0.0),
                        "cascade_risk": self.delay_predictions.get('cascade_risk_level', 'none'),
                        "prediction_accuracy": self.metrics.get('delay_prediction_accuracy', 0.0)
                    },
                    "timestamp": time.time()
                }
            }
            
            message_str = json.dumps(update_message)
            
            # Send to all connected clients
            disconnected = set()
            for ws in self.ws_connections:
                try:
                    await ws.send(message_str)
                except Exception as e:
                    logger.warning(f"Failed to send update to WebSocket: {e}")
                    disconnected.add(ws)
            
            # Clean up disconnected clients
            self.ws_connections -= disconnected
            
        except Exception as e:
            logger.error(f"Error broadcasting delay-aware updates: {e}")
    
    async def accept_action(self, action_data: Dict) -> Dict:
        """Accept and queue an action from the frontend with delay impact tracking"""
        try:
            action_id = action_data.get('id')
            train_id = action_data.get('train_id')
            action_type = action_data.get('action')
            
            if not all([action_id, train_id is not None, action_type is not None]):
                return {"success": False, "error": "Missing required fields"}
            
            # Validate train_id
            if not (0 <= train_id < self.env.n_trains):
                return {"success": False, "error": "Invalid train ID"}
            
            # Store the accepted action with enhanced tracking
            self.pending_actions[action_id] = {
                'train_id': train_id,
                'action': action_type,
                'timestamp': time.time(),
                'predicted_delay_impact': action_data.get('delay_impact', 0.0),
                'cascade_prevention': action_data.get('cascade_prevention', False),
                'schedule_recovery': action_data.get('schedule_recovery', False)
            }
            
            # Track acceptance in metrics
            if action_data.get('schedule_recovery', False):
                self.metrics["schedule_recovery_actions"] = self.metrics.get("schedule_recovery_actions", 0) + 1
            if action_data.get('cascade_prevention', False):
                self.metrics["cascade_delays_prevented"] = self.metrics.get("cascade_delays_prevented", 0) + 1
            
            # Remove from suggestions queue if it exists
            self.suggestions_queue = deque([s for s in self.suggestions_queue if s.id != action_id], 
                                         maxlen=20)
            
            logger.info(f"Accepted delay-aware action {action_id} for train {train_id}: {action_type}")
            
            return {
                "success": True, 
                "message": "Action accepted and queued",
                "expected_delay_impact": action_data.get('delay_impact', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error accepting action: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_comprehensive_status(self) -> Dict:
        """Get comprehensive system status including detailed delay analysis"""
        try:
            return {
                "trains": [state.to_frontend_format() for state in self.train_states],
                "suggestions": [sug.to_frontend_format() for sug in list(self.suggestions_queue)],
                "metrics": self.metrics,
                "delay_analysis": {
                    "current_analysis": self.delay_predictions,
                    "cascade_delays": dict(self.cascade_delays),
                    "delay_trends": {f"train_{i}": list(self.delay_history[i])[-3:] 
                                   for i in range(self.env.n_trains) if len(self.delay_history[i]) > 0},
                    "system_health": {
                        "overall_delay_pressure": self.delay_predictions.get('system_delay_pressure', 0.0),
                        "cascade_risk_assessment": self.delay_predictions.get('cascade_risk_level', 'none'),
                        "prediction_reliability": self.metrics.get('delay_prediction_accuracy', 0.0),
                        "trains_at_risk": len([s for s in self.train_states 
                                             if s.current_delay > 10 or s.collision_risk > 0.5])
                    }
                },
                "system_info": {
                    "step": int(self.env.current_step),
                    "total_trains": self.env.n_trains,
                    "active_trains": sum(1 for s in self.train_states if s.status == "active"),
                    "model_loaded": self.model is not None,
                    "last_delay_update": self.last_delay_update,
                    "update_frequency": self.config.get("delay_update_interval", 10)
                },
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error getting comprehensive status: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def cleanup(self):
        """Enhanced cleanup with delay tracking cleanup"""
        try:
            # Clear all delay tracking data
            self.delay_predictions.clear()
            self.cascade_delays.clear()
            for history in self.delay_history.values():
                history.clear()
            
            # Clear other queues
            self.suggestions_queue.clear()
            self.pending_actions.clear()
            self.ws_connections.clear()
            
            logger.info("Enhanced delay-aware dispatcher cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Factory function for creating delay-aware dispatcher
def create_delay_aware_dispatcher(config_path: str = "./config.yaml") -> DelayAwareDispatcher:
    """Create and initialize a delay-aware rail dispatcher"""
    return DelayAwareDispatcher(config_path)

# Async context manager for proper resource management
class DelayAwareDispatcherContext:
    def __init__(self, config_path: str = "./config.yaml"):
        self.config_path = config_path
        self.dispatcher = None
    
    async def __aenter__(self):
        self.dispatcher = create_delay_aware_dispatcher(self.config_path)
        return self.dispatcher
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.dispatcher:
            self.dispatcher.cleanup()

if __name__ == "__main__":
    import asyncio
    import logging

    logger = logging.getLogger(__name__)
    dispatcher = DelayAwareDispatcher()

    async def main():
        """Runs the dispatcher in a simple endless loop without WebSocket server."""
        update_interval = 1.0 / dispatcher.config.get("update_rate", 2)
        logger.info("Starting dispatcher loop (no WebSocket server)...")

        while True:
            try:
                await dispatcher.step()
                await asyncio.sleep(update_interval)
            except KeyboardInterrupt:
                logger.info("Dispatcher stopped by user.")
                break
            except Exception as e:
                logger.error(f"Error in dispatcher loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    asyncio.run(main())
