from typing import Tuple, Dict, Any, List, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import time
import copy
import csv
import os
from collections import deque
import logging

# Simplified configuration with hardcoded values
SIMPLIFIED_CONFIG = {
    # Fixed topology
    "n_tracks": 4,
    "n_trains": 10,  # Hardcoded
    "track_length": 1200,
    "train_length": 25,
    "station_halt_time": 12,
    "max_speed": 4,
    "accel_units": 1,
    "decel_units": 1,
    "max_steps": 1800,  # Hardcoded
    "stations": {
        "A": (20, 50),
        "B": (760, 790),
        "C": (1420, 1450)
    },
    "junctions": [320, 680, 1450],
    "spawn_points": [50, 320, 680],
    "unit_meters": 10.0,
    "timestep_seconds": 2.0,
    "brake_mps2": 1.2,
    "track_speed_limits": [4, 4, 4, 4],
    "cascade_N": 8,
    "log_dir": "runs",
    
    # Simplified reward system
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
    
    # Observation configuration
    "observation_config": {
        "include_network_state": True,
        "include_predictive_features": True,
        "include_neighbor_states": True,
        "spatial_awareness_radius": 200.0,
        
    }
}

class PotentialBasedShaping:
    """Simplified potential-based reward shaping."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.prev_potentials = {}
    
    def compute_potential(self, state: Dict, train_id: int) -> float:
        """Compute potential function value for a train state."""
        potential = 0.0
        
        # Distance to destination potential
        if "position" in state and "destination" in state:
            progress = state["position"] / max(1.0, state["destination"])
            potential += 10.0 * progress
        
        # Schedule adherence potential
        if "current_time" in state and "planned_arrival" in state:
            time_buffer = max(0, state["planned_arrival"] - state["current_time"])
            potential += 25.0 * time_buffer / 100.0
        
        # Safety potential (inverse of collision risk)
        if "collision_risk" in state:
            safety_potential = 1.0 / (1.0 + state["collision_risk"])
            potential += 100.0 * safety_potential
        
        return potential
    
    def get_shaping_reward(self, state: Dict, train_id: int) -> float:
        """Get the potential-based shaping reward."""
        current_potential = self.compute_potential(state, train_id)
        prev_potential = self.prev_potentials.get(train_id, 0.0)
        
        shaping_reward = current_potential - prev_potential
        self.prev_potentials[train_id] = current_potential
        
        return shaping_reward
    
    def reset(self):
        """Reset potential tracking for new episode."""
        self.prev_potentials.clear()

class RailEnv(gym.Env):
    """Simplified RailEnv with hardcoded values and no curriculum learning."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, config: Dict = None, render_mode: str = None):
        super().__init__()
        
        # Use simplified configuration
        base = copy.deepcopy(SIMPLIFIED_CONFIG)
        if config:
            self._deep_update(base, config)
        self.cfg = base
        
        

        # Initialize base environment properties (hardcoded values)
        self._init_base_properties()
        
        # Simplified components
        self.potential_shaping = PotentialBasedShaping(self.cfg["reward_config"])
        
        # Performance tracking
        self.episode_metrics = []
        self.performance_window = deque(maxlen=100)
        
        # Initialize observation and action spaces
        self._init_spaces()
        
        # Initialize environment state
        self._init_environment_state()
        
        self.render_mode = render_mode
        
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _init_base_properties(self):
        """Initialize basic environment properties with hardcoded values."""
        self.n_tracks = 4
        self.n_trains = 10 # Hardcoded
        self.track_length = 1200.0
        self.train_length = 25.0
        self.max_speed = 4
        self.max_steps = 1800  # Hardcoded
        
        # Layout
        self.stations = {k: tuple(v) for k, v in self.cfg["stations"].items()}
        self.junctions = list(self.cfg["junctions"])
        self.spawn_points = list(self.cfg["spawn_points"])
        
        # Physics
        self.unit_m = 10.0
        self.timestep_s = 2.0
        self.brake_mps2 = 1.2
        
        # Compute speed tables and braking distances
        self._compute_speed_tables()
    
    def _init_spaces(self):
        """Initialize observation and action spaces."""
        # Action space: 0=no-op,1=accel,2=decel,3=emergency_brake,4=hold,5=switch_left,6=switch_right,7=skip_station,8=request_priority
        self.n_actions_per_train = 9
        self.action_space = spaces.MultiDiscrete([self.n_actions_per_train] * self.n_trains)
        
        self.features_per_train = self._count_features_per_train()
        # Dynamically compute observation shape
        
        obs_size = self.n_trains * self.features_per_train
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),  
            dtype=np.float32
        )

        # Observation space - calculate features per train
        

    
    def _init_environment_state(self):
        """Initialize environment state arrays."""
        self.tracks = np.zeros(self.n_trains, dtype=np.int32)
        self.positions = np.zeros(self.n_trains, dtype=np.float32)
        self.speeds = np.zeros(self.n_trains, dtype=np.int32)
        self.destinations = np.full(self.n_trains, self.track_length, dtype=np.float32)
        self.halt_remaining = np.zeros(self.n_trains, dtype=np.int32)
        self.started = np.zeros(self.n_trains, dtype=bool)
        self.arrived = np.zeros(self.n_trains, dtype=bool)
        self.disabled = np.zeros(self.n_trains, dtype=bool)
        self.start_times = np.zeros(self.n_trains, dtype=np.int32)
        self.planned_arrival = np.zeros(self.n_trains, dtype=np.int32)
        self.actual_arrival = np.full(self.n_trains, -1, dtype=np.int32)
        self.forced_stop_steps = np.zeros(self.n_trains, dtype=int)
        # Enhanced state tracking
        self.last_speeds = np.zeros(self.n_trains, dtype=np.int32)
        self.acceleration_history = [deque(maxlen=5) for _ in range(self.n_trains)]
        self.collision_risks = np.zeros(self.n_trains, dtype=np.float32)
        self.emergency_brake_count = np.zeros(self.n_trains, dtype=np.int32)
        
        # Network state
        self.network_congestion = 0.0
        self.system_throughput = 0.0
        self.cascade_delay_factor = 1.0
        
        # Performance tracking
        self.current_step = 0
        self._episode_collision_count = 0
        self._episode_near_miss_count = 0
        self._episode_signal_violations = 0
        
        # Initialize signals
        self.signal_positions = sorted(list(set([r[1] for r in self.stations.values()] + self.junctions)))
        self.signal_states = {p: True for p in self.signal_positions}
        self.track_blocked_timer = np.zeros(self.n_tracks, dtype=np.int32)
        
        # RNG
        self.rng = np.random.default_rng()
    
    def _compute_speed_tables(self):
        """Compute speed conversion tables and braking distances."""
        self.speed_mps = {}
        for s in range(self.max_speed + 1):
            self.speed_mps[s] = float(s) * (self.unit_m / max(1.0, self.timestep_s))
        
        self.braking_distance_map = {}
        for s, v in self.speed_mps.items():
            if self.brake_mps2 <= 0.0 or v <= 0.0:
                bd_m = 0.0
            else:
                bd_m = (v * v) / (2.0 * self.brake_mps2)
            bd_units = math.ceil(bd_m / self.unit_m)
            self.braking_distance_map[s] = int(bd_units)
    
    def reset(self, *, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        """Reset environment."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Reset all state
        self._init_environment_state()
        
        # Initialize trains
        self._initialize_trains()
        
        # Reset shaping function
        self.potential_shaping.reset()
        
        obs = self._get_enhanced_obs()
        return obs, {"step": 0}
    
    def _initialize_trains(self):
        """Initialize trains with randomized starts."""
        for i in range(self.n_trains):
            self.tracks[i] = int(self.rng.integers(0, self.n_tracks))
            self.positions[i] = float(self.rng.choice(self.spawn_points))
            self.speeds[i] = 0
            
            # Random start times
            self.start_times[i] = int(self.rng.integers(0, 60))
            self.started[i] = bool(self.positions[i] != 0)
            
            # Arrival planning with uncertainty
            distance_units = max(0.0, self.track_length - self.positions[i])
            nominal_speed = max(1.0, float(self.max_speed) * 0.7)
            est_travel_steps = math.ceil(distance_units / nominal_speed)
            est_halts = len(self.stations) * self.cfg["station_halt_time"]
            
            # Add some planning uncertainty
            uncertainty = int(self.rng.normal(0, 10))
            self.planned_arrival[i] = int(self.start_times[i] + est_travel_steps + est_halts + uncertainty)
    
    def _count_features_per_train(self) -> int:
        """Count features per train based on observation config."""
        obs_config = self.cfg["observation_config"]
        base = 20  # Base features
        if obs_config.get("include_network_state", False):
            base += 8
        if obs_config.get("include_predictive_features", False):
            base += 6
        if obs_config.get("include_neighbor_states", False):
            base += 10  # 2 neighbors × 5 features
        return base

    def _get_enhanced_obs(self) -> np.ndarray:
        """Generate enhanced observations."""
        obs_config = self.cfg["observation_config"]
        feats = np.zeros((self.n_trains, self.features_per_train), dtype=np.float32)
        
        for i in range(self.n_trains):
            feat_idx = 0
            
            # Basic features (first 20)
            feats[i, feat_idx] = self._normalize(self.tracks[i], self.n_tracks - 1)
            feat_idx += 1
            feats[i, feat_idx] = self._normalize(self.positions[i], self.track_length)
            feat_idx += 1
            feats[i, feat_idx] = self._normalize(self.speeds[i], self.max_speed)
            feat_idx += 1
            feats[i, feat_idx] = self._normalize(self.destinations[i], self.track_length)
            feat_idx += 1
            
            # Enhanced safety features
            dist_next = self._distance_to_next_train(i)
            feats[i, feat_idx] = self._normalize(min(dist_next, self.track_length), self.track_length)
            feat_idx += 1
            
            braking_dist = self.braking_distance_map.get(int(self.speeds[i]), 0)
            feats[i, feat_idx] = self._normalize(min(braking_dist, self.track_length), self.track_length)
            feat_idx += 1
            
            # Collision risk assessment
            collision_risk = self._compute_collision_risk(i)
            feats[i, feat_idx] = min(1.0, collision_risk)
            feat_idx += 1
            
            # Signal state and distance
            sig_state, sig_dist = self._next_signal_for_train(i)
            feats[i, feat_idx] = 1.0 if sig_state else 0.0
            feat_idx += 1
            feats[i, feat_idx] = self._normalize(min(sig_dist, self.track_length), self.track_length)
            feat_idx += 1
            
            # Station and timing features
            feats[i, feat_idx] = 1.0 if self._is_in_any_station(i) else 0.0
            feat_idx += 1
            feats[i, feat_idx] = self._normalize(self.halt_remaining[i], self.cfg["station_halt_time"])
            feat_idx += 1
            
            # Schedule adherence
            if not self.started[i]:
                time_until = max(0, self.start_times[i] - self.current_step)
                feats[i, feat_idx] = self._normalize(time_until, self.max_steps)
            else:
                feats[i, feat_idx] = 0.0
            feat_idx += 1
            
            # Delay tracking
            delay = self._compute_current_delay(i)
            feats[i, feat_idx] = self._normalize(min(delay, 300), 300)
            feat_idx += 1
            
            # Speed efficiency
            optimal_speed = self._compute_optimal_speed(i)
            speed_efficiency = 1.0 - abs(self.speeds[i] - optimal_speed) / max(1, self.max_speed)
            feats[i, feat_idx] = speed_efficiency
            feat_idx += 1
            
            # Junction proximity
            junc_dist = self._distance_to_next_junction(i)
            feats[i, feat_idx] = self._normalize(min(junc_dist, self.track_length), self.track_length)
            feat_idx += 1
            
            # State flags
            feats[i, feat_idx] = 1.0 if self.disabled[i] else 0.0
            feat_idx += 1
            feats[i, feat_idx] = 1.0 if self.arrived[i] else 0.0
            feat_idx += 1
            
            # Acceleration pattern
            if len(self.acceleration_history[i]) > 0:
                recent_accel = np.mean(list(self.acceleration_history[i]))
                feats[i, feat_idx] = self._normalize(recent_accel + 2, 4)  # Range -2 to +2
            feat_idx += 1
            
            # Emergency brake usage
            feats[i, feat_idx] = self._normalize(self.emergency_brake_count[i], 10)
            feat_idx += 1
            
            # Track congestion
            track_congestion = self._compute_track_congestion(self.tracks[i])
            feats[i, feat_idx] = track_congestion
            feat_idx += 1
            
            # Network state features (if enabled)
            if obs_config["include_network_state"]:
                feats[i, feat_idx] = self.network_congestion
                feat_idx += 1
                feats[i, feat_idx] = self.system_throughput
                feat_idx += 1
                feats[i, feat_idx] = self.cascade_delay_factor
                feat_idx += 1
                feats[i, feat_idx] = self._normalize(self._episode_collision_count, 10)
                feat_idx += 1
                feats[i, feat_idx] = self._normalize(self._episode_near_miss_count, 20)
                feat_idx += 1
                feats[i, feat_idx] = self._normalize(self._episode_signal_violations, 15)
                feat_idx += 1
                feats[i, feat_idx] = self._normalize(len([t for t in range(self.n_trains) if self.started[t] and not self.arrived[t]]), self.n_trains)
                feat_idx += 1
                feats[i, feat_idx] = self._normalize(self.current_step, self.max_steps)
                feat_idx += 1
            
            # Predictive features (if enabled)
            if obs_config["include_predictive_features"]:
                # Time to collision prediction
                ttc = self._time_to_collision(i)
                feats[i, feat_idx] = self._normalize(min(ttc, 50), 50)
                feat_idx += 1
                
                # Predicted arrival time
                eta = self._estimate_arrival_time(i)
                feats[i, feat_idx] = self._normalize(eta, self.max_steps)
                feat_idx += 1
                
                # Schedule pressure
                pressure = max(0, self.current_step - self.planned_arrival[i] + 50) / 100
                feats[i, feat_idx] = min(1.0, pressure)
                feat_idx += 1
                
                # Downstream congestion prediction
                downstream_risk = self._predict_downstream_congestion(i)
                feats[i, feat_idx] = downstream_risk
                feat_idx += 1
                
                # Optimal action hint
                optimal_action = self._compute_optimal_action_hint(i)
                feats[i, feat_idx] = self._normalize(optimal_action, 8)
                feat_idx += 1
                
                # System efficiency trend
                efficiency_trend = self._compute_system_efficiency_trend()
                feats[i, feat_idx] = efficiency_trend
                feat_idx += 1
            
            # Neighbor awareness (if enabled)
            if obs_config["include_neighbor_states"]:
                neighbors = self._get_nearby_trains(i, obs_config["spatial_awareness_radius"])
                
                # Encode up to 2 nearest neighbors
                for j in range(2):
                    if j < len(neighbors):
                        neighbor_id, neighbor_dist, neighbor_speed = neighbors[j]
                        feats[i, feat_idx] = self._normalize(neighbor_dist, obs_config["spatial_awareness_radius"])
                        feat_idx += 1
                        feats[i, feat_idx] = self._normalize(neighbor_speed, self.max_speed)
                        feat_idx += 1
                        feats[i, feat_idx] = 1.0 if self.tracks[neighbor_id] == self.tracks[i] else 0.0
                        feat_idx += 1
                        feats[i, feat_idx] = self._normalize(neighbor_id, self.n_trains - 1)
                        feat_idx += 1
                        
                        # Relative velocity
                        rel_vel = self.speeds[i] - neighbor_speed
                        feats[i, feat_idx] = self._normalize(rel_vel + self.max_speed, 2 * self.max_speed)
                        feat_idx += 1
                    else:
                        # No neighbor - fill with neutral values
                        for _ in range(5):
                            feats[i, feat_idx] = 0.5
                            feat_idx += 1
        
        return feats.flatten().astype(np.float32)
    
    def _normalize(self, value: float, max_value: float) -> float:
        """Normalize value to [0, 1] range."""
        if max_value == 0:
            return 0.0
        return max(0.0, min(1.0, float(value) / float(max_value)))
    
    def _compute_collision_risk(self, train_idx: int) -> float:
        """Compute collision risk for a train."""
        if not self.started[train_idx] or self.arrived[train_idx] or self.disabled[train_idx]:
            return 0.0
        
        risk = 0.0
        track = self.tracks[train_idx]
        pos = self.positions[train_idx]
        speed = self.speeds[train_idx]
        braking_dist = self.braking_distance_map.get(speed, 0)
        
        for other_idx in range(self.n_trains):
            if other_idx == train_idx or self.tracks[other_idx] != track:
                continue
            if not self.started[other_idx] or self.arrived[other_idx] or self.disabled[other_idx]:
                continue
            
            other_pos = self.positions[other_idx]
            distance = abs(pos - other_pos)
            
            if distance < self.train_length:
                return 1.0  # Immediate collision
            
            # Check if we're approaching from behind
            if pos < other_pos and distance <= braking_dist * 1.5:
                risk += 1.0 / (1.0 + distance)
        
        return min(1.0, risk)
    
    def _compute_current_delay(self, train_idx: int) -> float:
        """Enhanced delay calculation returning float (minutes)"""
        if self.actual_arrival[train_idx] >= 0:
            # Train has arrived - calculate actual delay
            return max(0.0, float(self.actual_arrival[train_idx] - self.planned_arrival[train_idx]) * (self.timestep_s / 60.0))
        elif self.started[train_idx] and not self.arrived[train_idx]:
            # Train is running - estimate current delay
            expected_position = (self.current_step - self.start_times[train_idx]) * 2.0  # Expected progress
            actual_position = self.positions[train_idx]
            
            if actual_position < expected_position:
                delay_steps = (expected_position - actual_position) / 2.0
                return max(0.0, delay_steps * (self.timestep_s / 60.0))  # Convert to minutes
        
        return 0.0
    
    def _time_to_collision(self, train_idx: int) -> float:
        """Calculate time to collision in seconds"""
        if not self.started[train_idx] or self.arrived[train_idx] or self.disabled[train_idx]:
            return float('inf')
        
        if self.speeds[train_idx] == 0:
            return float('inf')
        
        track = self.tracks[train_idx]
        pos = self.positions[train_idx]
        speed = self.speeds[train_idx]
        
        min_ttc = float('inf')
        
        for other_idx in range(self.n_trains):
            if other_idx == train_idx or self.tracks[other_idx] != track:
                continue
            if not self.started[other_idx] or self.arrived[other_idx] or self.disabled[other_idx]:
                continue
            
            other_pos = self.positions[other_idx]
            other_speed = self.speeds[other_idx]
            
            # Only consider if we're approaching from behind
            if pos < other_pos:
                relative_speed = speed - other_speed
                if relative_speed > 0:  # We're catching up
                    distance = other_pos - pos - self.train_length
                    if distance > 0:
                        ttc_steps = distance / relative_speed
                        ttc_seconds = ttc_steps * self.timestep_s
                        min_ttc = min(min_ttc, ttc_seconds)
        
        return min_ttc if min_ttc != float('inf') else 999.0
    
    def _distance_to_next_junction(self, train_idx: int) -> float:
        """Calculate distance to next junction"""
        if not self.started[train_idx] or self.arrived[train_idx]:
            return float('inf')
        
        current_pos = self.positions[train_idx]
        
        # Find next junction ahead
        for junction_pos in sorted(self.junctions):
            if junction_pos > current_pos:
                return float(junction_pos - current_pos)
        
        return float('inf')
        
    def _is_in_switch_zone(self, train_idx: int) -> bool:
        """Check if train is in a track switching zone"""
        if not self.started[train_idx] or self.arrived[train_idx] or self.disabled[train_idx]:
            return False
        
        current_pos = self.positions[train_idx]
        
        # Check if near any junction (within switching range)
        for junction_pos in self.junctions:
            if abs(current_pos - junction_pos) <= 10.0:  # 10 units switching zone
                return True
        
        return False
    
    def max_speed(self) -> int:
        """Maximum allowed speed for trains"""
        return self.cfg.get("max_speed", 4)

    def _compute_optimal_speed(self, train_idx: int) -> float:
        """Compute optimal speed for current situation."""
        if not self.started[train_idx] or self.arrived[train_idx] or self.disabled[train_idx]:
            return 0.0
        
        # Consider distance to next obstacle
        dist_next = self._distance_to_next_train(train_idx)
        braking_dist = self.braking_distance_map.get(self.max_speed, 0)
        
        if dist_next < braking_dist * 2:
            return max(1.0, self.speeds[train_idx] - 1)  # Slow down
        elif self._is_in_any_station(train_idx):
            return 0.0  # Stop at station
        else:
            track_limit = self.cfg.get("track_speed_limits", [self.max_speed] * self.n_tracks)[self.tracks[train_idx]]
            return min(track_limit, self.max_speed * 0.8)
    
    def _distance_to_next_junction(self, train_idx: int) -> float:
        """Distance to next junction."""
        pos = self.positions[train_idx]
        nexts = [j for j in self.junctions if j > pos]
        if not nexts:
            return float(self.track_length)
        return float(min(nexts) - pos)
    
    def _compute_track_congestion(self, track_idx: int) -> float:
        """Compute congestion level for a track."""
        active_trains = sum(1 for i in range(self.n_trains) 
                          if self.tracks[i] == track_idx and self.started[i] 
                          and not self.arrived[i] and not self.disabled[i])
        return min(1.0, active_trains / max(1, self.n_trains // self.n_tracks))
    
    def _time_to_collision(self, train_idx: int) -> float:
        """Predict time to collision if current trajectory continues."""
        if not self.started[train_idx] or self.speeds[train_idx] == 0:
            return float('inf')
        
        track = self.tracks[train_idx]
        pos = self.positions[train_idx]
        speed = self.speeds[train_idx]
        
        min_ttc = float('inf')
        for other_idx in range(self.n_trains):
            if other_idx == train_idx or self.tracks[other_idx] != track:
                continue
            if not self.started[other_idx] or self.arrived[other_idx]:
                continue
            
            other_pos = self.positions[other_idx]
            other_speed = self.speeds[other_idx]
            
            # Only consider if we're approaching from behind
            if pos < other_pos:
                relative_speed = speed - other_speed
                if relative_speed > 0:  # We're catching up
                    distance = other_pos - pos - self.train_length
                    if distance > 0:
                        ttc = distance / relative_speed
                        min_ttc = min(min_ttc, ttc)
        
        return min_ttc if min_ttc != float('inf') else 50.0
    
    def _estimate_arrival_time(self, train_idx: int) -> float:
        """Estimate when train will arrive at destination."""
        if not self.started[train_idx] or self.arrived[train_idx]:
            return self.current_step
        
        remaining_dist = self.destinations[train_idx] - self.positions[train_idx]
        if remaining_dist <= 0:
            return self.current_step
        
        avg_speed = max(1.0, self.max_speed * 0.6)
        est_steps = remaining_dist / avg_speed
        
        # Add station halt times
        stations_ahead = sum(1 for (start, end) in self.stations.values() 
                           if start > self.positions[train_idx])
        halt_time = stations_ahead * self.cfg["station_halt_time"]
        
        return self.current_step + est_steps + halt_time
    
    def _predict_downstream_congestion(self, train_idx: int) -> float:
        """Predict congestion in the path ahead."""
        if not self.started[train_idx]:
            return 0.0
        
        pos = self.positions[train_idx]
        track = self.tracks[train_idx]
        look_ahead = 200.0
        
        trains_ahead = 0
        for other_idx in range(self.n_trains):
            if other_idx == train_idx or self.tracks[other_idx] != track:
                continue
            if self.started[other_idx] and not self.arrived[other_idx]:
                other_pos = self.positions[other_idx]
                if pos < other_pos <= pos + look_ahead:
                    trains_ahead += 1
        
        return min(1.0, trains_ahead / 3.0)
    
    def _compute_optimal_action_hint(self, train_idx: int) -> int:
        """Provide hint about optimal action."""
        if not self.started[train_idx] or self.arrived[train_idx] or self.disabled[train_idx]:
            return 0  # no-op
        
        collision_risk = self._compute_collision_risk(train_idx)
        
        if collision_risk > 0.7:
            return 3  # emergency brake
        elif collision_risk > 0.3:
            return 2  # decelerate
        elif self._is_in_any_station(train_idx) and self.halt_remaining[train_idx] == 0:
            return 4  # hold/stop
        elif self.speeds[train_idx] < self._compute_optimal_speed(train_idx):
            return 1  # accelerate
        else:
            return 0  # maintain
    
    def _compute_system_efficiency_trend(self) -> float:
        """Compute trend in system-wide efficiency."""
        window = list(self.performance_window)
        if len(window) < 10:
            return 0.5
        
        recent_performance = [x["system_efficiency"] for x in window[-10:]]
        if len(window) >= 20:
            older_performance = [x["system_efficiency"] for x in window[-20:-10]]
        else:
            older_performance = recent_performance
        
        recent_avg = float(np.mean(recent_performance))
        older_avg = float(np.mean(older_performance))
        
        trend = (recent_avg - older_avg) * 0.5 + 0.5
        return max(0.0, min(1.0, trend))
    
    def _get_nearby_trains(self, train_idx: int, radius: float) -> List[Tuple[int, float, int]]:
        """Get nearby trains within radius, sorted by distance."""
        if not self.started[train_idx]:
            return []
        
        pos = self.positions[train_idx]
        nearby = []
        
        for other_idx in range(self.n_trains):
            if other_idx == train_idx or not self.started[other_idx]:
                continue
            if self.arrived[other_idx] or self.disabled[other_idx]:
                continue
            
            other_pos = self.positions[other_idx]
            distance = abs(pos - other_pos)
            
            if distance <= radius:
                nearby.append((other_idx, distance, self.speeds[other_idx]))
        
        # Sort by distance
        nearby.sort(key=lambda x: x[1])
        return nearby[:2]  # Return closest 2
    
    def _distance_to_next_train(self, train_idx: int) -> float:
        """Distance to next train on same track."""
        if not self.started[train_idx]:
            return float(self.track_length)
        
        track = self.tracks[train_idx]
        pos = self.positions[train_idx]
        distances = []
        
        for other_idx in range(self.n_trains):
            if other_idx == train_idx or self.tracks[other_idx] != track:
                continue
            if not self.started[other_idx] or self.arrived[other_idx] or self.disabled[other_idx]:
                continue
            
            other_pos = self.positions[other_idx]
            if other_pos > pos:  # Only trains ahead
                distances.append(other_pos - pos)
        
        return min(distances) if distances else float(self.track_length)
    
    def _next_signal_for_train(self, train_idx: int) -> Tuple[bool, float]:
        """Get next signal state and distance."""
        pos = self.positions[train_idx]
        next_signals = [p for p in self.signal_positions if p > pos]
        
        if not next_signals:
            return True, float(self.track_length)
        
        next_pos = min(next_signals)
        return bool(self.signal_states.get(next_pos, True)), float(next_pos - pos)
    
    def _is_in_any_station(self, train_idx: int) -> bool:
        """Check if train is in any station."""
        pos = self.positions[train_idx]
        for (start, end) in self.stations.values():
            if start <= pos < end:
                return True
        return False
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Simplified step function with enhanced reward computation."""
        action = np.array(action, dtype=np.int32).flatten()
        
        # Ensure correct action size
        if action.size != self.n_trains:
            raise ValueError(f"Action must have length {self.n_trains}, got {action.size}")
        
        # Store previous state for shaping
        prev_positions = self.positions.copy()
        prev_speeds = self.speeds.copy()
        
        # Process train starts
        for i in range(self.n_trains):
            if self.forced_stop_steps[i] > 0:
                self.speeds[i] = 0
                action[i] = 4  
                self.forced_stop_steps[i] -= 1

            if not self.started[i] and self.current_step >= self.start_times[i]:
                self.started[i] = True
                self.speeds[i] = 0
        
        # Process actions
        action_results = self._process_actions(action)
        
        # Update train positions and states
        movement_results = self._update_train_movement()
        
        # Update network state
        self._update_network_state()
        
        # Handle station logic and arrivals
        arrival_results = self._process_stations_and_arrivals()
        
        # Detect safety events
        safety_results = self._detect_safety_events()
        
        # Compute enhanced reward
        reward_components = {**action_results, **movement_results, 
                            **arrival_results, **safety_results}
        reward, detailed_breakdown = self._compute_reward(reward_components)
        
        # Update performance tracking
        self._update_performance_tracking(reward_components)
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # Prepare info dict
        info = self._prepare_info_dict(reward_components, detailed_breakdown, terminated or truncated)
        
        self.current_step += 1
        
        obs = self._get_enhanced_obs()
        return obs, float(reward), bool(terminated), bool(truncated), info
    
    def _process_actions(self, actions) -> Dict[str, float]:
        """Process actions with enhanced safety and efficiency checks."""
        results = {
            "emergency_brakes": 0,
            "illegal_switches": 0,
            "switch_speed_violations": 0,
            "smooth_accelerations": 0,
            "junction_efficiency_bonus": 0,
            "priority_requests": 0,
        }
        
        for i in range(self.n_trains):
            if not self.started[i] or self.arrived[i] or self.disabled[i]:
                continue
            
            action = int(actions[i])
            prev_speed = self.speeds[i]
            
            # Process action
            if action == 1:  # accelerate
                new_speed = min(self.max_speed, self.speeds[i] + self.cfg["accel_units"])
                self.speeds[i] = new_speed
                
                # Reward smooth acceleration
                if new_speed - prev_speed == 1:
                    results["smooth_accelerations"] += 1
                    
            elif action == 2:  # decelerate
                self.speeds[i] = max(0, self.speeds[i] - self.cfg["decel_units"])
                
            elif action == 3:  # emergency brake
                self.speeds[i] = 0
                self.emergency_brake_count[i] += 1
                results["emergency_brakes"] += 1
                
            elif action == 4:  # hold
                if self.speeds[i] > 0:
                    self.speeds[i] = max(0, self.speeds[i] - self.cfg["decel_units"])
                    
            elif action in (5, 6):  # switch tracks
                if self._is_in_switch_zone(i):
                    if self.speeds[i] <= 1:  # Safe switching speed
                        old_track = self.tracks[i]
                        new_track = max(0, old_track - 1) if action == 5 else min(self.n_tracks - 1, old_track + 1)
                        
                        if self.track_blocked_timer[new_track] == 0:
                            self.tracks[i] = new_track
                            results["junction_efficiency_bonus"] += 1
                        else:
                            results["illegal_switches"] += 1
                    else:
                        results["switch_speed_violations"] += 1
                        results["illegal_switches"] += 1
                else:
                    results["illegal_switches"] += 1
                    
            elif action == 8:  # request priority (new action)
                results["priority_requests"] += 1
                # Could implement priority logic here
            
            # Update acceleration history
            accel = self.speeds[i] - prev_speed
            self.acceleration_history[i].append(accel)
        
        return results
    
    def _update_train_movement(self) -> Dict[str, float]:
        """Update train positions and compute movement-related metrics."""
        results = {
            "total_progress": 0.0,
            "idle_trains": 0,
            "overspeed_violations": 0,
            "headway_violations": 0,
            "optimal_spacing_count": 0,
        }
        
        # Update positions
        for i in range(self.n_trains):
            if not self.started[i] or self.arrived[i] or self.disabled[i]:
                continue
            
            old_pos = self.positions[i]
            self.positions[i] = min(self.track_length, self.positions[i] + self.speeds[i])
            results["total_progress"] += max(0, self.positions[i] - old_pos)
            
            # Check for overspeed
            track_limit = self.cfg.get("track_speed_limits", [self.max_speed] * self.n_tracks)[self.tracks[i]]
            if self.speeds[i] > track_limit:
                results["overspeed_violations"] += 1
            
            # Check for unnecessary idling
            if not self._is_in_any_station(i) and self.speeds[i] == 0 and self.halt_remaining[i] == 0:
                collision_risk = self._compute_collision_risk(i)
                if collision_risk < 0.2:  # No good reason to be idle
                    results["idle_trains"] += 1
            
            # Check headway optimization
            dist_next = self._distance_to_next_train(i)
            optimal_headway = self.cfg["reward_config"]["safety"]["min_safe_distance"]
            braking_dist = self.braking_distance_map.get(self.speeds[i], 0)
            
            if dist_next < optimal_headway:
                results["headway_violations"] += 1
            elif optimal_headway <= dist_next <= optimal_headway + braking_dist:
                results["optimal_spacing_count"] += 1
        
        return results
    
    def _update_network_state(self):
        """Update global network state metrics."""
        # Compute network congestion
        active_trains = sum(1 for i in range(self.n_trains) 
                          if self.started[i] and not self.arrived[i] and not self.disabled[i])
        self.network_congestion = min(1.0, active_trains / max(1, self.n_trains * 0.7))
        
        # Compute system throughput
        completed_trains = sum(1 for i in range(self.n_trains) if self.arrived[i])
        self.system_throughput = completed_trains / max(1, self.current_step / 100)
        
        # Update cascade delay factor
        total_delays = sum(max(0, self._compute_current_delay(i)) for i in range(self.n_trains))
        expected_delays = self.n_trains * 10  # Expected baseline delays
        self.cascade_delay_factor = min(2.0, 1.0 + total_delays / max(1, expected_delays))
    
    def _process_stations_and_arrivals(self) -> Dict[str, float]:
        """Handle station stops and arrival processing."""
        results = {
            "arrivals_on_time": 0,
            "arrivals_early": 0,
            "arrivals_grace": 0,
            "arrivals_late": 0,
            "total_delay_time": 0,
            "station_efficiency": 0,
        }
        
        # Process station halts
        station_stops = 0
        for i in range(self.n_trains):
            if self.halt_remaining[i] > 0:
                self.halt_remaining[i] -= 1
                station_stops += 1
                
                if self.halt_remaining[i] == 0 and not self._can_leave_station(i):
                    self.halt_remaining[i] = 1  # Extend halt
                    self.speeds[i] = 0
            
            # Check for new station entry
            if (self._is_in_any_station(i) and self.halt_remaining[i] == 0 and 
                self._should_halt_now(i)):
                self.halt_remaining[i] = self.cfg["station_halt_time"]
                self.speeds[i] = 0
        
        # Process arrivals
        for i in range(self.n_trains):
            if not self.arrived[i] and self.positions[i] >= self.destinations[i]:
                self.arrived[i] = True
                self.actual_arrival[i] = self.current_step
                
                # Classify arrival punctuality
                delay = self.current_step - self.planned_arrival[i]
                results["total_delay_time"] += max(0, delay)
                
                if delay <= -5:  # Early arrival
                    results["arrivals_early"] += 1
                elif delay <= 2:  # On time
                    results["arrivals_on_time"] += 1
                elif delay <= 10:  # Grace period
                    results["arrivals_grace"] += 1
                else:  # Late
                    results["arrivals_late"] += 1
                
                # Cascade delay propagation
                if delay > 0:
                    self._propagate_cascade_delays(i, delay)
        
        # Station efficiency metric
        expected_stops = len(self.stations) * sum(1 for i in range(self.n_trains) if self.started[i])
        if expected_stops > 0:
            results["station_efficiency"] = 1.0 - (station_stops / expected_stops)
        
        return results
    
    def _propagate_cascade_delays(self, source_train: int, delay: int):
        """Cascade delay propagation."""
        cascade_N = self.cfg["cascade_N"]
        base_propagation = delay / max(1, cascade_N)
        
        for i in range(source_train + 1, min(source_train + cascade_N + 1, self.n_trains)):
            # Decay factor based on distance in schedule
            distance_factor = 1.0 / (1.0 + (i - source_train) * 0.3)
            
            # Track-based propagation (same track gets more delay)
            track_factor = 1.5 if self.tracks[i] == self.tracks[source_train] else 1.0
            
            propagated_delay = int(base_propagation * distance_factor * track_factor)
            self.planned_arrival[i] = min(self.max_steps, self.planned_arrival[i] + propagated_delay)
    
    def _detect_safety_events(self) -> Dict[str, float]:
        """Detect and count safety-related events."""
        results = {
            "collisions": 0,
            "near_misses": 0,
            "signal_violations": 0,
            "safe_distance_maintenance": 0,
        }
        
        # Check for collisions and near misses
        for i in range(self.n_trains):
            if not self.started[i] or self.arrived[i] or self.disabled[i]:
                continue
                
            for j in range(i + 1, self.n_trains):
                if not self.started[j] or self.arrived[j] or self.disabled[j]:
                    continue
                if self.tracks[i] != self.tracks[j]:
                    continue
                
                distance = abs(self.positions[i] - self.positions[j])
                
                if distance < self.train_length:
                    # Collision!
                    results["collisions"] += 1
                    self._episode_collision_count += 1
                    
                    # Disable both trains and block track
                    self.disabled[i] = self.disabled[j] = True
                    track_idx = self.tracks[i]
                    self.track_blocked_timer[track_idx] = 50
                    
                elif distance < self.cfg["reward_config"]["safety"]["min_safe_distance"]:
                    # Near miss
                    results["near_misses"] += 1
                    self._episode_near_miss_count += 1
                    
                elif distance >= self.cfg["reward_config"]["safety"]["min_safe_distance"] * 1.5:
                    # Safe distance maintained
                    results["safe_distance_maintenance"] += 0.5
        
        # Check signal violations
        for i in range(self.n_trains):
            if not self.started[i] or self.arrived[i] or self.disabled[i]:
                continue
            
            # Check if train passed a red signal
            pos = self.positions[i]
            for sig_pos in self.signal_positions:
                if not self.signal_states[sig_pos]:  # Red signal
                    if abs(pos - sig_pos) < 2.0 and self.speeds[i] > 0:
                        results["signal_violations"] += 1
                        self._episode_signal_violations += 1
        
        # Update collision risks
        for i in range(self.n_trains):
            self.collision_risks[i] = self._compute_collision_risk(i)
        
        return results
    
    def _compute_reward(self, components: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Compute reward using simplified scheme."""
        reward_config = self.cfg["reward_config"]
        breakdown = {}
        
        # SAFETY REWARDS (highest priority)
        safety_cfg = reward_config["safety"]
        
        # Collision penalty (severe)
        breakdown["collision_penalty"] = -safety_cfg["collision_penalty"] * components["collisions"]
        
        # Near miss penalty
        breakdown["near_miss_penalty"] = -safety_cfg["near_miss_penalty"] * components["near_misses"]
        
        # Safe distance bonus
        breakdown["safe_distance_bonus"] = safety_cfg["safe_distance_bonus"] * components["safe_distance_maintenance"]
        
        # Emergency brake penalty
        breakdown["emergency_brake_penalty"] = -safety_cfg["emergency_brake_penalty"] * components["emergency_brakes"]
        
        # Signal violation penalty
        breakdown["signal_violation_penalty"] = -safety_cfg["signal_violation_penalty"] * components["signal_violations"]
        
        # EFFICIENCY REWARDS
        efficiency_cfg = reward_config["efficiency"]
        
        # Arrival rewards
        breakdown["on_time_arrival"] = efficiency_cfg["on_time_arrival"] * components["arrivals_on_time"]
        breakdown["grace_arrival"] = efficiency_cfg["grace_arrival"] * components["arrivals_grace"]
        breakdown["late_arrival"] = efficiency_cfg["late_arrival"] * components["arrivals_late"]
        
        # Progress reward
        breakdown["progress_reward"] = components["total_progress"] * 0.5
        
        # Idle penalty
        breakdown["idle_penalty"] = efficiency_cfg["idle_penalty"] * components["idle_trains"]
        
        # Speed efficiency
        breakdown["speed_efficiency"] = efficiency_cfg["speed_efficiency"] * self._compute_speed_efficiency()
        
        # Delay penalty
        breakdown["delay_penalty"] = -components["total_delay_time"] * 2.0 * self.cascade_delay_factor
        
        # FLOW OPTIMIZATION REWARDS
        flow_cfg = reward_config["flow"]
        
        # Smooth operation
        breakdown["smooth_acceleration"] = flow_cfg["smooth_acceleration"] * components["smooth_accelerations"]
        
        # Junction efficiency
        breakdown["junction_efficiency"] = flow_cfg["junction_efficiency"] * components["junction_efficiency_bonus"]
        
        # Headway optimization
        breakdown["headway_optimization"] = flow_cfg["headway_optimization"] * components["optimal_spacing_count"]
        
        # Network flow bonus
        network_efficiency = 1.0 - self.network_congestion + self.system_throughput * 0.1
        breakdown["network_flow"] = flow_cfg["network_flow_bonus"] * network_efficiency
        
        # Potential-based shaping rewards
        total_shaping = 0.0
        for i in range(self.n_trains):
            if self.started[i] and not self.arrived[i] and not self.disabled[i]:
                train_state = {
                    "position": self.positions[i],
                    "destination": self.destinations[i],
                    "current_time": self.current_step,
                    "planned_arrival": self.planned_arrival[i],
                    "collision_risk": self.collision_risks[i],
                }
                total_shaping += self.potential_shaping.get_shaping_reward(train_state, i)
        
        breakdown["potential_shaping"] = total_shaping * 0.1
        
        # Sum all components
        total_reward = sum(breakdown.values())
        
        # Apply scaling factors for severe violations
        if components["collisions"] > 0:
            total_reward *= 2.0  # Double penalty for collisions
        
        if components["total_delay_time"] > 20:
            total_reward *= 0.7  # Penalty for excessive delays
        
        if self.network_congestion > 0.8:
            total_reward *= 0.8  # Penalty for congestion
        
        # Clip reward to prevent extreme values
        total_reward = np.clip(total_reward, -3000.0, 2000.0)
        
        return total_reward, breakdown
    
    def _compute_speed_efficiency(self) -> float:
        """Compute overall speed efficiency metric."""
        total_efficiency = 0.0
        active_trains = 0
        
        for i in range(self.n_trains):
            if self.started[i] and not self.arrived[i] and not self.disabled[i]:
                optimal_speed = self._compute_optimal_speed(i)
                if optimal_speed > 0:
                    efficiency = min(1.0, self.speeds[i] / optimal_speed)
                    total_efficiency += efficiency
                    active_trains += 1
        
        return total_efficiency / max(1, active_trains)
    
    def _update_performance_tracking(self, components: Dict[str, float]):
        """Update performance tracking."""
        # Compute episode metrics
        collision_free = 1.0 if components["collisions"] == 0 else 0.0
        on_time_rate = components["arrivals_on_time"] / max(1, sum([
            components["arrivals_on_time"], 
            components["arrivals_early"],
            components["arrivals_grace"], 
            components["arrivals_late"]
        ]))
        
        system_efficiency = (
            collision_free * 0.4 + 
            on_time_rate * 0.3 + 
            (1.0 - self.network_congestion) * 0.3
        )
        
        # Update performance window
        episode_perf = {
            "collision_free_rate": collision_free,
            "on_time_rate": on_time_rate,
            "system_efficiency": system_efficiency,
        }
        self.performance_window.append(episode_perf)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # All trains arrived or disabled
        if np.all(self.arrived | self.disabled):
            return True
        
        # Too many collisions (safety termination)
        if self._episode_collision_count >= 3:
            return True
        
        # System completely gridlocked
        active_trains = sum(1 for i in range(self.n_trains) 
                          if self.started[i] and not self.arrived[i] and not self.disabled[i])
        if active_trains > 0 and all(self.speeds[i] == 0 for i in range(self.n_trains) if self.started[i]):
            # Check if all are legitimately stopped (in stations)
            legitimate_stops = sum(1 for i in range(self.n_trains) 
                                 if self.started[i] and (self._is_in_any_station(i) or self.halt_remaining[i] > 0))
            if legitimate_stops < active_trains * 0.5:  # More than half illegitimately stopped
                return True
        
        return False
    
    def _calculate_eta_deviation(self, train_id: int) -> float:
        """Calculate how far off the train is from expected arrival time"""
        if self.arrived[train_id]:
            return 0.0
        
        # Estimate remaining time based on current speed and distance
        remaining_dist = self.destinations[train_id] - self.positions[train_id]
        if remaining_dist <= 0:
            return 0.0
        
        current_speed = max(1, self.speeds[train_id])
        estimated_remaining_time = remaining_dist / current_speed
        
        # Compare with planned remaining time
        planned_remaining = max(0, self.planned_arrival[train_id] - self.current_step)
        
        return float(estimated_remaining_time - planned_remaining)
    
    def _prepare_info_dict(self, components: Dict[str, float], breakdown: Dict[str, float], 
                          episode_end: bool) -> Dict[str, Any]:
        """Prepare comprehensive info dictionary."""
        info = {
            "step": self.current_step,
            "active_trains": sum(1 for i in range(self.n_trains) 
                               if self.started[i] and not self.arrived[i] and not self.disabled[i]),
            "collision_count": self._episode_collision_count,
            "near_miss_count": self._episode_near_miss_count,
            "signal_violations": self._episode_signal_violations,
            "network_congestion": self.network_congestion,
            "system_throughput": self.system_throughput,
            "cascade_delay_factor": self.cascade_delay_factor,
            "reward_breakdown": breakdown,
        }
        
        if episode_end:
            total_arrivals = sum([
                components["arrivals_on_time"],
                components["arrivals_early"], 
                components["arrivals_grace"],
                components["arrivals_late"]
            ])
            
            info.update({
                "episode_summary": {
                    "total_trains": self.n_trains,
                    "completed_trains": int(total_arrivals),
                    "completion_rate": total_arrivals / self.n_trains,
                    "on_time_rate": components["arrivals_on_time"] / max(1, total_arrivals),
                    "collision_free": self._episode_collision_count == 0,
                    "safety_score": 1.0 - min(1.0, (self._episode_collision_count + self._episode_near_miss_count * 0.1) / 10),
                    "efficiency_score": components["arrivals_on_time"] / max(1, self.n_trains),
                    "total_delay": components["total_delay_time"],
                    "avg_delay": components["total_delay_time"] / max(1, total_arrivals),
                }
            })
        
        return info
    
    def _calculate_schedule_pressure(self, train_id: int) -> float:
        """Calculate pressure from schedule constraints (0-1 scale)"""
        if self.arrived[train_id]:
            return 0.0
        
        current_delay = self._compute_current_delay(train_id)
        time_remaining = max(1, self.planned_arrival[train_id] - self.current_step)
        
        # Higher pressure means less time buffer
        pressure = min(1.0, current_delay / time_remaining)
        return float(pressure)
    
    def _can_request_priority(self, train_idx: int) -> bool:
        """Check if train can request priority"""
        # Simple logic - can request if near junction and delayed
        return (self._distance_to_next_junction(train_idx) < 50 and
                self._compute_current_delay(train_idx) > 5)
    
    def _can_switch_track(self, train_idx: int) -> bool:
        """Check if train can safely switch tracks"""
        return (self._is_in_switch_zone(train_idx) and 
                self.speeds[train_idx] <= 1 and
                self.collision_risks[train_idx] < 0.3)
    
    def _is_in_switch_zone(self, train_idx: int) -> bool:
        """Check if train is in a switching zone."""
        pos = self.positions[train_idx]
        for junction_pos in self.junctions:
            if abs(pos - junction_pos) <= 3.0:
                return True
        return False
    
    def _should_halt_now(self, train_idx: int) -> bool:
        """Check if train should halt at current station."""
        return (self._is_in_any_station(train_idx) and 
                not self.arrived[train_idx] and 
                self.halt_remaining[train_idx] == 0)
    
    def _can_leave_station(self, train_idx: int) -> bool:
        """Check if train can leave station."""
        pos = self.positions[train_idx]
        
        # Check station exit signals
        for (start, end) in self.stations.values():
            if start <= pos < end:
                signal_pos = end
                if not self.signal_states.get(signal_pos, True):
                    return False
        
        # Check track availability
        if self.track_blocked_timer[self.tracks[train_idx]] > 0:
            return False
        
        return True
    
    def render(self, mode="human"):
        """Enhanced rendering with safety and performance indicators."""
        if mode == "human":
            print(f"\n{'='*80}")
            print(f"Step {self.current_step}")
            print(f"Network Congestion: {self.network_congestion:.2f} | Throughput: {self.system_throughput:.2f}")
            print(f"Collisions: {self._episode_collision_count} | Near Misses: {self._episode_near_miss_count}")
            print(f"Signal Violations: {self._episode_signal_violations}")
            print(f"{'='*80}")
            
            # Track visualization
            for track_idx in range(self.n_tracks):
                line = ['-'] * min(200, int(self.track_length))
                
                # Mark stations
                for name, (start, end) in self.stations.items():
                    for pos in range(int(start), min(int(end), len(line))):
                        if 0 <= pos < len(line):
                            line[pos] = 'S'
                
                # Mark junctions
                for junction_pos in self.junctions:
                    if 0 <= int(junction_pos) < len(line):
                        line[int(junction_pos)] = 'J'
                
                # Mark signals (red signals as 'X')
                for sig_pos in self.signal_positions:
                    if 0 <= int(sig_pos) < len(line):
                        line[int(sig_pos)] = 'X' if not self.signal_states[sig_pos] else 'G'
                
                # Place trains
                for train_idx in range(self.n_trains):
                    if self.tracks[train_idx] != track_idx:
                        continue
                    
                    pos = int(min(self.positions[train_idx], len(line) - 1))
                    if 0 <= pos < len(line):
                        # Color code trains by state
                        if self.disabled[train_idx]:
                            symbol = 'X'  # Crashed
                        elif self.arrived[train_idx]:
                            symbol = '+'  # Arrived
                        elif not self.started[train_idx]:
                            symbol = 'o'  # Not started
                        elif self.collision_risks[train_idx] > 0.7:
                            symbol = '!'  # High risk
                        elif self.speeds[train_idx] == 0:
                            symbol = str(train_idx % 10)  # Stopped
                        else:
                            symbol = str(train_idx % 10)  # Moving
                        
                        line[pos] = symbol
                
                print(f"Track {track_idx}: {''.join(line)}")
                
                # Show train details for this track
                track_trains = [i for i in range(self.n_trains) if self.tracks[i] == track_idx]
                if track_trains:
                    details = []
                    for i in track_trains:
                        status = []
                        if not self.started[i]:
                            status.append("WAIT")
                        elif self.arrived[i]:
                            status.append("DONE")
                        elif self.disabled[i]:
                            status.append("CRASH")
                        else:
                            status.append(f"SPD{self.speeds[i]}")
                            if self.halt_remaining[i] > 0:
                                status.append(f"HALT{self.halt_remaining[i]}")
                            if self.collision_risks[i] > 0.5:
                                status.append("RISK")
                        
                        delay = self._compute_current_delay(i)
                        delay_str = f"D{delay}" if delay > 0 else "OT"
                        details.append(f"T{i}({','.join(status)},{delay_str})")
                    
                    print(f"         {' | '.join(details)}")
            
            print(f"{'='*80}\n")
        
        return None
    
    def save_episode_log(self, filename: str = None) -> str:
        """Save episode log with performance metrics."""
        if filename is None:
            timestamp = int(time.time())
            filename = os.path.join(self.cfg["log_dir"], f"simplified_episode_{timestamp}.csv")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "train_id", "start_time", "planned_arrival", "actual_arrival", 
                "delay", "arrived", "disabled", "collision_risk", "emergency_brakes",
                "avg_speed"
            ])
            
            # Write train data
            for i in range(self.n_trains):
                actual = int(self.actual_arrival[i]) if self.actual_arrival[i] >= 0 else -1
                delay = actual - int(self.planned_arrival[i]) if actual >= 0 else -1
                avg_speed = np.mean([h for h in self.acceleration_history[i]]) if self.acceleration_history[i] else 0
                
                writer.writerow([
                    i, self.start_times[i], self.planned_arrival[i], actual, delay,
                    self.arrived[i], self.disabled[i], self.collision_risks[i],
                    self.emergency_brake_count[i], avg_speed
                ])
            
            # Write episode summary
            writer.writerow([])
            writer.writerow(["Episode Summary"])
            writer.writerow(["Total Steps", self.current_step])
            writer.writerow(["Collisions", self._episode_collision_count])
            writer.writerow(["Near Misses", self._episode_near_miss_count])
            writer.writerow(["Signal Violations", self._episode_signal_violations])
            writer.writerow(["Network Congestion", f"{self.network_congestion:.3f}"])
            writer.writerow(["System Throughput", f"{self.system_throughput:.3f}"])
        
        return filename


# SIMPLIFIED TRAINING UTILITIES

class SimplifiedTrainingManager:
    """Simplified training manager with curriculum learning capabilities."""
    
    def __init__(self, env_config: Dict = None):
        self.env_config = env_config or {}
        self.training_metrics = []
        
    def create_vectorized_env(self, n_envs: int = 8):
        """Create vectorized environment for parallel training."""
        try:
            from stable_baselines3.common.vec_env import SubprocVecEnv
            
            def make_env():
                return RailEnv(config=self.env_config)
            
            vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
            return vec_env
        except ImportError:
            print("Warning: stable-baselines3 not available. Creating single environment.")
            return RailEnv(config=self.env_config)
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get simplified training configuration."""
        return {
            # PPO Hyperparameters
            "algorithm": "PPO",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            
            # Network architecture
            "policy_kwargs": {
                "net_arch": [dict(pi=[256, 256, 128], vf=[256, 256, 128])],
                "activation_fn": "tanh",
                "ortho_init": True,
            },
            
            # Training schedule for each stage
            "total_timesteps": 2_500_000,  # <-- Set this for your current stage (e.g., 500k for Stage 2)
            "eval_freq": 10_000,
            "n_eval_episodes": 10,
            
            # Logging
            "tensorboard_log": "./tensorboard_logs/",
            "verbose": 1,
        }
    
    def setup_callbacks(self):
        """Setup basic training callbacks."""
        try:
            from stable_baselines3.common.callbacks import (
                EvalCallback, CheckpointCallback
            )
            
            callbacks = []
            
            # Evaluation callback
            eval_env = self.create_vectorized_env(n_envs=1)
            eval_callback = EvalCallback(
                eval_env, 
                best_model_save_path='./models/best/',
                log_path='./logs/eval/',
                eval_freq=10000,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
            
            # Checkpoint callback
            checkpoint_callback = CheckpointCallback(
                save_freq=50000,
                save_path='./models/checkpoints/',
                name_prefix='simplified_rail_env'
            )
            callbacks.append(checkpoint_callback)
            
            return callbacks
            
        except ImportError:
            print("Warning: stable-baselines3 callbacks not available")
            return []
    
    # --- THIS IS THE MAIN MODIFIED METHOD ---
    def train(self, resume_from_model: Optional[str] = None, stage_name: str = "stage1"):
        """Train with simplified configuration, with option to resume."""
        print(f"--- Setting up training for: {stage_name} ---")
        
        vec_env = self.create_vectorized_env(n_envs=4)
        config = self.get_training_config()
        
        try:
            from stable_baselines3 import PPO
            
            if resume_from_model:
                # <-- CHANGE: Load the model if a path is provided
                print(f"Resuming training from model: {resume_from_model}")
                model = PPO.load(resume_from_model, env=vec_env)
                # Lowering the learning rate can be helpful for fine-tuning
                # model.learning_rate = 1e-4 
            else:
                # <-- CHANGE: This part is the original model creation
                print("Starting training from scratch...")
                model = PPO(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=config["learning_rate"],
                    n_steps=config["n_steps"],
                    batch_size=config["batch_size"],
                    n_epochs=config["n_epochs"],
                    gamma=config["gamma"],
                    gae_lambda=config["gae_lambda"],
                    clip_range=config["clip_range"],
                    ent_coef=config["ent_coef"],
                    vf_coef=config["vf_coef"],
                    max_grad_norm=config["max_grad_norm"],
                    policy_kwargs=config["policy_kwargs"],
                    tensorboard_log=config["tensorboard_log"],
                    verbose=config["verbose"]
                )
            
            callbacks = self.setup_callbacks()
            
            print(f"Starting training for {config['total_timesteps']} timesteps...")
            
            model.learn(
                total_timesteps=config["total_timesteps"],
                callback=callbacks,
                tb_log_name=f"rail_env_ppo_{stage_name}", # <-- CHANGE: Use a unique name for logs
                reset_num_timesteps=False               # <-- CHANGE: Do NOT reset the step counter
            )
            
            model.save(f"./models/final_{stage_name}_model")
            print(f"Training for {stage_name} completed! Model saved.")
            
            return model
            
        except ImportError:
            print("Error: stable-baselines3 not available.")
            return None


# Example usage and testing
if __name__ == "__main__":
    # --- STEP 1: DEFINE YOUR STAGE ---
    # The model you just trained and saved from the first batch
    # Rename your 'best_model.zip' from Stage 1 to this, or update the path.
    STAGE_1_MODEL_PATH = "./best_model/20250912-074011/best_model.zip" 

    # --- STEP 2: UPDATE YOUR CONFIG FOR THE NEW STAGE ---
    # This will be automatically used by the RailEnv when the trainer starts
    SIMPLIFIED_CONFIG["n_tracks"] = 4
    SIMPLIFIED_CONFIG["n_trains"] = 10 # Moving up from 4 to 7 trains
    SIMPLIFIED_CONFIG["track_speed_limits"] = [4, 4, 4, 4]

    # --- STEP 3: RUN THE TRAINING FOR THE NEW STAGE ---
    print(f"--- Starting Training Plan (Stage 2: 3 Tracks, 7 Trains) ---")
    
    trainer = SimplifiedTrainingManager(env_config=SIMPLIFIED_CONFIG)
    
    # Pass the Stage 1 model path to the train method
    trainer.train(resume_from_model=STAGE_1_MODEL_PATH, stage_name="stage2")
    
    print("\n--- Stage 2 training script finished! ---")