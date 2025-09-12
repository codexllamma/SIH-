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

# Enhanced configuration with advanced reward shaping
ENHANCED_CONFIG = {
    # topology (same as original)
    "n_tracks": 4,
    "n_trains": 6,
    "track_length": 1600,
    "train_length": 25,
    "station_halt_time": 12,
    "max_speed": 4,
    "accel_units": 1,
    "decel_units": 1,
    "max_steps": 1800,
    "stations": {
        "A": (120, 140),
        "B": (760, 780),
        "C": (1320, 1340),
    },
    "junctions": [320, 880, 1160],
    "spawn_points": [0, 40, 80],
    "switch_decision_point": 300,
    "unit_meters": 10.0,
    "timestep_seconds": 2.0,
    "accel_mps2": 0.6,
    "brake_mps2": 1.2,
    "track_speed_limits": [4, 4, 4, 4],
    "cascade_N": 8,
    "log_dir": "runs",
    
    # ENHANCED REWARD SYSTEM
    "reward_v2": {
        # Multi-objective reward components
        "safety": {
            "collision_penalty": -2000.0,
            "near_miss_penalty": -50.0,     # New: penalize close calls
            "safe_distance_bonus": 15.0,     # Reward maintaining safe distance
            "emergency_brake_penalty": -25.0,  # Discourage panic braking
            "signal_violation_penalty": -300.0, # New: strict signal compliance
            "derail_penalty": -1500.0,
            "min_safe_distance": 8.0,        # Minimum safe following distance
        },
        
        "efficiency": {
            "on_time_arrival": 500.0,        # Increased bonus for punctuality
            "early_arrival": 200.0,          # Bonus for early arrival (within reason)
            "grace_arrival": 150.0,          # Moderate bonus for slight delays
            "late_arrival": 25.0,            # Small consolation for very late
            "throughput_bonus": 100.0,       # Per successful train completion
            "speed_efficiency": 2.0,         # Bonus for maintaining good speed
            "idle_penalty": -15.0,           # Increased penalty for unnecessary stops
        },
        
        "flow_optimization": {
            "smooth_acceleration": 5.0,      # Reward gradual speed changes
            "junction_efficiency": 30.0,     # Bonus for efficient switching
            "headway_optimization": 20.0,    # Reward optimal train spacing
            "station_dwell_penalty": -5.0,   # Penalty per extra dwell time
            "network_flow_bonus": 50.0,      # System-wide efficiency metric
        },
        
        # Adaptive curriculum learning weights
        "curriculum": {
            "safety_importance": [1.0, 1.0, 1.0],      # [easy, medium, hard]
            "efficiency_importance": [0.3, 0.7, 1.0],   # Gradually increase efficiency focus
            "flow_importance": [0.1, 0.5, 0.9],        # Advanced optimization comes later
        },
        
        # Dynamic reward scaling
        "scaling": {
            "collision_avoidance_multiplier": 2.0,      # Extra emphasis on safety
            "delay_cascading_prevention": 1.5,          # Prevent cascade effects
            "network_congestion_factor": 1.3,           # Account for system load
        },
        
        # Shaped rewards for better learning
        "shaping": {
            "potential_based": True,                     # Use potential-based shaping
            "distance_to_goal_weight": 10.0,            # Progress toward destination
            "time_to_collision_weight": 100.0,          # Proactive collision avoidance
            "schedule_adherence_weight": 25.0,          # Stay close to planned schedule
        },
        
        # Annealing schedule
        "annealing": {
            "initial_exploration": 0.8,     # Start with high exploration bonus
            "final_exploitation": 1.0,      # End with pure performance
            "exploration_steps": 500_000,   # Steps to transition
            "safety_strictness": {          # Gradually increase safety requirements
                "start": 0.5,
                "end": 2.0,
                "steps": 300_000,
            }
        }
    },
    
    # CURRICULUM LEARNING CONFIGURATION
    "curriculum": {
        "enabled": True,
        "stages": [
            {   # Stage 1: Basic movement and collision avoidance
                "name": "basic_safety",
                "duration_steps": 200_000,
                "n_trains": 3,
                "max_speed": 2,
                "focus": "collision_avoidance",
                "success_threshold": 0.8,  # 80% collision-free episodes
            },
            {   # Stage 2: Efficiency with moderate complexity
                "name": "efficiency_focus",
                "duration_steps": 300_000,
                "n_trains": 4,
                "max_speed": 3,
                "focus": "on_time_performance",
                "success_threshold": 0.7,  # 70% on-time arrivals
            },
            {   # Stage 3: Full complexity optimization
                "name": "full_optimization",
                "duration_steps": -1,      # Indefinite
                "n_trains": 6,
                "max_speed": 4,
                "focus": "system_optimization",
                "success_threshold": 0.9,  # 90% system efficiency
            }
        ]
    },
    
    # ADVANCED OBSERVATION SPACE
    "observation_v2": {
        "include_network_state": True,      # Global system information
        "include_predictive_features": True, # Forward-looking features
        "include_neighbor_states": True,    # Multi-agent awareness
        "temporal_window": 5,               # Steps of history to include
        "spatial_awareness_radius": 200.0,  # Radius for local awareness
    }
}

class PotentialBasedShaping:
    """Implements potential-based reward shaping for stable learning."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.prev_potentials = {}
    
    def compute_potential(self, state: Dict, train_id: int) -> float:
        """Compute potential function value for a train state."""
        potential = 0.0
        
        # Distance to destination potential
        if "position" in state and "destination" in state:
            progress = state["position"] / max(1.0, state["destination"])
            potential += self.config["distance_to_goal_weight"] * progress
        
        # Schedule adherence potential
        if "current_time" in state and "planned_arrival" in state:
            time_buffer = max(0, state["planned_arrival"] - state["current_time"])
            potential += self.config["schedule_adherence_weight"] * time_buffer / 100.0
        
        # Safety potential (inverse of collision risk)
        if "collision_risk" in state:
            safety_potential = 1.0 / (1.0 + state["collision_risk"])
            potential += self.config["time_to_collision_weight"] * safety_potential
        
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

class CurriculumManager:
    """Manages curriculum learning progression."""
    
    def __init__(self, config: Dict):
        self.config = config["curriculum"]
        self.current_stage = 0
        self.stage_start_step = 0
        self.performance_history = deque(maxlen=100)
        
    def get_current_stage(self, global_step: int) -> Dict:
        """Get current curriculum stage configuration."""
        if not self.config["enabled"]:
            return self.config["stages"][-1]  # Return full complexity
        
        stage = self.config["stages"][self.current_stage]
        return stage
    
    def should_advance_stage(self, performance_metrics: Dict) -> bool:
        """Check if ready to advance to next stage."""
        if self.current_stage >= len(self.config["stages"]) - 1:
            return False
        
        current_stage = self.config["stages"][self.current_stage]
        threshold = current_stage["success_threshold"]
        
        # Check performance based on stage focus
        focus = current_stage["focus"]
        if focus == "collision_avoidance":
            success_rate = performance_metrics.get("collision_free_rate", 0.0)
        elif focus == "on_time_performance":
            success_rate = performance_metrics.get("on_time_rate", 0.0)
        else:  # system_optimization
            success_rate = performance_metrics.get("system_efficiency", 0.0)
        
        self.performance_history.append(success_rate)
        
        # Need sustained good performance
        if len(self.performance_history) >= 50:
            avg_performance = np.mean(list(self.performance_history)[-50:])
            return avg_performance >= threshold
        
        return False
    
    def advance_stage(self, global_step: int):
        """Advance to next curriculum stage."""
        if self.current_stage < len(self.config["stages"]) - 1:
            self.current_stage += 1
            self.stage_start_step = global_step
            self.performance_history.clear()
            print(f"Advanced to curriculum stage: {self.config['stages'][self.current_stage]['name']}")

class RailEnv(gym.Env):
    """Enhanced RailEnv with advanced reward shaping and training techniques."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, config: Dict = None, render_mode: str = None):
        super().__init__()
        
        # Merge configurations
        base = copy.deepcopy(ENHANCED_CONFIG)
        if config:
            self._deep_update(base, config)
        self.cfg = base
        
        # Initialize base environment properties
        self._init_base_properties()
        
        # Enhanced components
        self.potential_shaping = PotentialBasedShaping(self.cfg["reward_v2"]["shaping"])
        self.curriculum = CurriculumManager(self.cfg)
        
        # Performance tracking
        self.episode_metrics = []
        self.performance_window = deque(maxlen=100)
        
        # Global step counter for curriculum and annealing
        self._global_step_counter = 0
        
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
        """Initialize basic environment properties."""
        c = self.cfg
        self.n_tracks = int(c["n_tracks"])
        self.n_trains = int(c["n_trains"])
        self.track_length = float(c["track_length"])
        self.train_length = float(c["train_length"])
        self.max_speed = int(c["max_speed"])
        self.max_steps = int(c["max_steps"])
        
        # Layout
        self.stations = {k: tuple(v) for k, v in c["stations"].items()}
        self.junctions = list(c["junctions"])
        self.spawn_points = list(c["spawn_points"])
        
        # Physics
        self.unit_m = float(c["unit_meters"])
        self.timestep_s = float(c["timestep_seconds"])
        self.brake_mps2 = float(c["brake_mps2"])
        
        # Compute speed tables and braking distances
        self._compute_speed_tables()
    
    def _init_spaces(self):
        """Initialize observation and action spaces."""
        # Enhanced action space with more granular control
        # 0=no-op,1=accel,2=decel,3=emergency_brake,4=hold,5=switch_left,6=switch_right,7=skip_station,8=request_priority
        self.n_actions_per_train = 9
        self.action_space = spaces.MultiDiscrete([self.n_actions_per_train] * self.n_trains)
        
        """
        # Enhanced observation space
        obs_config = self.cfg["observation_v2"]
        features_per_train = 20  # Expanded feature set
        
        if obs_config["include_network_state"]:
            features_per_train += 8  # Global network features
        
        if obs_config["include_predictive_features"]:
            features_per_train += 6  # Predictive features
        
        if obs_config["include_neighbor_states"]:
            features_per_train += 10  # Neighbor awareness
        """
        
        
        self.features_per_train = self._count_features_per_train()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_trains * self.features_per_train,),
            dtype=np.float32
        )

    
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
        """Reset environment with curriculum learning support."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Get current curriculum stage
        current_stage = self.curriculum.get_current_stage(self._global_step_counter)
        
        
        # Adapt environment complexity based on curriculum
        self.n_trains = min(current_stage.get("n_trains", self.n_trains), len(self.spawn_points))
        self.max_speed = current_stage.get("max_speed", self.max_speed)
        
        # Reset all state
        self._init_environment_state()
        
        # Initialize trains with curriculum-aware difficulty
        self._initialize_trains(current_stage)
        
        # Reset shaping function
        self.potential_shaping.reset()
        

        obs = self._get_enhanced_obs()
        return obs, {"curriculum_stage": current_stage["name"]}
    
    def _initialize_trains(self, stage_config: Dict):
        """Initialize trains based on curriculum stage."""
        for i in range(self.n_trains):
            self.tracks[i] = int(self.rng.integers(0, self.n_tracks))
            self.positions[i] = float(self.rng.choice(self.spawn_points))
            self.speeds[i] = 0
            
            # Curriculum-based start time variation
            if stage_config["name"] == "basic_safety":
                # Larger gaps between starts in early training
                self.start_times[i] = int(i * 20 + self.rng.integers(0, 10))
            else:
                # More realistic random starts in later stages
                self.start_times[i] = int(self.rng.integers(0, 60))
            
            self.started[i] = bool(self.positions[i] != 0)
            
            # Enhanced arrival planning with uncertainty
            distance_units = max(0.0, self.track_length - self.positions[i])
            nominal_speed = max(1.0, float(self.max_speed) * 0.7)
            est_travel_steps = math.ceil(distance_units / nominal_speed)
            est_halts = len(self.stations) * self.cfg["station_halt_time"]
            
            # Add some planning uncertainty
            uncertainty = int(self.rng.normal(0, 10))
            self.planned_arrival[i] = int(self.start_times[i] + est_travel_steps + est_halts + uncertainty)
    
    def _count_features_per_train(self) -> int:
        """Count features per train based on current obs_config flags."""
        obs_config = self.cfg["observation_v2"]
        base = 20
        if obs_config.get("include_network_state", False):
            base += 8
        if obs_config.get("include_predictive_features", False):
            base += 6
        if obs_config.get("include_neighbor_states", False):
            base += 10  # 2 neighbors × 5 features
        return base

    def _get_enhanced_obs(self) -> np.ndarray:
        """Generate enhanced observations with predictive and network features."""
        obs_config = self.cfg["observation_v2"]
        max_trains = self.cfg.get("max_trains", 6)  # <- NEW: define once in config

        feats = np.zeros((max_trains, self.features_per_train), dtype=np.float32)

        
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
                
                # Schedule pressure (how urgent is adherence)
                pressure = max(0, self.current_step - self.planned_arrival[i] + 50) / 100
                feats[i, feat_idx] = min(1.0, pressure)
                feat_idx += 1
                
                # Downstream congestion prediction
                downstream_risk = self._predict_downstream_congestion(i)
                feats[i, feat_idx] = downstream_risk
                feat_idx += 1
                
                # Optimal action hint (what would optimal controller do)
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
        
        obs = feats.flatten().astype(np.float32)
        
        return obs
    
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
    
    def _compute_current_delay(self, train_idx: int) -> int:
        """Compute current delay for a train."""
        if self.actual_arrival[train_idx] >= 0:
            return max(0, self.actual_arrival[train_idx] - self.planned_arrival[train_idx])
        elif self.started[train_idx]:
            return max(0, self.current_step - self.planned_arrival[train_idx])
        return 0
    
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
            # Balance speed with safety
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
        
        avg_speed = max(1.0, self.max_speed * 0.6)  # Conservative estimate
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
        look_ahead = 200.0  # Look 200 units ahead
        
        trains_ahead = 0
        for other_idx in range(self.n_trains):
            if other_idx == train_idx or self.tracks[other_idx] != track:
                continue
            if self.started[other_idx] and not self.arrived[other_idx]:
                other_pos = self.positions[other_idx]
                if pos < other_pos <= pos + look_ahead:
                    trains_ahead += 1
        
        return min(1.0, trains_ahead / 3.0)  # Normalize by expected capacity
    
    def _compute_optimal_action_hint(self, train_idx: int) -> int:
        """Provide hint about optimal action (for learning guidance)."""
        if not self.started[train_idx] or self.arrived[train_idx] or self.disabled[train_idx]:
            return 0  # no-op
        
        # Simple heuristic controller
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
        """Compute trend in system-wide efficiency based on recent vs older performance."""
        window = list(self.performance_window)  # Convert deque -> list first
        if len(window) < 10:
            return 0.5  # Neutral baseline until we have enough data

        # Extract just the scalar system_efficiency values
        recent_performance = [x["system_efficiency"] for x in window[-10:]]
        if len(window) >= 20:
            older_performance = [x["system_efficiency"] for x in window[-20:-10]]
        else:
            older_performance = recent_performance  # fallback

        recent_avg = float(np.mean(recent_performance))
        older_avg = float(np.mean(older_performance))

        # Normalize trend: 0.5 = stable, >0.5 improving, <0.5 declining
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
        """Enhanced step function with advanced reward computation and padded actions."""
        action = np.array(action, dtype=np.int32).flatten()
        max_trains = self.cfg.get("max_trains", 6)

        # Make sure policy always outputs max_trains actions
        if action.size != max_trains:
            raise ValueError(f"Action must have length {max_trains}, got {action.size}")

        # Slice actions to match active trains
        active_actions = action[:self.n_trains]

        # Store previous state for shaping
        prev_positions = self.positions.copy()
        prev_speeds = self.speeds.copy()

        # Update global step counter
        self._global_step_counter += 1

        # Process train starts
        for i in range(self.n_trains):
            if not self.started[i] and self.current_step >= self.start_times[i]:
                self.started[i] = True
                self.speeds[i] = 0

        # Process actions with enhanced logic
        action_results = self._process_enhanced_actions(active_actions)

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
        reward, detailed_breakdown = self._compute_enhanced_reward(reward_components)

        # Update performance tracking
        self._update_performance_tracking(reward_components)

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps

        # Prepare info dict
        info = self._prepare_info_dict(reward_components, detailed_breakdown, terminated or truncated)

        # Update curriculum if needed
        if terminated or truncated:
            performance_metrics = self._compute_episode_performance()
            if self.curriculum.should_advance_stage(performance_metrics):
                self.curriculum.advance_stage(self._global_step_counter)

        self.current_step += 1

        #Always return padded observation
        obs = self._get_enhanced_obs()

        return obs, float(reward), bool(terminated), bool(truncated), info

    
    def _process_enhanced_actions(self, actions) -> Dict[str, float]:
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
            optimal_headway = self.cfg["reward_v2"]["safety"]["min_safe_distance"]
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
                
                # Cascade delay propagation (enhanced)
                if delay > 0:
                    self._propagate_cascade_delays(i, delay)
        
        # Station efficiency metric
        expected_stops = len(self.stations) * sum(1 for i in range(self.n_trains) if self.started[i])
        if expected_stops > 0:
            results["station_efficiency"] = 1.0 - (station_stops / expected_stops)
        
        return results
    
    def _propagate_cascade_delays(self, source_train: int, delay: int):
        """Enhanced cascade delay propagation."""
        cascade_N = self.cfg["cascade_N"]
        
        # More sophisticated cascade model
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
                    
                elif distance < self.cfg["reward_v2"]["safety"]["min_safe_distance"]:
                    # Near miss
                    results["near_misses"] += 1
                    self._episode_near_miss_count += 1
                    
                elif distance >= self.cfg["reward_v2"]["safety"]["min_safe_distance"] * 1.5:
                    # Safe distance maintained
                    results["safe_distance_maintenance"] += 0.5  # Split between both trains
        
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
    
    def _compute_enhanced_reward(self, components: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Compute reward using enhanced multi-objective scheme."""
        reward_config = self.cfg["reward_v2"]
        
        # Get current curriculum stage for adaptive weighting
        current_stage = self.curriculum.get_current_stage(self._global_step_counter)
        stage_name = current_stage["name"]
        
        # Get curriculum-based importance weights
        curriculum_weights = reward_config["curriculum"]
        if stage_name == "basic_safety":
            safety_weight, efficiency_weight, flow_weight = curriculum_weights["safety_importance"][0], curriculum_weights["efficiency_importance"][0], curriculum_weights["flow_importance"][0]
        elif stage_name == "efficiency_focus":
            safety_weight, efficiency_weight, flow_weight = curriculum_weights["safety_importance"][1], curriculum_weights["efficiency_importance"][1], curriculum_weights["flow_importance"][1]
        else:  # full_optimization
            safety_weight, efficiency_weight, flow_weight = curriculum_weights["safety_importance"][2], curriculum_weights["efficiency_importance"][2], curriculum_weights["flow_importance"][2]
        
        # Annealing factors
        safety_strictness = self._get_annealing_factor("safety_strictness")
        exploration_factor = self._get_annealing_factor("exploration")
        
        breakdown = {}
        
        # SAFETY REWARDS (highest priority)
        safety_cfg = reward_config["safety"]
        
        # Collision penalty (severe)
        breakdown["collision_penalty"] = (-safety_cfg["collision_penalty"] * components["collisions"] * 
                                         safety_weight * safety_strictness)
        
        # Near miss penalty
        breakdown["near_miss_penalty"] = (-safety_cfg["near_miss_penalty"] * components["near_misses"] * 
                                         safety_weight * safety_strictness)
        
        # Safe distance bonus
        breakdown["safe_distance_bonus"] = (safety_cfg["safe_distance_bonus"] * components["safe_distance_maintenance"] * 
                                          safety_weight)
        
        # Emergency brake penalty (discourage panic)
        breakdown["emergency_brake_penalty"] = (-safety_cfg["emergency_brake_penalty"] * components["emergency_brakes"] * 
                                              safety_weight * 0.5)  # Less severe
        
        # Signal violation penalty
        breakdown["signal_violation_penalty"] = (-safety_cfg["signal_violation_penalty"] * components["signal_violations"] * 
                                               safety_weight * safety_strictness)
        
        # EFFICIENCY REWARDS
        efficiency_cfg = reward_config["efficiency"]
        
        # Arrival rewards
        breakdown["on_time_arrival"] = (efficiency_cfg["on_time_arrival"] * components["arrivals_on_time"] * 
                                      efficiency_weight)
        
        breakdown["early_arrival"] = (efficiency_cfg["early_arrival"] * components["arrivals_early"] * 
                                    efficiency_weight * 0.8)  # Slight discount for being too early
        
        breakdown["grace_arrival"] = (efficiency_cfg["grace_arrival"] * components["arrivals_grace"] * 
                                    efficiency_weight)
        
        breakdown["late_arrival"] = (efficiency_cfg["late_arrival"] * components["arrivals_late"] * 
                                   efficiency_weight)
        
        # Progress reward
        breakdown["progress_reward"] = (components["total_progress"] * 0.5 * efficiency_weight)
        
        # Idle penalty
        breakdown["idle_penalty"] = (efficiency_cfg["idle_penalty"] * components["idle_trains"] * 
                                   efficiency_weight)
        
        # Speed efficiency
        breakdown["speed_efficiency"] = (efficiency_cfg["speed_efficiency"] * self._compute_speed_efficiency() * 
                                       efficiency_weight)
        
        # Delay penalty (enhanced)
        delay_penalty = -components["total_delay_time"] * 2.0 * efficiency_weight * self.cascade_delay_factor
        breakdown["delay_penalty"] = delay_penalty
        
        # FLOW OPTIMIZATION REWARDS
        flow_cfg = reward_config["flow_optimization"]
        
        # Smooth operation
        breakdown["smooth_acceleration"] = (flow_cfg["smooth_acceleration"] * components["smooth_accelerations"] * 
                                          flow_weight)
        
        # Junction efficiency
        breakdown["junction_efficiency"] = (flow_cfg["junction_efficiency"] * components["junction_efficiency_bonus"] * 
                                          flow_weight)
        
        # Headway optimization
        breakdown["headway_optimization"] = (flow_cfg["headway_optimization"] * components["optimal_spacing_count"] * 
                                           flow_weight)
        
        # Network flow bonus
        network_efficiency = 1.0 - self.network_congestion + self.system_throughput * 0.1
        breakdown["network_flow"] = (flow_cfg["network_flow_bonus"] * network_efficiency * flow_weight)
        
        # Potential-based shaping rewards
        if reward_config["shaping"]["potential_based"]:
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
            
            breakdown["potential_shaping"] = total_shaping * 0.1  # Scale down shaping
        
        # Sum all components
        total_reward = sum(breakdown.values())
        
        # Apply scaling factors
        scaling_cfg = reward_config["scaling"]
        
        # Extra penalty for safety violations
        if components["collisions"] > 0 or components["signal_violations"] > 0:
            total_reward *= scaling_cfg["collision_avoidance_multiplier"]
        
        # Penalty for cascade delays
        if components["total_delay_time"] > 20:
            total_reward *= (1.0 / scaling_cfg["delay_cascading_prevention"])
        
        # Network congestion factor
        if self.network_congestion > 0.8:
            total_reward *= (1.0 / scaling_cfg["network_congestion_factor"])
        
        # Clip reward to prevent extreme values
        total_reward = np.clip(total_reward, -3000.0, 2000.0)
        
        return total_reward, breakdown
    
    def _get_annealing_factor(self, factor_type: str) -> float:
        """Get annealing factor based on training progress."""
        annealing_cfg = self.cfg["reward_v2"]["annealing"]
        
        if factor_type == "safety_strictness":
            config = annealing_cfg["safety_strictness"]
            progress = min(1.0, self._global_step_counter / config["steps"])
            return config["start"] + (config["end"] - config["start"]) * progress
        
        elif factor_type == "exploration":
            steps = annealing_cfg["exploration_steps"]
            progress = min(1.0, self._global_step_counter / steps)
            initial = annealing_cfg["initial_exploration"]
            final = annealing_cfg["final_exploitation"]
            return initial + (final - initial) * progress
        
        return 1.0
    
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
        """Update performance tracking for curriculum learning."""
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
    
    def _prepare_info_dict(self, components: Dict[str, float], breakdown: Dict[str, float], 
                          episode_end: bool) -> Dict[str, Any]:
        """Prepare comprehensive info dictionary."""
        info = {
            "step": self.current_step,
            "global_step": self._global_step_counter,
            "active_trains": sum(1 for i in range(self.n_trains) 
                               if self.started[i] and not self.arrived[i] and not self.disabled[i]),
            "collision_count": self._episode_collision_count,
            "near_miss_count": self._episode_near_miss_count,
            "signal_violations": self._episode_signal_violations,
            "network_congestion": self.network_congestion,
            "system_throughput": self.system_throughput,
            "cascade_delay_factor": self.cascade_delay_factor,
            "reward_breakdown": breakdown,
            "curriculum_stage": self.curriculum.get_current_stage(self._global_step_counter)["name"],
        }
        
        if episode_end:
            # Comprehensive episode summary
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
    
    def _compute_episode_performance(self) -> Dict[str, float]:
        """Compute episode performance metrics for curriculum."""
        if not self.performance_window:
            return {"collision_free_rate": 0.0, "on_time_rate": 0.0, "system_efficiency": 0.0}
        
        return self.performance_window[-1]  # Return latest performance
    
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
        """Check if train can leave station (signals clear, track not blocked)."""
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
            # Terminal rendering
            print(f"\n{'='*80}")
            print(f"Step {self.current_step} | Global Step {self._global_step_counter}")
            print(f"Stage: {self.curriculum.get_current_stage(self._global_step_counter)['name']}")
            print(f"Network Congestion: {self.network_congestion:.2f} | Throughput: {self.system_throughput:.2f}")
            print(f"Collisions: {self._episode_collision_count} | Near Misses: {self._episode_near_miss_count}")
            print(f"Signal Violations: {self._episode_signal_violations}")
            print(f"{'='*80}")
            
            # Track visualization
            for track_idx in range(self.n_tracks):
                line = ['-'] * min(200, int(self.track_length))  # Limit display width
                
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
    
    def save_enhanced_episode_log(self, filename: str = None) -> str:
        """Save enhanced episode log with performance metrics."""
        if filename is None:
            timestamp = int(time.time())
            stage = self.curriculum.get_current_stage(self._global_step_counter)["name"]
            filename = os.path.join(self.cfg["log_dir"], f"enhanced_episode_{stage}_{timestamp}.csv")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "train_id", "start_time", "planned_arrival", "actual_arrival", 
                "delay", "arrived", "disabled", "collision_risk", "emergency_brakes",
                "track_switches", "station_stops", "avg_speed"
            ])
            
            # Write train data
            for i in range(self.n_trains):
                actual = int(self.actual_arrival[i]) if self.actual_arrival[i] >= 0 else -1
                delay = actual - int(self.planned_arrival[i]) if actual >= 0 else -1
                avg_speed = np.mean([h for h in self.acceleration_history[i]]) if self.acceleration_history[i] else 0
                
                writer.writerow([
                    i, self.start_times[i], self.planned_arrival[i], actual, delay,
                    self.arrived[i], self.disabled[i], self.collision_risks[i],
                    self.emergency_brake_count[i], 0, 0, avg_speed  # TODO: track switches/stops
                ])
            
            # Write episode summary
            writer.writerow([])
            writer.writerow(["Episode Summary"])
            writer.writerow(["Total Steps", self.current_step])
            writer.writerow(["Global Steps", self._global_step_counter])
            writer.writerow(["Curriculum Stage", self.curriculum.get_current_stage(self._global_step_counter)["name"]])
            writer.writerow(["Collisions", self._episode_collision_count])
            writer.writerow(["Near Misses", self._episode_near_miss_count])
            writer.writerow(["Signal Violations", self._episode_signal_violations])
            writer.writerow(["Network Congestion", f"{self.network_congestion:.3f}"])
            writer.writerow(["System Throughput", f"{self.system_throughput:.3f}"])
        
        return filename


# PRODUCTION-GRADE TRAINING UTILITIES

class RailEnvTrainingManager:
    """Production-grade training manager with advanced techniques."""
    
    def __init__(self, env_config: Dict = None):
        self.env_config = env_config or {}
        self.training_metrics = []
        self.best_performance = 0.0
        self.plateau_counter = 0
        self.adaptive_lr_schedule = True
        
    def create_vectorized_env(self, n_envs: int = 8):
        """Create vectorized environment for parallel training."""
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
            from stable_baselines3.common.env_util import make_vec_env
            
            def make_env():
                return RailEnv(config=self.env_config)
            
            # Use SubprocVecEnv for better parallelization
            vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
            return vec_env
        except ImportError:
            print("Warning: stable-baselines3 not available. Creating single environment.")
            return RailEnv(config=self.env_config)
    
    def get_advanced_training_config(self) -> Dict[str, Any]:
        """Get state-of-the-art training configuration."""
        return {
            # PPO Hyperparameters (tuned for multi-agent environments)
            "algorithm": "PPO",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.995,  # Higher gamma for long-term rewards
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "ent_coef": 0.01,  # Encourage exploration
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": 0.01,
            
            # Advanced techniques
            "use_sde": True,  # State Dependent Exploration
            "sde_sample_freq": 4,
            "normalize_advantage": True,
            "policy_kwargs": {
                "net_arch": [dict(pi=[256, 256, 128], vf=[256, 256, 128])],
                "activation_fn": "tanh",
                "ortho_init": True,
            },
            
            # Training schedule
            "total_timesteps": 2_000_000,
            "eval_freq": 10_000,
            "n_eval_episodes": 20,
            
            # Curriculum learning
            "curriculum_enabled": True,
            "curriculum_success_threshold": 0.8,
            
            # Advanced logging
            "tensorboard_log": "./tensorboard_logs/",
            "verbose": 1,
        }
    
    def setup_advanced_callbacks(self):
        """Setup advanced training callbacks."""
        try:
            from stable_baselines3.common.callbacks import (
                EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold,
                ProgressBarCallback
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
                name_prefix='rail_env'
            )
            callbacks.append(checkpoint_callback)
            
            # Early stopping on good performance
            stop_callback = StopTrainingOnRewardThreshold(
                reward_threshold=1500.0,  # Stop when consistently achieving good performance
                verbose=1
            )
            callbacks.append(stop_callback)
            
            # Progress bar
            progress_callback = ProgressBarCallback()
            callbacks.append(progress_callback)
            
            return callbacks
            
        except ImportError:
            print("Warning: stable-baselines3 callbacks not available")
            return []
    
    def train_with_advanced_techniques(self):
        """Train with state-of-the-art techniques."""
        print("Setting up advanced training environment...")
        
        # Create environment
        vec_env = self.create_vectorized_env(n_envs=8)
        
        # Get training config
        config = self.get_advanced_training_config()
        
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_checker import check_env
            
            print("Training configuration:")
            for key, value in config.items():
                if key != "policy_kwargs":
                    print(f"  {key}: {value}")
            
            # Create model with advanced configuration
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
                use_sde=config["use_sde"],
                sde_sample_freq=config["sde_sample_freq"],
                target_kl=config["target_kl"],
                policy_kwargs=config["policy_kwargs"],
                tensorboard_log=config["tensorboard_log"],
                verbose=config["verbose"]
            )
            
            # Setup callbacks
            callbacks = self.setup_advanced_callbacks()
            
            print(f"Starting training for {config['total_timesteps']} timesteps...")
            
            # Train the model
            model.learn(
                total_timesteps=config["total_timesteps"],
                callback=callbacks,
                eval_freq=config["eval_freq"],
                tb_log_name="rail_env_ppo"
            )
            
            # Save final model
            model.save("./models/final_rail_env_model")
            print("Training completed! Model saved to ./models/final_rail_env_model")
            
            return model
            
        except ImportError:
            print("Error: stable-baselines3 not available. Please install it for advanced training.")
            print("pip install stable-baselines3[extra]")
            return None

# Example usage and testing
if __name__ == "__main__":
    # Test enhanced environment
    print("Testing Enhanced RailEnv...")
    
    env = RailEnv()
    obs, info = env.reset(seed=42)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Initial curriculum stage: {info['curriculum_stage']}")
    
    total_reward = 0.0
    for step in range(100):
        # Random actions for testing
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"Step {step}: Reward={reward:.2f}, Active trains={info['active_trains']}")
            env.render()
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            print(f"Episode summary: {info.get('episode_summary', {})}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    env.save_episode_log("runs/test_episode.csv")
    print("Saved log to runs/test_episode.csv | Return:", round(total_reward, 2))