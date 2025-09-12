import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Dict, Any, List, Optional
import gymnasium as gym
from gymnasium import spaces
import csv
import os
import math
import time
import copy

# Default configuration with delay prediction integration
DEFAULT_CONFIG = {
    # topology / geometry
    "n_tracks": 2,
    "n_trains": 4,
    "track_length": 140,           # units
    "train_length": 10,            # units (scaled)
    "station_halt_time": 5,        # timesteps to wait at a station
    "max_speed": 3,                # discrete speeds: 0..max_speed
    "accel": 1,
    "decel": 1,
    "max_steps": 500,              # episode length
    # layout: positions (units)
    "spawn_points": [0, 10],       # trains may spawn at these positions (signal or station)
    "stationA_range": (10, 20),
    "switch1_range": (85, 88),     # junction A->B
    "switch2_range": (97, 100),    # junction B->A
    "stationB_range": (120, 130),
    # switching constraints
    "switch_decision_point": 80,   # must decide before this pos to plan switching (configurable)
    # safety distances (braking distances) per speed
    "braking_distance_map": {3: 6, 2: 3, 1: 1, 0: 0},  # realistic braking distances
    # track-specific speed limits (array length n_tracks)
    "track_speed_limits": [3, 3],  # same for both tracks by default
    # base penalties (kept for semantics; weights in reward block control impact)
    "collision_penalty": -25.0,    # softened to reduce large dips
    "collision_block_recovery": 20, # reduced to avoid long stalls
    "switch_speeding_penalty": -10.0,
    "overspeed_penalty": -10.0,
    "proximity_penalty": -20.0,    # per event if within braking distance
    # OBS controls
    "obs_clip_extra": 1.0,
    # logging
    "log_dir": "runs",
    # Delay prediction parameters
    "delay_prediction": {
        "enabled": True,
        "overlap_threshold_minutes": 2.0,  # Trigger RL when predicted arrivals overlap by this much
        "prediction_horizon": 50,          # Look ahead this many steps
        "update_frequency": 5,             # Update predictions every N steps
        "confidence_level": 0.8,           # Use 80th percentile for predictions
    },
    # >>> New reward controls (dense shaping)
    "reward": {
        # Core shaping
        "w_progress": 0.7,              # increased to prioritize movement
        "w_lateness": 0.02,             # reduced to soften late-game penalty
        "w_idle": 0.02,                # per train per step idle (started, not in station, speed==0)
        "w_safe_spacing_bonus": 0.10,   # per train that exceeds braking distance + margin
        "safe_spacing_margin": 5.0,
        "w_safe_switch_bonus": 0.10,    # per safe switch (in zone, low speed)
        "w_illegal_switch": 0.1,        # new: penalize illegal switch attempts
        # Event penalties (count-based; severity annealed)
        "w_proximity": 0.25,            # per proximity event (within braking distance)
        "w_overspeed": 0.50,           # per overspeed event
        "w_switch_speeding": 0.50,      # per switch-speeding event
        "w_collision": 0.30,            # softened to reduce dip severity
        # Arrival shaping
        "arrival_on_time": 50.0,        # boosted to incentivize arrivals
        "arrival_grace": 30.0,          # boosted
        "arrival_late": 15.0,           # boosted
        "w_arrival_delay": 0.5,         # subtract per unit delay at arrival (per train)
        # Delay prediction rewards
        "w_conflict_avoidance": 1.0,    # bonus for avoiding predicted conflicts
        # Reward clipping per step (min, max). Set to None to disable.
        "clip": (-25.0, 25.0),
        # Linear annealing for penalties (helps exploration early)
        # factor = lerp(start, end, global_steps / steps)
        "anneal_penalties": {"start": 0.6, "end": 1.0, "steps": 100000},  # extended for slower ramp
    },
}

def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def normalize(val: float, max_val: float) -> float:
    if max_val == 0:
        return 0.0
    return float(val) / float(max_val)

class DelayPredictor:
    """Monte Carlo delay prediction model for trains"""
    
    def __init__(self):
        # Default delay parameters (mean, std) for different train types
        # These would typically be learned from historical data
        self.train_delay_params = {
            'A': {'mean': mean_a, 'std': std_a},
            'B': {'mean': mean_b, 'std': std_b},
            'C': {'mean': mean_c, 'std': std_c},
            'default': {'mean': 2.0, 'std': 1.0}
        }
        self.n_scenarios = 1000
        
    def update_delay_parameters(self, train_delays: Dict[str, List[float]]):
        """Update delay parameters from historical data"""
        for train_id, delays in train_delays.items():
            if len(delays) < 3:
                continue
                
            # Remove outliers using z-score
            delays_array = np.array(delays)
            z_scores = np.abs(stats.zscore(delays_array))
            clean_delays = delays_array[z_scores < 2.0]
            
            if len(clean_delays) >= 3:
                self.train_delay_params[train_id] = {
                    'mean': np.mean(clean_delays),
                    'std': np.std(clean_delays)
                }
    
    def predict_delays(self, train_ids: List[str], confidence_level: float = 0.8) -> Dict[str, float]:
        """Predict delays for given trains using Monte Carlo simulation"""
        predictions = {}
        
        for train_id in train_ids:
            # Get parameters for this train type
            params = self.train_delay_params.get(train_id, self.train_delay_params['default'])
            
            # Generate Monte Carlo scenarios
            scenarios = np.maximum(0, np.random.normal(
                params['mean'], params['std'], self.n_scenarios
            ))
            
            # Use specified confidence level (e.g., 80th percentile)
            predicted_delay = np.percentile(scenarios, confidence_level * 100)
            predictions[train_id] = predicted_delay
            
        return predictions
    
    def check_arrival_conflicts(self, scheduled_arrivals: Dict[str, float], 
                              predicted_delays: Dict[str, float],
                              overlap_threshold: float = 2.0) -> List[Tuple[str, str, float]]:
        """Check for potential arrival conflicts due to predicted delays"""
        conflicts = []
        train_ids = list(scheduled_arrivals.keys())
        
        for i, train1 in enumerate(train_ids):
            for train2 in train_ids[i+1:]:
                # Calculate predicted arrival times
                arrival1 = scheduled_arrivals[train1] + predicted_delays.get(train1, 0)
                arrival2 = scheduled_arrivals[train2] + predicted_delays.get(train2, 0)
                
                # Check for overlap
                overlap = overlap_threshold - abs(arrival1 - arrival2)
                if overlap > 0:
                    conflicts.append((train1, train2, overlap))
                    
        return conflicts

class RailEnvWithDelayPrediction(gym.Env):
    """Enhanced Rail Environment with Delay Prediction Integration"""
    
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: Dict = None, render_mode: str = None, 
                 historical_delays: Dict[str, List[float]] = None):
        super().__init__()
        base = copy.deepcopy(DEFAULT_CONFIG)
        if config:
            # shallow update at top level; nested dicts (like reward) get replaced if provided by user
            for k, v in config.items():
                base[k] = v
        self.cfg = base

        c = self.cfg
        self.n_tracks = c["n_tracks"]
        self.n_trains = c["n_trains"]
        self.track_length = c["track_length"]
        self.train_length = c["train_length"]
        self.max_speed = c["max_speed"]
        self.max_steps = c["max_steps"]

        # Initialize delay predictor
        self.delay_predictor = DelayPredictor()
        if historical_delays:
            self.delay_predictor.update_delay_parameters(historical_delays)

        # actions: per-train: 0=no-op,1=accel,2=decel,3=stop,4=switch_left,5=switch_right
        self.n_actions_per_train = 6
        self.action_space = spaces.MultiDiscrete([self.n_actions_per_train] * self.n_trains)

        # Enhanced observation features per train (added conflict prediction features):
        # [track_norm, pos_norm, speed_norm, dest_norm, dist_to_next_train_norm,
        #  next_signal_state, at_station_bool, time_until_start_norm, remaining_halt_norm,
        #  predicted_delay_norm, conflict_risk, time_to_conflict_norm]
        self.features_per_train = 12
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_trains * self.features_per_train,), dtype=np.float32
        )

        # layout positions
        self.stationA_range = tuple(c["stationA_range"])
        self.stationB_range = tuple(c["stationB_range"])
        self.switch1_range = tuple(c["switch1_range"])
        self.switch2_range = tuple(c["switch2_range"])
        self.spawn_points = list(c["spawn_points"])
        self.switch_decision_point = c["switch_decision_point"]

        # dynamic state arrays
        self.tracks = np.zeros(self.n_trains, dtype=np.int32)   # track index per train
        self.positions = np.zeros(self.n_trains, dtype=np.float32)
        self.speeds = np.zeros(self.n_trains, dtype=np.int32)
        self.destinations = np.full(self.n_trains, self.track_length, dtype=np.float32)
        self.halt_remaining = np.zeros(self.n_trains, dtype=np.int32)
        self.started = np.zeros(self.n_trains, dtype=bool)
        self.arrived = np.zeros(self.n_trains, dtype=bool)
        self.disabled = np.zeros(self.n_trains, dtype=bool)    # disabled due to collision
        self.start_times = np.zeros(self.n_trains, dtype=np.int32)
        self.planned_arrival = np.zeros(self.n_trains, dtype=np.int32)
        self.actual_arrival = np.full(self.n_trains, -1, dtype=np.int32)
        
        # Train identifiers for delay prediction
        self.train_ids = [chr(ord('A') + i) for i in range(self.n_trains)]

        # Delay prediction state
        self.predicted_delays = {}
        self.current_conflicts = []
        self.rl_triggered = False
        self.last_prediction_step = 0

        # signals state mapping (positions -> boolean open/closed)
        self.signal_positions = [0, self.stationA_range[1], self.switch1_range[0], 
                                self.switch2_range[0], self.stationB_range[0]]
        self.signal_states = {p: True for p in self.signal_positions}  # True = green initially

        # track-specific blocked state due to collision (timers)
        self.track_blocked_timer = np.zeros(self.n_tracks, dtype=np.int32)

        # logging
        self.current_step = 0
        self._global_step_counter = 0
        self._episode_idx = -1
        self.episode_logs: List[Dict[str, Any]] = []
        self._episode_acc_reward = 0.0
        self._episode_collision_count = 0

        # other
        self.rng = np.random.default_rng()

        # internal RNG-seed placeholder
        self._seed = None

        # finalize
        self.render_mode = render_mode
        os.makedirs(self.cfg["log_dir"], exist_ok=True)

    def _update_delay_predictions(self):
        """Update delay predictions and check for conflicts"""
        if not self.cfg["delay_prediction"]["enabled"]:
            return

        # Only update predictions at specified frequency
        if (self.current_step - self.last_prediction_step < 
            self.cfg["delay_prediction"]["update_frequency"]):
            return

        self.last_prediction_step = self.current_step

        # Get current scheduled arrivals for active trains
        scheduled_arrivals = {}
        for i in range(self.n_trains):
            if not (self.arrived[i] or self.disabled[i]):
                scheduled_arrivals[self.train_ids[i]] = float(self.planned_arrival[i])

        # Predict delays
        confidence = self.cfg["delay_prediction"]["confidence_level"]
        self.predicted_delays = self.delay_predictor.predict_delays(
            list(scheduled_arrivals.keys()), confidence
        )

        # Check for conflicts
        overlap_threshold = self.cfg["delay_prediction"]["overlap_threshold_minutes"]
        self.current_conflicts = self.delay_predictor.check_arrival_conflicts(
            scheduled_arrivals, self.predicted_delays, overlap_threshold
        )

        # Trigger RL intervention if conflicts detected
        self.rl_triggered = len(self.current_conflicts) > 0

    def _get_conflict_features(self, train_idx: int) -> Tuple[float, float, float]:
        """Get conflict-related features for a specific train"""
        train_id = self.train_ids[train_idx]
        
        # Predicted delay (normalized)
        predicted_delay = self.predicted_delays.get(train_id, 0.0)
        predicted_delay_norm = normalize(predicted_delay, 10.0)  # Assume max 10 min delay
        
        # Conflict risk (binary: involved in any conflict)
        conflict_risk = 0.0
        time_to_conflict = float('inf')
        
        for conflict in self.current_conflicts:
            if train_id in conflict[:2]:  # train_id in (train1, train2)
                conflict_risk = 1.0
                # Estimate time to conflict based on current position and speed
                remaining_distance = self.destinations[train_idx] - self.positions[train_idx]
                if self.speeds[train_idx] > 0:
                    time_to_conflict = min(time_to_conflict, remaining_distance / self.speeds[train_idx])
                else:
                    time_to_conflict = min(time_to_conflict, remaining_distance)
        
        # Normalize time to conflict
        time_to_conflict_norm = normalize(
            time_to_conflict if time_to_conflict != float('inf') else 0,
            self.cfg["delay_prediction"]["prediction_horizon"]
        )
        
        return predicted_delay_norm, conflict_risk, time_to_conflict_norm

    def reset(self, *, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)

        self._episode_idx += 1

        # zero state
        self.tracks[:] = 0
        self.positions[:] = 0.0
        self.speeds[:] = 0
        self.halt_remaining[:] = 0
        self.started[:] = False
        self.arrived[:] = False
        self.disabled[:] = False
        self.track_blocked_timer[:] = 0
        self.signal_states = {p: True for p in self.signal_positions}
        self.current_step = 0
        self._episode_acc_reward = 0.0
        self._episode_collision_count = 0
        self.actual_arrival[:] = -1
        self.episode_logs.clear()

        # Reset delay prediction state
        self.predicted_delays = {}
        self.current_conflicts = []
        self.rl_triggered = False
        self.last_prediction_step = 0

        # initialize trains: spawn only at spawn_points (randomly assign), start times random within [0, 10]
        spawn_pos_choices = self.spawn_points
        for i in range(self.n_trains):
            self.tracks[i] = int(self.rng.integers(0, self.n_tracks))
            self.positions[i] = float(self.rng.choice(spawn_pos_choices))
            self.speeds[i] = 0
            self.halt_remaining[i] = 0
            self.started[i] = True if self.positions[i] != 0 else False
            # start_time: if spawn at 0 allow random delay start, else start immediately
            if self.positions[i] == 0:
                self.start_times[i] = int(self.rng.integers(0, 8))  # randomized start delay
                self.started[i] = False
            else:
                self.start_times[i] = 0
                self.started[i] = True

            # planned arrival estimate: account for both stations A and B
            distance = max(0.0, self.track_length - self.positions[i])
            est_travel_steps = math.ceil(distance / max(1, (self.max_speed / 1.0)))
            est_halts = 2 * self.cfg["station_halt_time"]  # A and B
            self.planned_arrival[i] = self.start_times[i] + est_travel_steps + est_halts

        # Initial delay prediction
        self._update_delay_predictions()

        obs = self._get_obs()
        return obs, {"rl_triggered": self.rl_triggered, "conflicts": len(self.current_conflicts)}

    def _get_obs(self) -> np.ndarray:
        feats = np.zeros((self.n_trains, self.features_per_train), dtype=np.float32)
        for i in range(self.n_trains):
            # Original features
            feats[i, 0] = normalize(self.tracks[i], max(self.n_tracks - 1, 1))
            feats[i, 1] = normalize(self.positions[i], self.track_length)
            feats[i, 2] = normalize(self.speeds[i], self.max_speed)
            feats[i, 3] = normalize(self.destinations[i], self.track_length)
            dist_next = self._distance_to_next_train(i)
            feats[i, 4] = normalize(min(dist_next, self.track_length), self.track_length)
            next_signal_state = self._next_signal_state_for_train(i)
            feats[i, 5] = 1.0 if next_signal_state else 0.0
            feats[i, 6] = 1.0 if self._is_in_station(i) else 0.0
            if not self.started[i]:
                time_until = max(0, self.start_times[i] - self.current_step)
                feats[i, 7] = normalize(time_until, self.max_steps)
            else:
                feats[i, 7] = 0.0
            feats[i, 8] = normalize(self.halt_remaining[i], self.cfg["station_halt_time"])
            
            # New delay prediction features
            pred_delay, conflict_risk, time_to_conflict = self._get_conflict_features(i)
            feats[i, 9] = pred_delay
            feats[i, 10] = conflict_risk
            feats[i, 11] = time_to_conflict

        return feats.flatten().astype(np.float32)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Update delay predictions periodically
        self._update_delay_predictions()
        
        # Rest of the step function remains the same as original
        action = np.array(action, dtype=np.int32).flatten()
        if action.size != self.n_trains:
            raise ValueError(f"Action must have length {self.n_trains}, got {action.size}")

        c = self.cfg
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        # decrement track block timers
        self.track_blocked_timer = np.maximum(0, self.track_blocked_timer - 1)

        # start trains whose start_time has arrived
        for i in range(self.n_trains):
            if (not self.started[i]) and (self.current_step >= self.start_times[i]):
                self.started[i] = True
                self.speeds[i] = 0

        # First pass: apply actions to speeds & switching
        switched = np.zeros(self.n_trains, dtype=bool)
        safe_switch = np.zeros(self.n_trains, dtype=bool)
        illegally_switched = np.zeros(self.n_trains, dtype=bool)

        overspeed_events = 0
        switch_speeding_events = 0
        illegal_switch_events = 0
        conflicts_avoided = 0  # New metric for conflict avoidance

        prev_speeds = self.speeds.copy()

        for i in range(self.n_trains):
            if (not self.started[i]) or self.arrived[i] or self.disabled[i]:
                continue

            # Check if this train is involved in a conflict and action helps avoid it
            train_id = self.train_ids[i]
            involved_in_conflict = any(train_id in conflict[:2] for conflict in self.current_conflicts)
            
            a = int(action[i])
            if a == 1:      # accel
                self.speeds[i] = min(self.max_speed, int(self.speeds[i] + c["accel"]))
            elif a == 2:    # decel
                self.speeds[i] = max(0, int(self.speeds[i] - c["decel"]))
                if involved_in_conflict and self.speeds[i] < prev_speeds[i]:
                    conflicts_avoided += 1  # Reward slowing down when in conflict
            elif a == 3:    # stop
                self.speeds[i] = 0
                if self._is_in_station(i):
                    self.halt_remaining[i] = max(self.halt_remaining[i], c["station_halt_time"])
                if involved_in_conflict:
                    conflicts_avoided += 1  # Reward stopping when in conflict
            elif a == 4:    # switch left
                if self._is_in_switch_zone(i):
                    if self.speeds[i] <= 1:
                        old_track = int(self.tracks[i])
                        new_track = max(0, old_track - 1)
                        if self.track_blocked_timer[new_track] == 0:
                            self.tracks[i] = new_track
                            switched[i] = True
                            safe_switch[i] = True
                            if involved_in_conflict:
                                conflicts_avoided += 1  # Reward switching away from conflict
                        else:
                            illegally_switched[i] = True
                            illegal_switch_events += 1
                    else:
                        switch_speeding_events += 1
                        illegally_switched[i] = True
                        illegal_switch_events += 1
                else:
                    illegally_switched[i] = True
                    illegal_switch_events += 1
            elif a == 5:    # switch right
                if self._is_in_switch_zone(i):
                    if self.speeds[i] <= 1:
                        old_track = int(self.tracks[i])
                        new_track = min(self.n_tracks - 1, old_track + 1)
                        if self.track_blocked_timer[new_track] == 0:
                            self.tracks[i] = new_track
                            switched[i] = True
                            safe_switch[i] = True
                            if involved_in_conflict:
                                conflicts_avoided += 1  # Reward switching away from conflict
                        else:
                            illegally_switched[i] = True
                            illegal_switch_events += 1
                    else:
                        switch_speeding_events += 1
                        illegally_switched[i] = True
                        illegal_switch_events += 1
                else:
                    illegally_switched[i] = True
                    illegal_switch_events += 1
            # else: noop or unknown -> no-op

            # track speed limit enforcement (event count)
            track_limit = c["track_speed_limits"][int(self.tracks[i])]
            if self.speeds[i] > track_limit:
                overspeed_events += 1

        # [Rest of the step function continues as in original code...]
        # [Including position updates, signal updates, arrivals, collisions, etc.]
        
        # Second pass: position update (movement)
        prev_positions = self.positions.copy()
        for i in range(self.n_trains):
            if (not self.started[i]) or self.arrived[i] or self.disabled[i]:
                continue
            self.positions[i] = min(float(self.track_length), self.positions[i] + float(self.speeds[i]))
            if self._is_at_station_entry(i) and self.halt_remaining[i] == 0 and self._should_halt_now(i):
                self.halt_remaining[i] = c["station_halt_time"]
                self.speeds[i] = 0

        # Update signals based on occupancy
        for sig_pos in self.signal_positions:
            self.signal_states[sig_pos] = self._is_section_ahead_clear(sig_pos)

        # Third pass: station halt decrement & arrivals
        arrivals_on_time = 0
        arrivals_grace = 0
        arrivals_late = 0
        arrival_delay_sum = 0

        for i in range(self.n_trains):
            if self.halt_remaining[i] > 0:
                self.halt_remaining[i] -= 1
                if self.halt_remaining[i] == 0:
                    if not self._can_leave_station(i):
                        self.halt_remaining[i] = 1
                        self.speeds[i] = 0

            if (not self.arrived[i]) and (self.positions[i] >= self.destinations[i]):
                self.arrived[i] = True
                self.actual_arrival[i] = self.current_step
                delay = int(max(0, self.actual_arrival[i] - self.planned_arrival[i]))
                arrival_delay_sum += delay
                if delay <= 2:
                    arrivals_on_time += 1
                elif delay <= 5:
                    arrivals_grace += 1
                else:
                    arrivals_late += 1

        # Proximity checks & collision detection
        proximity_events = 0
        collision_events = 0
        for t in range(self.n_trains):
            if (not self.started[t]) or self.arrived[t] or self.disabled[t]:
                continue
            for u in range(t + 1, self.n_trains):
                if (not self.started[u]) or self.arrived[u] or self.disabled[u]:
                    continue
                if int(self.tracks[t]) != int(self.tracks[u]):
                    continue
                if self.positions[t] == self.positions[u]:
                    distance = 0.0
                else:
                    distance = abs(self.positions[t] - self.positions[u])
                if distance < float(self.train_length):
                    collision_events += 1
                    self._episode_collision_count += 1
                    track_idx = int(self.tracks[t])
                    self.track_blocked_timer[track_idx] = c["collision_block_recovery"]
                    self.disabled[t] = True
                    self.disabled[u] = True
                else:
                    if self.positions[t] < self.positions[u]:
                        trailing, ahead = t, u
                    else:
                        trailing, ahead = u, t
                    trailing_speed = int(self.speeds[trailing])
                    braking_dist = int(c["braking_distance_map"].get(trailing_speed, 0))
                    if distance <= braking_dist:
                        proximity_events += 1

        # [Continue with rest of original step logic, including reward calculation...]
        
        # Movement delta
        delta_pos = np.sum(np.maximum(0.0, self.positions - prev_positions))

        # Lateness shaping (for active, started trains)
        lateness_total = 0
        idle_count = 0
        for i in range(self.n_trains):
            if self.started[i] and (not self.arrived[i]) and (not self.disabled[i]):
                lateness_total += max(0, self.current_step - self.planned_arrival[i])
                if (not self._is_in_station(i)) and self.speeds[i] == 0:
                    idle_count += 1

        safe_spacing_count = 0
        spacing_margin = float(self.cfg.get("reward", {}).get("safe_spacing_margin", 5.0))

        for i in range(self.n_trains):
            if self.started[i] and (not self.arrived[i]) and (not self.disabled[i]):
                dist_next = self._distance_to_next_train(i)
                braking_dist = int(c["braking_distance_map"].get(int(self.speeds[i]), 0))
                if dist_next > (braking_dist + spacing_margin):
                    safe_spacing_count += 1

        safe_switch_count = int(np.sum(safe_switch))

        # Build reward components and compute reward (enhanced with conflict avoidance)
        comps = {
            "delta_pos": float(delta_pos),
            "lateness_total": float(min(lateness_total, 100 * self.n_trains)),  # capped per train
            "idle_count": int(idle_count),
            "safe_spacing_count": int(safe_spacing_count),
            "safe_switch_count": int(safe_switch_count),
            "proximity_events": int(proximity_events),
            "overspeed_events": int(overspeed_events),
            "switch_speeding_events": int(switch_speeding_events),
            "collision_events": int(collision_events),
            "arrivals_on_time": int(arrivals_on_time),
            "arrivals_grace": int(arrivals_grace),
            "arrivals_late": int(arrivals_late),
            "arrival_delay_sum": int(arrival_delay_sum),
            "illegal_switch_events": int(illegal_switch_events),
            "conflicts_avoided": int(conflicts_avoided),  # New component
        }

        reward, weighted = self._compute_reward(comps)

        # Accumulate
        self._episode_acc_reward += float(reward)
        self._global_step_counter += 1

        # Step counters
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        if np.all(self.arrived | self.disabled):
            terminated = True

        # Package info (enhanced with delay prediction info)
        info["step"] = int(self.current_step)
        info["collision_count"] = int(self._episode_collision_count)
        info["active_trains"] = int(np.sum((~self.arrived) & (~self.disabled)))
        info["delta_pos"] = float(delta_pos)
        info["rl_triggered"] = bool(self.rl_triggered)
        info["current_conflicts"] = len(self.current_conflicts)
        info["predicted_delays"] = dict(self.predicted_delays)
        info["conflicts_avoided"] = int(conflicts_avoided)
        info["reward_breakdown"] = {
            "raw": comps,
            "weighted": weighted,
            "total": float(reward),
            "anneal_factor": float(self._penalty_anneal_factor()),
        }
        
        if terminated or truncated:
            total_trains = int(self.n_trains)
            arrivals_total = int(arrivals_on_time + arrivals_grace + arrivals_late)
            zero_collision = 1 if self._episode_collision_count == 0 else 0
            avg_delay = float(arrival_delay_sum) / max(1, arrivals_total)

            info["kpi_arrivals_total"] = arrivals_total
            info["kpi_arrivals_on_time"] = int(arrivals_on_time)
            info["kpi_arrivals_grace"] = int(arrivals_grace)
            info["kpi_arrivals_late"] = int(arrivals_late)
            info["kpi_avg_delay"] = avg_delay
            info["kpi_zero_collision"] = zero_collision
            info["kpi_completion_rate"] = float(arrivals_total) / max(1, total_trains)
            info["kpi_conflicts_avoided_total"] = sum(log.get("conflicts_avoided", 0) for log in self.episode_logs)
        
        # Per-step log (enhanced with delay prediction data)
        self.episode_logs.append({
            "step": int(self.current_step),
            "reward": float(reward),
            "rl_triggered": bool(self.rl_triggered),
            "conflicts": len(self.current_conflicts),
            "conflicts_avoided": int(conflicts_avoided),
            **{f"raw_{k}": v for k, v in comps.items()},
        })

        obs = self._get_obs()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _penalty_anneal_factor(self) -> float:
        rcfg = self.cfg.get("reward", {})
        sched = rcfg.get("anneal_penalties", None)
        if not sched:
            return 1.0
        start = float(sched.get("start", 1.0))
        end = float(sched.get("end", 1.0))
        steps = float(max(1, sched.get("steps", 1)))
        t = min(1.0, self._global_step_counter / steps)
        return start + (end - start) * t

    def _compute_reward(self, comps: Dict[str, float]):
        """Compute shaped reward from components (enhanced with conflict avoidance). 
        Returns (reward, weighted_components).
        """
        rc = self.cfg.get("reward", {})
        # Weights
        w_progress = float(rc.get("w_progress", 0.5))
        w_lateness = float(rc.get("w_lateness", 0.05))
        w_idle = float(rc.get("w_idle", 0.02))
        w_safe_spacing = float(rc.get("w_safe_spacing_bonus", 0.10))
        w_safe_switch = float(rc.get("w_safe_switch_bonus", 0.10))
        w_proximity = float(rc.get("w_proximity", 0.25))
        w_overspeed = float(rc.get("w_overspeed", 0.50))
        w_switch_spd = float(rc.get("w_switch_speeding", 0.50))
        w_collision = float(rc.get("w_collision", 0.50))
        w_arrival_delay = float(rc.get("w_arrival_delay", 0.5))
        w_illegal_switch = float(rc.get("w_illegal_switch", 0.1))
        w_conflict_avoidance = float(rc.get("w_conflict_avoidance", 1.0))  # New weight

        b_on_time = float(rc.get("arrival_on_time", 30.0))
        b_grace = float(rc.get("arrival_grace", 20.0))
        b_late = float(rc.get("arrival_late", 10.0))

        anneal = self._penalty_anneal_factor()

        # Convert counts to weighted values
        weighted = {}
        weighted["progress"] = w_progress * comps["delta_pos"]
        weighted["lateness"] = - anneal * w_lateness * comps["lateness_total"]
        weighted["idle"] = - anneal * w_idle * comps["idle_count"]
        weighted["safe_spacing_bonus"] = w_safe_spacing * comps["safe_spacing_count"]
        weighted["safe_switch_bonus"] = w_safe_switch * comps["safe_switch_count"]
        weighted["proximity"] = - anneal * w_proximity * comps["proximity_events"]
        weighted["overspeed"] = - anneal * w_overspeed * comps["overspeed_events"]
        weighted["switch_speeding"] = - anneal * w_switch_spd * comps["switch_speeding_events"]
        weighted["collision"] = - anneal * w_collision * comps["collision_events"] * abs(self.cfg.get("collision_penalty", -50.0)) / 50.0
        weighted["illegal_switch"] = - anneal * w_illegal_switch * comps["illegal_switch_events"]
        weighted["conflict_avoidance"] = w_conflict_avoidance * comps["conflicts_avoided"]  # New reward component
        
        # Arrival shaping
        weighted["arrival_bonus"] = (
            b_on_time * comps["arrivals_on_time"]
            + b_grace * comps["arrivals_grace"]
            + b_late * comps["arrivals_late"]
        )
        weighted["arrival_delay"] = - anneal * w_arrival_delay * comps["arrival_delay_sum"]

        total = float(sum(weighted.values()))

        # Clip per-step reward if configured
        clip_cfg = rc.get("clip", None)
        if clip_cfg is not None:
            lo, hi = float(clip_cfg[0]), float(clip_cfg[1])
            total = float(np.clip(total, lo, hi))

        return total, weighted

    # [Rest of the helper methods remain the same as original]
    def _distance_to_next_train(self, idx: int) -> float:
        track = self.tracks[idx]
        pos = self.positions[idx]
        dists = []
        for j in range(self.n_trains):
            if j == idx or self.tracks[j] != track:
                continue
            if self.positions[j] > pos:
                dists.append(self.positions[j] - pos)
        if not dists:
            return float(self.track_length)
        return float(min(dists))

    def _next_signal_state_for_train(self, idx: int) -> bool:
        pos = self.positions[idx]
        next_signals = [p for p in self.signal_positions if p > pos]
        if not next_signals:
            return True
        next_p = min(next_signals)
        return bool(self.signal_states.get(next_p, True))

    def _is_section_ahead_clear(self, sig_pos: float) -> bool:
        """Check if section ahead of signal is clear of trains."""
        for i in range(self.n_trains):
            if self.started[i] and not (self.arrived[i] or self.disabled[i]):
                pos = self.positions[i]
                if sig_pos <= pos < sig_pos + self.cfg["braking_distance_map"][self.max_speed]:
                    return False
        return True

    def _is_in_station(self, idx: int) -> bool:
        pos = self.positions[idx]
        return (self.stationA_range[0] <= pos < self.stationA_range[1]) or (self.stationB_range[0] <= pos < self.stationB_range[1])

    def _is_at_station_entry(self, idx: int) -> bool:
        pos = self.positions[idx]
        return (self.stationA_range[0] <= pos < self.stationA_range[1]) or (self.stationB_range[0] <= pos < self.stationB_range[1])

    def _is_in_switch_zone(self, idx: int) -> bool:
        pos = self.positions[idx]
        return (self.switch1_range[0] <= pos <= self.switch1_range[1]) or (self.switch2_range[0] <= pos <= self.switch2_range[1])

    def _should_halt_now(self, idx: int) -> bool:
        return self._is_in_station(idx) and not self.arrived[idx] and self.halt_remaining[idx] == 0

    def _can_leave_station(self, idx: int) -> bool:
        pos = self.positions[idx]
        if self.stationA_range[0] <= pos < self.stationA_range[1]:
            sigpos = self.stationA_range[1]
        elif self.stationB_range[0] <= pos < self.stationB_range[1]:
            sigpos = self.stationB_range[0]
        else:
            return True
        if not self.signal_states.get(sigpos, True):
            return False
        if self.track_blocked_timer[int(self.tracks[idx])] > 0:
            return False
        return True

    def render(self, mode="human"):
        out_lines = []
        for t in range(self.n_tracks):
            line = ["-" for _ in range(self.track_length + 1)]
            for s in range(self.stationA_range[0], self.stationA_range[1]):
                if 0 <= s <= self.track_length:
                    line[int(s)] = "S"
            for s in range(self.stationB_range[0], self.stationB_range[1]):
                if 0 <= s <= self.track_length:
                    line[int(s)] = "s"
            for x in self.switch1_range:
                if 0 <= x <= self.track_length:
                    line[int(x)] = "J"
            for x in self.switch2_range:
                if 0 <= x <= self.track_length:
                    line[int(x)] = "J"

            for i in range(self.n_trains):
                if int(self.tracks[i]) != t:
                    continue
                pos_idx = int(round(min(self.track_length, self.positions[i])))
                if 0 <= pos_idx <= self.track_length:
                    line[pos_idx] = str((i % 10))
            out_lines.append("".join(line))

        # Enhanced rendering with conflict information
        conflict_info = f" | Conflicts: {len(self.current_conflicts)}"
        rl_status = " | RL: ON" if self.rl_triggered else " | RL: OFF"
        print(f"\nStep {self.current_step} | AccReward={self._episode_acc_reward:.1f} | Collisions={self._episode_collision_count}{conflict_info}{rl_status}")
        
        for idx, ln in enumerate(out_lines):
            print(f"Track {idx}: {ln[:200]}")
        
        if self.current_conflicts:
            print("Active Conflicts:")
            for conflict in self.current_conflicts:
                print(f"  {conflict[0]} vs {conflict[1]}: overlap {conflict[2]:.1f} min")

    def save_episode_log(self, filename: str = None):
        if filename is None:
            filename = os.path.join(self.cfg["log_dir"], f"episode_{int(time.time())}.csv")
        
        # Enhanced logging with delay prediction data
        header = [
            "train_idx", "train_id", "start_time", "planned_arrival", "actual_arrival", 
            "delay", "arrived", "disabled", "predicted_delay", "conflicts_involved"
        ]
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(self.n_trains):
                actual = int(self.actual_arrival[i]) if self.actual_arrival[i] >= 0 else -1
                delay = actual - int(self.planned_arrival[i]) if actual >= 0 else -1
                train_id = self.train_ids[i]
                predicted_delay = self.predicted_delays.get(train_id, 0.0)
                conflicts_count = sum(1 for conflict in self.current_conflicts if train_id in conflict[:2])
                
                writer.writerow([
                    i, train_id, int(self.start_times[i]), int(self.planned_arrival[i]), 
                    actual, delay, bool(self.arrived[i]), bool(self.disabled[i]),
                    predicted_delay, conflicts_count
                ])
        return filename

    def seed(self, seed=None):
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        return [seed]

    def update_historical_delays(self, historical_delays: Dict[str, List[float]]):
        """Update the delay predictor with new historical data"""
        self.delay_predictor.update_delay_parameters(historical_delays)

    def get_delay_predictions(self) -> Dict[str, float]:
        """Get current delay predictions for all active trains"""
        return self.predicted_delays.copy()

    def get_current_conflicts(self) -> List[Tuple[str, str, float]]:
        """Get current arrival conflicts"""
        return self.current_conflicts.copy()


# Usage example and demonstration
def demonstrate_integration():
    """Demonstrate the integrated delay prediction and RL system"""
    
    # Example historical delay data (would typically come from real data)
    historical_delays = {
        'A': [2.1, 3.2, 1.8, 2.9, 3.5, 2.3, 1.9, 2.8, 3.1, 2.2],
        'B': [3.4, 2.8, 4.1, 3.0, 2.6, 3.8, 3.2, 2.9, 3.6, 3.3],
        'C': [2.9, 3.1, 2.4, 3.3, 2.7, 2.8, 3.0, 2.6, 3.2, 2.5],
        'D': [1.8, 2.2, 1.9, 2.4, 2.1, 1.7, 2.0, 2.3, 1.6, 2.2]
    }
    
    # Create environment with delay prediction enabled
    config = {
        "delay_prediction": {
            "enabled": True,
            "overlap_threshold_minutes": 2.0,
            "prediction_horizon": 50,
            "update_frequency": 5,
            "confidence_level": 0.8
        },
        "reward": {
            "w_conflict_avoidance": 2.0,  # Higher weight for conflict avoidance
        }
    }
    
    env = RailEnvWithDelayPrediction(config=config, historical_delays=historical_delays)
    obs, info = env.reset(seed=42)
    
    print("=== Integrated Delay Prediction + RL Rail Environment ===")
    print(f"Enhanced observation space: {env.observation_space.shape}")
    print(f"RL Triggered: {info['rl_triggered']}")
    print(f"Initial Conflicts: {info['conflicts']}")
    
    total_reward = 0.0
    conflicts_avoided_total = 0
    
    for step in range(50):
        # Sample action (in practice, this would come from your RL agent)
        actions = env.action_space.sample()
        
        obs, reward, done, trunc, info = env.step(actions)
        total_reward += reward
        conflicts_avoided_total += info.get("conflicts_avoided", 0)
        
        if step % 10 == 0 or info.get("rl_triggered", False):
            print(f"\nStep {step}:")
            print(f"  RL Triggered: {info.get('rl_triggered', False)}")
            print(f"  Active Conflicts: {info.get('current_conflicts', 0)}")
            print(f"  Conflicts Avoided: {info.get('conflicts_avoided', 0)}")
            print(f"  Predicted Delays: {info.get('predicted_delays', {})}")
            print(f"  Step Reward: {reward:.2f}")
        
        if done or trunc:
            break
    
    print(f"\n=== Episode Summary ===")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Conflicts Avoided: {conflicts_avoided_total}")
    print(f"Final Conflicts: {info.get('current_conflicts', 0)}")
    
    # Save enhanced log
    log_file = env.save_episode_log()
    print(f"Enhanced episode log saved to: {log_file}")
    
    return env


if __name__ == "__main__":
    # Run the demonstration
    env = demonstrate_integration()
    
    # Example of how to use the enhanced features
    print("\n=== Delay Prediction Features ===")
    predictions = env.get_delay_predictions()
    conflicts = env.get_current_conflicts()
    
    print(f"Current Delay Predictions: {predictions}")
    print(f"Current Conflicts: {conflicts}")
    
    # Update with new historical data
    new_delays = {'A': [2.5, 3.0], 'B': [3.2, 2.8]}
    env.update_historical_delays(new_delays)
    print("Updated delay predictor with new historical data")
