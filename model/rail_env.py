
"""
RailEnv - Gymnasium environment for 2-track / 4-train toy railway optimization.

Features:
- Homogeneous trains (configurable)
- Track layout with stations and two junctions (switch zones)
- Signals (boolean), braking-distance safety checks, switching rules
- Schedule (start_time + planned arrival), completion/delay rewards
- Actions per-train: noop / accel / decel / stop / switch_left / switch_right
- Observations: flattened per-train normalized features
- Logging and a simple text renderer
"""

from typing import Tuple, Dict, Any, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import csv
import os
import math
import time

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
    "braking_distance_map": {3: 40, 2: 20, 1: 10, 0: 0},
    # track-specific speed limits (array length n_tracks)
    "track_speed_limits": [3, 3],  # same for both tracks by default
    # collision / blocking
    "collision_penalty": -50.0,
    "collision_block_recovery": 50,  # timesteps that track remains blocked after collision
    # reward weights (normalized)
    "move_reward_per_unit": 1.0,    # reward for forward progress (per unit moved)
    "switch_speeding_penalty": -10.0,
    "overspeed_penalty": -10.0,
    "proximity_penalty": -20.0,     # per-timestep if within braking distance
    "arrival_on_time_bonus": 100.0,
    "arrival_grace_bonus": 80.0,
    "arrival_late_bonus": 50.0,
    "delay_penalty_per_timestep": -2.0,
    # observation normals
    "obs_clip_extra": 1.0,
    # logging
    "log_dir": "runs",
}


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def normalize(val: float, max_val: float) -> float:
    if max_val == 0:
        return 0.0
    return float(val) / float(max_val)


class RailEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: Dict = None, render_mode: str = None):
        super().__init__()
        self.cfg = DEFAULT_CONFIG.copy()
        if config:
            self.cfg.update(config)

        c = self.cfg
        self.n_tracks = c["n_tracks"]
        self.n_trains = c["n_trains"]
        self.track_length = c["track_length"]
        self.train_length = c["train_length"]
        self.max_speed = c["max_speed"]
        self.max_steps = c["max_steps"]

        # actions: per-train: 0=no-op,1=accel,2=decel,3=stop,4=switch_left,5=switch_right
        self.n_actions_per_train = 6
        self.action_space = spaces.MultiDiscrete([self.n_actions_per_train] * self.n_trains)

        # observation features per traine:
        # [track_norm, pos_norm, speed_norm, dest_norm, dist_to_next_train_norm,
        #  next_signal_state, at_station_bool, time_until_start_norm, remaining_halt_norm]
        self.features_per_train = 9
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

        # signals state mapping (positions -> boolean open/closed)
        self.signal_positions = [0, self.stationA_range[1], self.switch1_range[0], self.switch2_range[0], self.stationB_range[0]]
        self.signal_states = {p: True for p in self.signal_positions}  # True = green initially

        # track-specific blocked state due to collision (timers)
        self.track_blocked_timer = np.zeros(self.n_tracks, dtype=np.int32)

        # logging
        self.current_step = 0
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

    
    def reset(self, *, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)

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

        # initialize trains: spawn only at spawn_points (randomly assign), start times random within [0, 10]
        spawn_pos_choices = self.spawn_points
        for i in range(self.n_trains):
            self.tracks[i] = int(self.rng.integers(0, self.n_tracks))
            # pick spawn point index round-robin / random
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

            # planned arrival estimate: simple linear estimate using max_speed
            distance = max(0.0, self.track_length - self.positions[i])
            est_travel_steps = math.ceil(distance / max(1, (self.max_speed / 1.0)))
            est_halts = 1 * self.cfg["station_halt_time"]  # single intermediate station
            self.planned_arrival[i] = self.start_times[i] + est_travel_steps + est_halts

        # return obs
        obs = self._get_obs()
        return obs, {}

    
    def _get_obs(self) -> np.ndarray:
        """
        Build flattened observation vector of size (n_trains * features_per_train,)
        All features normalized to [0,1].
        """
        feats = np.zeros((self.n_trains, self.features_per_train), dtype=np.float32)
        for i in range(self.n_trains):
            # track id normalized
            feats[i, 0] = normalize(self.tracks[i], max(self.n_tracks - 1, 1))
            # position normalized by track length
            feats[i, 1] = normalize(self.positions[i], self.track_length)
            # speed normalized
            feats[i, 2] = normalize(self.speeds[i], self.max_speed)
            # dest normalized (destination is fixed at track_length)
            feats[i, 3] = normalize(self.destinations[i], self.track_length)
            # distance to next train ahead on same track
            dist_next = self._distance_to_next_train(i)
            feats[i, 4] = normalize(min(dist_next, self.track_length), self.track_length)
            # next signal state (boolean) for nearest upcoming signal on that track (simple)
            next_signal_state = self._next_signal_state_for_train(i)
            feats[i, 5] = 1.0 if next_signal_state else 0.0
            # at station bool
            feats[i, 6] = 1.0 if self._is_in_station(i) else 0.0
            # time until start normalized (if not yet started)
            if not self.started[i]:
                time_until = max(0, self.start_times[i] - self.current_step)
                feats[i, 7] = normalize(time_until, self.max_steps)
            else:
                feats[i, 7] = 0.0
            # remaining halt normalized
            feats[i, 8] = normalize(self.halt_remaining[i], self.cfg["station_halt_time"])

        return feats.flatten().astype(np.float32)

    def _distance_to_next_train(self, idx: int) -> float:
        """
        Distance along track to the nearest train ahead on the same track.
        If none ahead, returns a big number (track_length).
        """
        track = self.tracks[idx]
        pos = self.positions[idx]
        # consider trains ahead only (pos > current pos)
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
        """
        Determine nearest signal ahead of train position and return its state (boolean).
        For now signals are same for both tracks and computed from self.signal_states map.
        """
        pos = self.positions[idx]
        # find first signal position > pos
        next_signals = [p for p in self.signal_positions if p > pos]
        if not next_signals:
            # no upcoming signal - treat as green
            return True
        next_p = min(next_signals)
        return bool(self.signal_states.get(next_p, True))


    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        action: array-like length n_trains, each in {0..n_actions_per_train-1}
        returns: obs, reward, terminated, truncated, info
        """
        action = np.array(action, dtype=np.int32).flatten()
        if action.size != self.n_trains:
            raise ValueError(f"Action must have length {self.n_trains}, got {action.size}")

        c = self.cfg
        reward = 0.0
        info: Dict[str, Any] = {}
        terminated = False
        truncated = False

        # decrement track block timers
        blocked_mask = self.track_blocked_timer > 0
        self.track_blocked_timer = np.maximum(0, self.track_blocked_timer - 1)

        # start trains whose start_time has arrived
        for i in range(self.n_trains):
            if (not self.started[i]) and (self.current_step >= self.start_times[i]):
                self.started[i] = True
                # small readiness delay (optional)
                self.speeds[i] = 0

        # apply actions (first pass: compute intended speed & switching requests)
        switched = np.zeros(self.n_trains, dtype=bool)
        illegally_switched = np.zeros(self.n_trains, dtype=bool)
        prox_penalty_this_step = 0.0
        overspeed_penalty = 0.0
        switch_speed_penalty = 0.0

        # action semantics applied only to trains that are started and not arrived/disabled
        for i in range(self.n_trains):
            if (not self.started[i]) or self.arrived[i] or self.disabled[i]:
                continue

            a = int(action[i])
            # accelerate
            if a == 1:
                self.speeds[i] = min(self.max_speed, int(self.speeds[i] + c["accel"]))
            # decelerate
            elif a == 2:
                self.speeds[i] = max(0, int(self.speeds[i] - c["decel"]))
            # stop (emergency/hold)
            elif a == 3:
                # force speed zero and set halt if in station zone
                self.speeds[i] = 0
                if self._is_in_station(i):
                    self.halt_remaining[i] = max(self.halt_remaining[i], c["station_halt_time"])
            # switch left
            elif a == 4:
                # allowed only in switch zones and if not overspeeding
                if self._is_in_switch_zone(i):
                    # check speed constraint for safe switching
                    if self.speeds[i] <= 1:
                        # perform track switch (wrap-around)
                        old_track = int(self.tracks[i])
                        new_track = max(0, old_track - 1)
                        # ensure the new track isn't blocked
                        if self.track_blocked_timer[new_track] == 0:
                            self.tracks[i] = new_track
                            switched[i] = True
                        else:
                            illegally_switched[i] = True
                    else:
                        # penalty for switching at high speed - derail risk
                        switch_speed_penalty += c["switch_speeding_penalty"]
                        illegally_switched[i] = True
                else:
                    # illegal attempt to switch (not in switch zone)
                    illegally_switched[i] = True
            # switch right
            elif a == 5:
                if self._is_in_switch_zone(i):
                    if self.speeds[i] <= 1:
                        old_track = int(self.tracks[i])
                        new_track = min(self.n_tracks - 1, old_track + 1)
                        if self.track_blocked_timer[new_track] == 0:
                            self.tracks[i] = new_track
                            switched[i] = True
                        else:
                            illegally_switched[i] = True
                    else:
                        switch_speed_penalty += c["switch_speeding_penalty"]
                        illegally_switched[i] = True
                else:
                    illegally_switched[i] = True
            # noop (0) - do nothing (maintain speed)
            elif a == 0:
                pass
            else:
                # unknown action -> treat as no-op
                pass

            # enforce track speed limit penalty
            track_limit = c["track_speed_limits"][int(self.tracks[i])]
            if self.speeds[i] > track_limit:
                overspeed_penalty += c["overspeed_penalty"]

        # second pass: position update (movement)
        prev_positions = self.positions.copy()
        for i in range(self.n_trains):
            if (not self.started[i]) or self.arrived[i] or self.disabled[i]:
                continue
            # advance by speed units
            self.positions[i] = min(float(self.track_length), self.positions[i] + float(self.speeds[i]))
            # if reached station region and has to halt
            if self._is_at_station_entry(i) and self.halt_remaining[i] == 0 and self._should_halt_now(i):
                # start halt
                self.halt_remaining[i] = c["station_halt_time"]
                self.speeds[i] = 0

        # third pass: decrement halts & finalize arrivals
        for i in range(self.n_trains):
            if self.halt_remaining[i] > 0:
                self.halt_remaining[i] -= 1
                # only leave when halt_remaining == 0 and exit signal green
                if self.halt_remaining[i] == 0:
                    # leaving station only if exit signal is green and not blocked
                    if not self._can_leave_station(i):
                        # remain halted until allowed
                        self.halt_remaining[i] = 1  # wait another step
                        self.speeds[i] = 0

            # check arrival at destination
            if (not self.arrived[i]) and (self.positions[i] >= self.destinations[i]):
                self.arrived[i] = True
                self.actual_arrival[i] = self.current_step
                # compute arrival reward immediately (plussed to reward)
                delay = int(max(0, self.actual_arrival[i] - self.planned_arrival[i]))
                if delay <= 2:
                    reward += c["arrival_on_time_bonus"]
                elif delay <= 5:
                    reward += c["arrival_grace_bonus"]
                else:
                    reward += c["arrival_late_bonus"]
                # also apply delay penalty (negative)
                reward += c["delay_penalty_per_timestep"] * float(delay)

        # IMPROVED proximity checks & collision detection with graduated penalties
        collision_happened = False
        safe_following_bonus = 0.0
        
        for t in range(self.n_trains):
            if (not self.started[t]) or self.arrived[t] or self.disabled[t]:
                continue
            for u in range(t + 1, self.n_trains):
                if (not self.started[u]) or self.arrived[u] or self.disabled[u]:
                    continue
                if int(self.tracks[t]) != int(self.tracks[u]):
                    continue
                
                # order by position for proper distance calculation
                if self.positions[t] == self.positions[u]:
                    distance = 0.0
                else:
                    distance = abs(self.positions[t] - self.positions[u])
                
                # identify trailing and ahead trains for braking distance calculation
                if self.positions[t] < self.positions[u]:
                    trailing, ahead = t, u
                else:
                    trailing, ahead = u, t
                    
                trailing_speed = int(self.speeds[trailing])
                ahead_speed = int(self.speeds[ahead])
                braking_dist = float(c["braking_distance_map"].get(trailing_speed, 0))
                
                # GRADUATED PENALTY SYSTEM - much more stable than binary collision/no-collision
                if distance < float(self.train_length):
                    # COLLISION: Scale penalty based on overlap severity instead of fixed huge penalty
                    overlap_ratio = 1.0 - (distance / float(self.train_length))
                    base_collision_penalty = c.get("collision_penalty", -200)  # Much smaller than before
                    scaled_collision_penalty = base_collision_penalty * (0.5 + 0.5 * overlap_ratio)
                    
                    collision_happened = True
                    reward += scaled_collision_penalty
                    self._episode_collision_count += 1
                    
                    # Force emergency braking instead of immediate disabling
                    self.speeds[t] = 0
                    self.speeds[u] = 0
                    
                    # Block track for shorter recovery time to allow learning
                    track_idx = int(self.tracks[t])
                    self.track_blocked_timer[track_idx] = max(1, c.get("collision_block_recovery", 5) // 2)
                    
                    # Only disable trains after repeated collisions at same position
                    if hasattr(self, '_collision_positions'):
                        pos_key = f"{int(self.positions[t])}_{int(self.positions[u])}"
                        if pos_key in self._collision_positions:
                            self._collision_positions[pos_key] += 1
                            if self._collision_positions[pos_key] > 2:  # After 3rd collision at same spot
                                self.disabled[t] = True
                                self.disabled[u] = True
                        else:
                            self._collision_positions[pos_key] = 1
                    else:
                        self._collision_positions = {f"{int(self.positions[t])}_{int(self.positions[u])}": 1}
                        
                elif distance < float(self.train_length) * 1.5:
                    # CRITICAL PROXIMITY: Very close but not colliding
                    critical_penalty = c.get("critical_proximity_penalty", -50)
                    reward += critical_penalty
                    prox_penalty_this_step += critical_penalty
                    
                elif distance <= braking_dist:
                    # UNSAFE FOLLOWING DISTANCE: Within braking distance
                    unsafe_penalty = c.get("proximity_penalty", -10)
                    reward += unsafe_penalty
                    prox_penalty_this_step += unsafe_penalty
                    
                elif distance <= braking_dist * 1.5:
                    # SAFE FOLLOWING DISTANCE: Close but safe - reward this!
                    safe_bonus = c.get("safe_following_bonus", 5)
                    safe_following_bonus += safe_bonus
                    reward += safe_bonus
                    
                # Additional penalty if ahead train is much slower and trailing train isn't slowing
                if distance <= braking_dist * 2 and trailing_speed > ahead_speed + 1:
                    # Trailing train should start slowing down when approaching slower train
                    approach_penalty = c.get("approach_penalty", -5)
                    reward += approach_penalty

        # finalize movement reward (sum of position deltas)
        delta_pos = np.sum(np.maximum(0.0, self.positions - prev_positions))
        reward += delta_pos * c["move_reward_per_unit"]
        reward += overspeed_penalty
        reward += switch_speed_penalty

        # schedule delay penalties per active train not yet arrived (penalize waiting if started)
        for i in range(self.n_trains):
            if self.started[i] and not self.arrived[i] and not self.disabled[i]:
                # if current time > planned arrival (i.e., running late already) penalize
                est_remaining = max(0, self.planned_arrival[i] - self.current_step)
                # not too aggressive here, major penalty occurs at actual arrival
                # small per-step penalty encourages timeliness
                reward += 0.0  # keep small or zero to avoid overfitting to speed hack

        # accumulate episode reward
        self._episode_acc_reward += float(reward)

        # step counters
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        # termination: all trains arrived OR all trains disabled OR truncated
        if np.all(self.arrived | self.disabled):
            terminated = True

        # package info
        info["step"] = int(self.current_step)
        info["collision_count"] = int(self._episode_collision_count)
        info["active_trains"] = int(np.sum((~self.arrived) & (~self.disabled)))
        info["delta_pos"] = float(delta_pos)
        info["prox_penalty"] = float(prox_penalty_this_step)
        info["safe_following_bonus"] = float(safe_following_bonus)  # New metric

        obs = self._get_obs()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _is_in_station(self, idx: int) -> bool:
        pos = self.positions[idx]
        return (self.stationA_range[0] <= pos < self.stationA_range[1]) or (self.stationB_range[0] <= pos < self.stationB_range[1])

    def _is_at_station_entry(self, idx: int) -> bool:
        pos = self.positions[idx]
        # entering station A or B from approach (simple)
        return pos < self.stationA_range[1] and pos >= self.stationA_range[0] or pos < self.stationB_range[1] and pos >= self.stationB_range[0]

    def _is_in_switch_zone(self, idx: int) -> bool:
        pos = self.positions[idx]
        return (self.switch1_range[0] <= pos <= self.switch1_range[1]) or (self.switch2_range[0] <= pos <= self.switch2_range[1])

    def _should_halt_now(self, idx: int) -> bool:
        """
        Decide whether train should halt at station (simple policy: if inside station range and arrival not recorded)
        """
        return self._is_in_station(idx) and not self.arrived[idx] and self.halt_remaining[idx] == 0

    def _can_leave_station(self, idx: int) -> bool:
        """
        Can leave if exit signal for that station is green and track is not blocked
        """
        # find station exit signal (approx)
        # station A exit pos = stationA_range[1], station B entry pos = stationB_range[0]
        pos = self.positions[idx]
        # for simplicity, when leaving station A check signal at stationA_range[1], station B at stationB_range[0]
        if self.stationA_range[0] <= pos < self.stationA_range[1]:
            sigpos = self.stationA_range[1]
        elif self.stationB_range[0] <= pos < self.stationB_range[1]:
            sigpos = self.stationB_range[0]
        else:
            return True
        if not self.signal_states.get(sigpos, True):
            return False
        # check track not blocked
        if self.track_blocked_timer[int(self.tracks[idx])] > 0:
            return False
        return True

    def render(self, mode="human"):
        """
        Simple textual rendering. Shows per-track rails with train positions.
        """
        out_lines = []
        for t in range(self.n_tracks):
            line = ["-" for _ in range(self.track_length + 1)]
            # place stations and switches markers
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

            # place trains
            for i in range(self.n_trains):
                if int(self.tracks[i]) != t:
                    continue
                pos_idx = int(round(min(self.track_length, self.positions[i])))
                if 0 <= pos_idx <= self.track_length:
                    line[pos_idx] = str((i % 10))
            out_lines.append("".join(line))

        print(f"\nStep {self.current_step} | AccReward={self._episode_acc_reward:.1f} | Collisions={self._episode_collision_count}")
        for idx, ln in enumerate(out_lines):
            print(f"Track {idx}: {ln[:200]}")  

    def save_episode_log(self, filename: str = None):
        """
        Dump simple episode log: planned arrival, actual arrival, delay for each train.
        """
        if filename is None:
            filename = os.path.join(self.cfg["log_dir"], f"episode_{int(time.time())}.csv")
        header = ["train_idx", "start_time", "planned_arrival", "actual_arrival", "delay", "arrived", "disabled"]
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(self.n_trains):
                actual = int(self.actual_arrival[i]) if self.actual_arrival[i] >= 0 else -1
                delay = actual - int(self.planned_arrival[i]) if actual >= 0 else -1
                writer.writerow([i, int(self.start_times[i]), int(self.planned_arrival[i]), actual, delay, bool(self.arrived[i]), bool(self.disabled[i])])
        return filename


    def seed(self, seed=None):
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        return [seed]


if __name__ == "__main__":
    # quick smoke test
    env = RailEnv()
    obs, _ = env.reset(seed=42)
    print("Initial obs shape:", obs.shape)
    for _ in range(30):
        acts = env.action_space.sample()
        obs, rew, done, trunc, info = env.step(acts)
        print("Reward:", rew, "Info:", info)
        env.render()
        if done or trunc:
            break
    env.save_episode_log("runs/test_episode.csv")
    print("Saved log to runs/test_episode.csv")
