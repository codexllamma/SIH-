"""
RailEnv - Gymnasium environment for 2-track / 4-train toy railway optimization.

Dense, shaped reward overhaul with diagnostics and penalty annealing.

Changes vs original:
- Introduced `_compute_reward()` helper that aggregates a rich set of components:
  progress, lateness, idle, safe spacing, safe switching, proximity/overspeed/switch-speeding/collision
  events, arrival bonuses and arrival delay.
- Scaled, *dense* rewards (small per-step shaping) + milder penalties to avoid freeze policies.
- Optional penalty-annealing schedule to start gentle and ramp difficulty.
- Per-step `info["reward_breakdown"]` for debugging + `episode_logs` entries.
- Reward clipping configurable.

This file is a drop-in replacement for the original.
"""

from typing import Tuple, Dict, Any, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import csv
import os
import math
import time
import copy

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
    # base penalties (kept for semantics; weights in reward block control impact)
    "collision_penalty": -50.0,
    "collision_block_recovery": 50,  # timesteps that track remains blocked after collision
    "switch_speeding_penalty": -10.0,
    "overspeed_penalty": -10.0,
    "proximity_penalty": -20.0,     # per event if within braking distance
    # OBS controls
    "obs_clip_extra": 1.0,
    # logging
    "log_dir": "runs",
    # >>> New reward controls (dense shaping)
    "reward": {
        # Core shaping
        "w_progress": 0.5,              # per unit moved
        "w_lateness": 0.05,             # per step per unit lateness (sum over active trains)
        "w_idle": 0.02,                 # per train per step idle (started, not in station, speed==0)
        "w_safe_spacing_bonus": 0.10,   # per train that exceeds braking distance + margin
        "safe_spacing_margin": 5.0,
        "w_safe_switch_bonus": 0.10,    # per safe switch (in zone, low speed)
        # Event penalties (count-based; severity annealed)
        "w_proximity": 0.25,            # per proximity event (within braking distance)
        "w_overspeed": 0.50,            # per overspeed event
        "w_switch_speeding": 0.50,      # per switch-speeding event
        "w_collision": 0.50,            # scales |collision_penalty| per collision
        # Arrival shaping
        "arrival_on_time": 30.0,
        "arrival_grace": 20.0,
        "arrival_late": 10.0,
        "w_arrival_delay": 0.5,         # subtract per unit delay at arrival (per train)
        # Reward clipping per step (min, max). Set to None to disable.
        "clip": (-25.0, 25.0),
        # Linear annealing for penalties (helps exploration early)
        # factor = lerp(start, end, global_steps / steps)
        "anneal_penalties": {"start": 0.6, "end": 1.0, "steps": 50000},
    },
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

        # actions: per-train: 0=no-op,1=accel,2=decel,3=stop,4=switch_left,5=switch_right
        self.n_actions_per_train = 6
        self.action_space = spaces.MultiDiscrete([self.n_actions_per_train] * self.n_trains)

        # observation features per train:
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

    # ------------------------------
    # Reset
    # ------------------------------
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

            # planned arrival estimate: simple linear estimate using max_speed
            distance = max(0.0, self.track_length - self.positions[i])
            est_travel_steps = math.ceil(distance / max(1, (self.max_speed / 1.0)))
            est_halts = 1 * self.cfg["station_halt_time"]  # single intermediate station
            self.planned_arrival[i] = self.start_times[i] + est_travel_steps + est_halts

        obs = self._get_obs()
        return obs, {}

    # ------------------------------
    # Observations
    # ------------------------------
    def _get_obs(self) -> np.ndarray:
        feats = np.zeros((self.n_trains, self.features_per_train), dtype=np.float32)
        for i in range(self.n_trains):
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

        return feats.flatten().astype(np.float32)

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

    # ------------------------------
    # Step
    # ------------------------------
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
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

        prev_speeds = self.speeds.copy()

        for i in range(self.n_trains):
            if (not self.started[i]) or self.arrived[i] or self.disabled[i]:
                continue

            a = int(action[i])
            if a == 1:      # accel
                self.speeds[i] = min(self.max_speed, int(self.speeds[i] + c["accel"]))
            elif a == 2:    # decel
                self.speeds[i] = max(0, int(self.speeds[i] - c["decel"]))
            elif a == 3:    # stop
                self.speeds[i] = 0
                if self._is_in_station(i):
                    self.halt_remaining[i] = max(self.halt_remaining[i], c["station_halt_time"])
            elif a == 4:    # switch left
                if self._is_in_switch_zone(i):
                    if self.speeds[i] <= 1:
                        old_track = int(self.tracks[i])
                        new_track = max(0, old_track - 1)
                        if self.track_blocked_timer[new_track] == 0:
                            self.tracks[i] = new_track
                            switched[i] = True
                            safe_switch[i] = True
                        else:
                            illegally_switched[i] = True
                    else:
                        switch_speeding_events += 1
                        illegally_switched[i] = True
                else:
                    illegally_switched[i] = True
            elif a == 5:    # switch right
                if self._is_in_switch_zone(i):
                    if self.speeds[i] <= 1:
                        old_track = int(self.tracks[i])
                        new_track = min(self.n_tracks - 1, old_track + 1)
                        if self.track_blocked_timer[new_track] == 0:
                            self.tracks[i] = new_track
                            switched[i] = True
                            safe_switch[i] = True
                        else:
                            illegally_switched[i] = True
                    else:
                        switch_speeding_events += 1
                        illegally_switched[i] = True
                else:
                    illegally_switched[i] = True
            # else: noop or unknown -> no-op

            # track speed limit enforcement (event count)
            track_limit = c["track_speed_limits"][int(self.tracks[i])]
            if self.speeds[i] > track_limit:
                overspeed_events += 1

        # Second pass: position update (movement)
        prev_positions = self.positions.copy()
        for i in range(self.n_trains):
            if (not self.started[i]) or self.arrived[i] or self.disabled[i]:
                continue
            self.positions[i] = min(float(self.track_length), self.positions[i] + float(self.speeds[i]))
            if self._is_at_station_entry(i) and self.halt_remaining[i] == 0 and self._should_halt_now(i):
                self.halt_remaining[i] = c["station_halt_time"]
                self.speeds[i] = 0

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
        spacing_margin = float(self.cfg["reward"]["safe_spacing_margin"]) if "reward" in self.cfg else 5.0
        for i in range(self.n_trains):
            if self.started[i] and (not self.arrived[i]) and (not self.disabled[i]):
                dist_next = self._distance_to_next_train(i)
                braking_dist = int(c["braking_distance_map"].get(int(self.speeds[i]), 0))
                if dist_next > (braking_dist + spacing_margin):
                    safe_spacing_count += 1

        safe_switch_count = int(np.sum(safe_switch))

        # Build reward components and compute reward
        comps = {
            "delta_pos": float(delta_pos),
            "lateness_total": float(lateness_total),
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

        # Package info
        info["step"] = int(self.current_step)
        info["collision_count"] = int(self._episode_collision_count)
        info["active_trains"] = int(np.sum((~self.arrived) & (~self.disabled)))
        info["delta_pos"] = float(delta_pos)
        info["reward_breakdown"] = {
            "raw": comps,
            "weighted": weighted,
            "total": float(reward),
            "anneal_factor": float(self._penalty_anneal_factor()),
        }

        # Per-step log (lightweight)
        self.episode_logs.append({
            "step": int(self.current_step),
            "reward": float(reward),
            **{f"raw_{k}": v for k, v in comps.items()},
        })

        obs = self._get_obs()
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------
    # Reward helper
    # ------------------------------
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
        """Compute shaped reward from components. Returns (reward, weighted_components).
        `comps` contains counts and magnitudes collected in step().
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
        # Scale by absolute collision penalty magnitude so tuning is intuitive
        collision_mag = abs(self.cfg.get("collision_penalty", -50.0))
        weighted["collision"] = - anneal * w_collision * comps["collision_events"] * collision_mag / 50.0
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

    # ------------------------------
    # Utility predicates
    # ------------------------------
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

    # ------------------------------
    # Render & Logging
    # ------------------------------
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

        print(f"\nStep {self.current_step} | AccReward={self._episode_acc_reward:.1f} | Collisions={self._episode_collision_count}")
        for idx, ln in enumerate(out_lines):
            print(f"Track {idx}: {ln[:200]}")

    def save_episode_log(self, filename: str = None):
        if filename is None:
            filename = os.path.join(self.cfg["log_dir"], f"episode_{int(time.time())}.csv")
        header = [
            "train_idx", "start_time", "planned_arrival", "actual_arrival", "delay", "arrived", "disabled"
        ]
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
    # quick smoke test (random policy)
    env = RailEnv()
    obs, _ = env.reset(seed=42)
    print("Initial obs shape:", obs.shape)
    total = 0.0
    for _ in range(40):
        acts = env.action_space.sample()
        obs, rew, done, trunc, info = env.step(acts)
        total += rew
        print("Reward:", round(rew, 2), "Info: total=", round(info.get("reward_breakdown", {}).get("total", rew), 2),
              "active=", info.get("active_trains"),
              "anneal=", round(info.get("reward_breakdown", {}).get("anneal_factor", 1.0), 2))
        if done or trunc:
            break
    env.save_episode_log("runs/test_episode.csv")
    print("Saved log to runs/test_episode.csv | Return:", round(total, 2))
