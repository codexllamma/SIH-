"""
rail_env2.py

Refactored RailEnv class (drop-in replacement for your original rail_env2 import).
- Class name: RailEnv (keeps import compatibility)
- No training code or side-effects; safe to import from training scripts
- Maintains the same public API: reset(), step(), render(), save_episode_log(), seed()

I kept the public behavior consistent while improving reward shaping, observations (braking margin + distance to next signal), and collision/proximity handling.

Place this file in your project and the existing PPO training script that does `from rail_env2 import RailEnv` will import this class unchanged.
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
    "track_length": 140,
    "train_length": 10,
    "station_halt_time": 5,
    "max_speed": 3,
    "accel": 1,
    "decel": 1,
    "max_steps": 500,
    # layout
    "spawn_points": [0, 10],
    "stationA_range": (10, 20),
    "switch1_range": (85, 88),
    "switch2_range": (97, 100),
    "stationB_range": (120, 130),
    "switch_decision_point": 80,
    "braking_distance_map": {3: 40, 2: 20, 1: 10, 0: 0},
    "track_speed_limits": [3, 3],
    "collision_penalty": -50.0,
    "collision_block_recovery": 50,
    "switch_speeding_penalty": -10.0,
    "overspeed_penalty": -10.0,
    "proximity_penalty": -20.0,
    "obs_clip_extra": 1.0,
    "log_dir": "runs",
    "reward": {
        # shaping
        "w_progress": 1.0,
        "w_lateness": 0.02,
        "w_idle": 0.01,
        "w_safe_spacing": 0.15,
        "safe_spacing_margin": 5.0,
        "w_proximity": 0.40,
        "w_overspeed": 0.40,
        "w_switch_speeding": 0.40,
        "w_collision": 1.0,
        "w_arrival_delay": 0.25,
        "arrival_on_time": 30.0,
        "arrival_grace": 20.0,
        "arrival_late": 10.0,
        "clip": (-25.0, 25.0),
        "anneal_penalties": {"start": 0.6, "end": 1.0, "steps": 50000},
        "initial_penalty_scale": 0.6,
        "penalty_ramp_steps": 50000,
    },
}


def normalize(val: float, max_val: float) -> float:
    if max_val == 0:
        return 0.0
    return float(val) / float(max_val)


class RailEnv(gym.Env):
    """Drop-in RailEnv replacement. Keep API stable for your PPO script."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: Dict = None, render_mode: str = None):
        super().__init__()
        base = copy.deepcopy(DEFAULT_CONFIG)
        if config:
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

        # observation features per train: extended but normalized
        self.features_per_train = 11
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

        # signals state mapping
        self.signal_positions = [0, self.stationA_range[1], self.switch1_range[0], self.switch2_range[0], self.stationB_range[0]]
        self.signal_states = {p: True for p in self.signal_positions}

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
        self._seed = None
        self.render_mode = render_mode
        os.makedirs(self.cfg["log_dir"], exist_ok=True)

        # internal diagnostics
        self._last_collision_pairs = set()

    # ------------------------------
    # Reset
    # ------------------------------
    def reset(self, *, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)

        self._episode_idx += 1

        # reset arrays
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
        self._last_collision_pairs.clear()

        spawn_pos_choices = self.cfg.get('spawn_points', [0, 10])
        for i in range(self.n_trains):
            self.tracks[i] = int(self.rng.integers(0, self.n_tracks))
            base_pos = float(self.rng.choice(spawn_pos_choices))
            jitter = float(self.rng.uniform(0.0, 1.0))
            self.positions[i] = min(self.track_length, base_pos + jitter)
            self.speeds[i] = 0
            self.halt_remaining[i] = 0
            self.started[i] = True if self.positions[i] != 0 else False
            if self.positions[i] == 0:
                self.start_times[i] = int(self.rng.integers(0, 6))
                self.started[i] = False
            else:
                self.start_times[i] = 0
                self.started[i] = True

            distance = max(0.0, self.track_length - self.positions[i])
            est_travel_steps = math.ceil(distance / max(1, (self.max_speed / 1.0)))
            est_halts = 1 * self.cfg["station_halt_time"]
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
            feats[i, 5] = 1.0 if self._next_signal_state_for_train(i) else 0.0
            feats[i, 6] = 1.0 if self._is_in_station(i) else 0.0
            feats[i, 7] = normalize(self.halt_remaining[i], self.cfg["station_halt_time"])
            braking_dist = float(self.cfg["braking_distance_map"].get(int(self.speeds[i]), 0))
            margin = float(self.cfg["reward"].get("safe_spacing_margin", 5.0))
            feats[i, 8] = normalize(braking_dist + margin, float(self.track_length))
            feats[i, 9] = normalize(self._distance_to_next_signal(i), self.track_length)
            if not self.started[i]:
                time_until = max(0, self.start_times[i] - self.current_step)
                feats[i, 10] = normalize(time_until, self.max_steps)
            else:
                feats[i, 10] = 0.0
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

    def _distance_to_next_signal(self, idx: int) -> float:
        pos = self.positions[idx]
        next_signals = [p for p in self.signal_positions if p > pos]
        if not next_signals:
            return float(self.track_length)
        return float(min(next_signals) - pos)

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

        # start trains
        for i in range(self.n_trains):
            if (not self.started[i]) and (self.current_step >= self.start_times[i]):
                self.started[i] = True
                self.speeds[i] = 0

        switched = np.zeros(self.n_trains, dtype=bool)
        safe_switch = np.zeros(self.n_trains, dtype=bool)
        illegally_switched = np.zeros(self.n_trains, dtype=bool)

        overspeed_events = 0
        switch_speeding_events = 0

        prev_positions = self.positions.copy()

        for i in range(self.n_trains):
            if (not self.started[i]) or self.arrived[i] or self.disabled[i]:
                continue
            a = int(action[i])
            if a == 1:
                self.speeds[i] = min(self.max_speed, int(self.speeds[i] + c["accel"]))
            elif a == 2:
                self.speeds[i] = max(0, int(self.speeds[i] - c["decel"]))
            elif a == 3:
                self.speeds[i] = 0
                if self._is_in_station(i):
                    self.halt_remaining[i] = max(self.halt_remaining[i], c["station_halt_time"])
            elif a in (4, 5):
                if self._is_in_switch_zone(i):
                    if self.speeds[i] <= 1:
                        old_track = int(self.tracks[i])
                        new_track = max(0, old_track - 1) if a == 4 else min(self.n_tracks - 1, old_track + 1)
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

            # overspeed event
            track_limit = c["track_speed_limits"][int(self.tracks[i])]
            if self.speeds[i] > track_limit:
                overspeed_events += 1

        # Update positions
        for i in range(self.n_trains):
            if (not self.started[i]) or self.arrived[i] or self.disabled[i]:
                continue
            self.positions[i] = min(float(self.track_length), self.positions[i] + float(self.speeds[i]))
            if self._is_at_station_entry(i) and self.halt_remaining[i] == 0 and self._should_halt_now(i):
                self.halt_remaining[i] = c["station_halt_time"]
                self.speeds[i] = 0

        # Decrement halts & arrivals
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

        # Collision & proximity
        proximity_events = 0
        collision_events = 0
        marked_pairs = set()
        for t in range(self.n_trains):
            if (not self.started[t]) or self.arrived[t] or self.disabled[t]:
                continue
            for u in range(t + 1, self.n_trains):
                if (not self.started[u]) or self.arrived[u] or self.disabled[u]:
                    continue
                if int(self.tracks[t]) != int(self.tracks[u]):
                    continue
                distance = abs(self.positions[t] - self.positions[u])
                pair = (min(t, u), max(t, u))
                # collision
                if distance < float(self.train_length):
                    if pair not in self._last_collision_pairs:
                        collision_events += 1
                        self._episode_collision_count += 1
                        track_idx = int(self.tracks[t])
                        self.track_blocked_timer[track_idx] = c["collision_block_recovery"]
                        self.disabled[t] = True
                        self.disabled[u] = True
                        marked_pairs.add(pair)
                else:
                    # proximity if within braking distance of trailing train
                    if self.positions[t] < self.positions[u]:
                        trailing = t
                        ahead = u
                    else:
                        trailing = u
                        ahead = t
                    trailing_speed = int(self.speeds[trailing])
                    braking_dist = int(c["braking_distance_map"].get(trailing_speed, 0))
                    if distance <= braking_dist:
                        proximity_events += 1

        self._last_collision_pairs = marked_pairs

        delta_pos = np.sum(np.maximum(0.0, self.positions - prev_positions))

        # lateness and idle
        lateness_total = 0
        idle_count = 0
        for i in range(self.n_trains):
            if self.started[i] and (not self.arrived[i]) and (not self.disabled[i]):
                lateness_total += max(0, self.current_step - self.planned_arrival[i])
                if (not self._is_in_station(i)) and self.speeds[i] == 0:
                    idle_count += 1

        safe_spacing_count = 0
        spacing_margin = float(self.cfg["reward"].get("safe_spacing_margin", 5.0))
        for i in range(self.n_trains):
            if self.started[i] and (not self.arrived[i]) and (not self.disabled[i]):
                dist_next = self._distance_to_next_train(i)
                braking_dist = int(c["braking_distance_map"].get(int(self.speeds[i]), 0))
                if dist_next > (braking_dist + spacing_margin):
                    safe_spacing_count += 1

        safe_switch_count = int(np.sum(safe_switch))

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
        rc = self.cfg.get("reward", {})
        w_progress = float(rc.get("w_progress", 1.0))
        w_lateness = float(rc.get("w_lateness", 0.02))
        w_idle = float(rc.get("w_idle", 0.01))
        w_safe_spacing = float(rc.get("w_safe_spacing", 0.15))
        w_proximity = float(rc.get("w_proximity", 0.40))
        w_overspeed = float(rc.get("w_overspeed", 0.40))
        w_switch_spd = float(rc.get("w_switch_speeding", 0.40))
        w_collision = float(rc.get("w_collision", 1.0))
        w_arrival_delay = float(rc.get("w_arrival_delay", 0.25))

        b_on_time = float(rc.get("arrival_on_time", 30.0))
        b_grace = float(rc.get("arrival_grace", 20.0))
        b_late = float(rc.get("arrival_late", 10.0))

        anneal = self._penalty_anneal_factor()

        weighted = {}
        norm_progress = comps["delta_pos"] / max(1.0, float(self.track_length))
        weighted["progress"] = w_progress * norm_progress

        weighted["lateness"] = - anneal * w_lateness * comps["lateness_total"]
        weighted["idle"] = - anneal * w_idle * comps["idle_count"]
        weighted["safe_spacing_bonus"] = w_safe_spacing * comps["safe_spacing_count"]

        weighted["proximity"] = - anneal * w_proximity * comps["proximity_events"]

        weighted["overspeed"] = - anneal * w_overspeed * comps["overspeed_events"]
        weighted["switch_speeding"] = - anneal * w_switch_spd * comps["switch_speeding_events"]

        collision_mag = abs(self.cfg.get("collision_penalty", -50.0))
        weighted["collision"] = - anneal * w_collision * comps["collision_events"] * (collision_mag / 50.0)

        weighted["arrival_bonus"] = (
            b_on_time * comps["arrivals_on_time"]
            + b_grace * comps["arrivals_grace"]
            + b_late * comps["arrivals_late"]
        )
        weighted["arrival_delay"] = - anneal * w_arrival_delay * comps["arrival_delay_sum"]

        total = float(sum(weighted.values()))

        clip_cfg = rc.get("clip", None)
        if clip_cfg is not None:
            lo, hi = float(clip_cfg[0]), float(clip_cfg[1])
            total = float(np.clip(total, lo, hi))

        return total, weighted

    # ------------------------------
    # Utility preds
    # ------------------------------
    def _is_in_station(self, idx: int) -> bool:
        pos = self.positions[idx]
        sa0, sa1 = self.cfg['stationA_range']
        sb0, sb1 = self.cfg['stationB_range']
        return (sa0 <= pos < sa1) or (sb0 <= pos < sb1)

    def _is_at_station_entry(self, idx: int) -> bool:
        return self._is_in_station(idx)

    def _is_in_switch_zone(self, idx: int) -> bool:
        pos = self.positions[idx]
        s1a, s1b = self.cfg['switch1_range']
        s2a, s2b = self.cfg['switch2_range']
        return (s1a <= pos <= s1b) or (s2a <= pos <= s2b)

    def _should_halt_now(self, idx: int) -> bool:
        return self._is_in_station(idx) and not self.arrived[idx] and self.halt_remaining[idx] == 0

    def _can_leave_station(self, idx: int) -> bool:
        pos = self.positions[idx]
        sa0, sa1 = self.cfg['stationA_range']
        sb0, sb1 = self.cfg['stationB_range']
        if sa0 <= pos < sa1:
            sigpos = sa1
        elif sb0 <= pos < sb1:
            sigpos = sb0
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
            sa0, sa1 = self.cfg['stationA_range']
            sb0, sb1 = self.cfg['stationB_range']
            for s in range(sa0, sa1):
                if 0 <= s <= self.track_length:
                    line[int(s)] = "S"
            for s in range(sb0, sb1):
                if 0 <= s <= self.track_length:
                    line[int(s)] = "s"
            for x in self.cfg['switch1_range']:
                if 0 <= x <= self.track_length:
                    line[int(x)] = "J"
            for x in self.cfg['switch2_range']:
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
