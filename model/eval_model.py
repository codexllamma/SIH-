# (copy/paste into rl/eval_model.py)
import argparse
import csv
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from rail_env6 import RailEnv

def evaluate(model_path, n_episodes=20, out_csv="eval_report.csv"):
    env = Monitor(RailEnv(), "eval_runs")
    model = PPO.load(model_path)
    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, trunc, info = env.step(action)
            if done or trunc:
                kpis = {k: info.get(k, None) for k in info.keys() if k.startswith("kpi_")}
                # safety: fill defaults
                row = {
                    "episode": ep,
                    "kpi_zero_collision": kpis.get("kpi_zero_collision", 0),
                    "kpi_avg_delay": kpis.get("kpi_avg_delay", None),
                    "kpi_arrivals_on_time": kpis.get("kpi_arrivals_on_time", None),
                    "kpi_completion_rate": kpis.get("kpi_completion_rate", None),
                    "kpi_delay_std": kpis.get("kpi_delay_std", None),
                }
                rows.append(row)
                break

    # write CSV
    keys = ["episode", "kpi_zero_collision", "kpi_avg_delay", "kpi_arrivals_on_time", "kpi_completion_rate", "kpi_delay_std"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary
    arr = np.array([r["kpi_zero_collision"] for r in rows], dtype=float)
    zero_rate = float(np.sum(arr)) / len(arr)
    avg_delay_vals = [r["kpi_avg_delay"] for r in rows if r["kpi_avg_delay"] is not None]
    avg_delay = float(np.mean(avg_delay_vals)) if avg_delay_vals else None

    print(f"Evaluated {n_episodes} episodes -> zero_collision_rate: {zero_rate:.3f}, avg_delay: {avg_delay}")
    print(f"Wrote CSV: {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to PPO model zip")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", default="eval_report.csv")
    args = p.parse_args()
    evaluate(args.model, n_episodes=args.episodes, out_csv=args.out)
