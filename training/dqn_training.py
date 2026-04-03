"""
Deep Q-Network Training for FarmManagementEnv.

Runs 10 distinct hyperparameter combinations, saves the models as zip files,
and generates performance charts for the final report.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environment.custom_env import FarmManagementEnv

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(os.path.join(MODELS_DIR, "dqn"), exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 30_000 
EVAL_EPISODES = 20

HYPERPARAMS = [
    dict(learning_rate=1e-3, gamma=0.99, batch_size=64, buffer_size=10000, exploration_fraction=0.20),
    dict(learning_rate=5e-4, gamma=0.99, batch_size=64, buffer_size=10000, exploration_fraction=0.30),
    dict(learning_rate=1e-3, gamma=0.95, batch_size=64, buffer_size=10000, exploration_fraction=0.40),
    dict(learning_rate=1e-3, gamma=0.99, batch_size=128, buffer_size=20000, exploration_fraction=0.20),
    dict(learning_rate=2e-3, gamma=0.99, batch_size=32, buffer_size=5000, exploration_fraction=0.25),
    dict(learning_rate=1e-3, gamma=0.90, batch_size=64, buffer_size=10000, exploration_fraction=0.20),
    dict(learning_rate=5e-4, gamma=0.99, batch_size=128, buffer_size=50000, exploration_fraction=0.20),
    dict(learning_rate=5e-3, gamma=0.95, batch_size=64, buffer_size=10000, exploration_fraction=0.15),
    dict(learning_rate=1e-3, gamma=0.99, batch_size=64, buffer_size=10000, exploration_fraction=0.10),
    dict(learning_rate=2e-4, gamma=0.99, batch_size=256, buffer_size=100000, exploration_fraction=0.25),
]

def make_env():
    return Monitor(FarmManagementEnv())

def train_dqn(run_id: int, params: dict):
    env = make_env()
    eval_env = make_env()
    tb_path = os.path.join(OUTPUTS_DIR, "tensorboard", "dqn")
    model = DQN(
        policy="MlpPolicy", 
        env=env, 
        verbose=0, 
        tensorboard_log=tb_path,
        **params
    )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False, tb_log_name=f"run_{run_id:02d}")
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES, warn=False)
    
    # Save model as models/dqn/{run_id}.zip
    save_path = os.path.join(MODELS_DIR, "dqn", str(run_id))
    model.save(save_path)
    print(f"  [Run {run_id:2d}] Mean R = {mean_r:.2f} +/- {std_r:.2f}")
    
    env.close()
    eval_env.close()
    return mean_r, std_r

def plot_dqn_results(df: pd.DataFrame):
    """Generates and saves the DQN hyperparameter performance charts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("DQN Hyperparameter Tuning - FarmManagementEnv", fontsize=14, fontweight="bold")

    # Bar chart of rewards per run
    ax = axes[0]
    ax.bar(df["run"], df["mean_reward"], color="steelblue", edgecolor="k", yerr=df["std_reward"], capsize=4)
    ax.set_title("Mean Reward per Run")
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Mean Reward (20 Eval Episodes)")
    
    # Scatter plot: Learning Rate vs Reward
    ax = axes[1]
    for lr, grp in df.groupby("learning_rate"):
        ax.scatter(grp["run"], grp["mean_reward"], label=f"LR={lr:.0e}", s=80)
    ax.set_title("Reward by Learning Rate")
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Mean Reward")
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "dqn_training_stats.png")
    plt.savefig(out_path, dpi=150)
    print(f"  -> Saved DQN chart: {out_path}")
    plt.close()

def run_all():
    records = []
    print("Starting DQN Hyperparameter Grid Search...")
    for i, params in enumerate(HYPERPARAMS, start=1):
        mean_r, std_r = train_dqn(i, params)
        records.append({"run": i, "mean_reward": mean_r, "std_reward": std_r, **params})
    
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUTS_DIR, "dqn_results.csv"), index=False)
    plot_dqn_results(df)
    print("\nDQN Training Complete.")

if __name__ == "__main__":
    run_all()