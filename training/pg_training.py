"""
Policy Gradient Training for FarmManagementEnv.

Implements PPO, A2C, and REINFORCE. Saves models cleanly to zip/pt files,
and generates the required convergence and comparison charts.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environment.custom_env import FarmManagementEnv

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

# Ensure clean directory structure: models/[algo]/
for algo in ["ppo", "a2c", "reinforce"]:
    os.makedirs(os.path.join(MODELS_DIR, algo), exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 30_000
DEVICE = torch.device("cpu")

# REINFORCE Implementation

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )
        
    def act(self, obs: np.ndarray):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(self.net(obs_t), dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

class ReinforceAgent:
    def __init__(self, obs_dim: int, act_dim: int, lr: float, gamma: float, hidden: int):
        self.gamma = gamma
        self.policy = PolicyNetwork(obs_dim, act_dim, hidden).to(DEVICE)
        self.optim = optim.Adam(self.policy.parameters(), lr=lr)

    def update(self, log_probs: list, rewards: list):
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -torch.stack(log_probs) * returns
        self.optim.zero_grad()
        loss.sum().backward()
        self.optim.step()
        
    def save(self, path: str):
        # Save PyTorch weights with .pt extension implicitly handled by the user
        torch.save(self.policy.state_dict(), path + ".pt")

def train_reinforce(run_id: int, params: dict, env_cls):
    env = env_cls()
    eval_env = env_cls()
    agent = ReinforceAgent(env.observation_space.shape[0], env.action_space.n, 
                           lr=params["lr"], gamma=params["gamma"], hidden=params["hidden"])
    
    episode_rewards = []
    for ep in range(1250): # ~30,000 steps
        obs, _ = env.reset()
        log_probs, rewards = [], []
        done = False
        while not done:
            action, log_p = agent.policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_p)
            rewards.append(reward)
            done = terminated or truncated
        agent.update(log_probs, rewards)
        episode_rewards.append(sum(rewards))

    # Evaluate
    eval_rewards = []
    for _ in range(10):
        obs, _ = eval_env.reset()
        ep_r, done = 0.0, False
        while not done:
            with torch.no_grad():
                act, _ = agent.policy.act(obs)
            obs, r, term, trunc, _ = eval_env.step(act)
            ep_r += r
            done = term or trunc
        eval_rewards.append(ep_r)
        
    mean_r, std_r = float(np.mean(eval_rewards)), float(np.std(eval_rewards))
    
    # Save as models/reinforce/{run_id}.pt
    agent.save(os.path.join(MODELS_DIR, "reinforce", str(run_id)))
    print(f"  [REINFORCE Run {run_id:2d}] Mean R = {mean_r:.2f}")
    return mean_r, std_r, episode_rewards

# --- Hyperparameter grids ---

PPO_PARAMS = [
    dict(learning_rate=3e-4, gamma=0.99, n_steps=512, ent_coef=0.01),
    dict(learning_rate=1e-4, gamma=0.99, n_steps=512, ent_coef=0.01),
    dict(learning_rate=3e-4, gamma=0.95, n_steps=512, ent_coef=0.01),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=1024, ent_coef=0.01),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=512, ent_coef=0.05),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=512, ent_coef=0.00),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=256, ent_coef=0.01),
    dict(learning_rate=5e-4, gamma=0.99, n_steps=2048, ent_coef=0.02),
    dict(learning_rate=1e-3, gamma=0.95, n_steps=512, ent_coef=0.01),
    dict(learning_rate=2e-4, gamma=0.99, n_steps=1024, ent_coef=0.05)
]

A2C_PARAMS = [
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5, ent_coef=0.01),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=5, ent_coef=0.01),
    dict(learning_rate=7e-4, gamma=0.95, n_steps=5, ent_coef=0.01),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=10, ent_coef=0.01),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=20, ent_coef=0.01),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5, ent_coef=0.05),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5, ent_coef=0.00),
    dict(learning_rate=1e-3, gamma=0.99, n_steps=5, ent_coef=0.01),
    dict(learning_rate=5e-4, gamma=0.99, n_steps=15, ent_coef=0.02),
    dict(learning_rate=7e-4, gamma=0.90, n_steps=5, ent_coef=0.01)
]

REINFORCE_PARAMS = [
    dict(lr=1e-3, gamma=0.99, hidden=128),
    dict(lr=5e-4, gamma=0.99, hidden=128),
    dict(lr=2e-3, gamma=0.99, hidden=128),
    dict(lr=1e-3, gamma=0.95, hidden=128),
    dict(lr=1e-3, gamma=0.90, hidden=128),
    dict(lr=1e-3, gamma=0.99, hidden=64),
    dict(lr=1e-3, gamma=0.99, hidden=256),
    dict(lr=5e-4, gamma=0.95, hidden=64),
    dict(lr=2e-3, gamma=0.95, hidden=256),
    dict(lr=5e-3, gamma=0.99, hidden=128)
]

def train_sb3(algo_cls, run_id: int, params: dict, algo_name: str):
    env = Monitor(FarmManagementEnv())
    tb_path = os.path.join(OUTPUTS_DIR, "tensorboard", algo_name)
    model = algo_cls("MlpPolicy", env, verbose=0, tensorboard_log=tb_path, **params)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False, tb_log_name=f"run_{run_id:02d}")
    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=10, warn=False)
    
    # Save as models/[algo]/{run_id}.zip
    model.save(os.path.join(MODELS_DIR, algo_name, str(run_id)))
    print(f"  [{algo_name.upper()} Run {run_id:2d}] Mean R = {mean_r:.2f}")
    return mean_r, std_r

# --- Plotting functions ---

def plot_pg_comparisons(dfs: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"PPO": "royalblue", "A2C": "darkorange", "REINFORCE": "seagreen"}
    
    for name, df in dfs.items():
        ax.plot(df["run"], df["mean_reward"], marker="o", label=name, color=colors[name])
        ax.fill_between(df["run"], df["mean_reward"] - df["std_reward"], df["mean_reward"] + df["std_reward"], alpha=0.15, color=colors[name])
        
    ax.set_title("Policy Gradient Algorithms: Mean Reward Across Runs")
    ax.set_xlabel("Hyperparameter Run ID")
    ax.set_ylabel("Mean Reward")
    ax.legend()
    
    out_path = os.path.join(OUTPUTS_DIR, "pg_algorithms_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n  -> Saved PG comparison chart: {out_path}")
    plt.close()

def plot_entropy_convergence(rf_curves: dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    for run_id, rewards in rf_curves.items():
        # Rolling average to smooth the convergence line
        smoothed = pd.Series(rewards).rolling(50, min_periods=1).mean()
        ax.plot(smoothed, alpha=0.6, label=f"Run {run_id}")
        
    ax.set_title("REINFORCE Convergence: Smoothed Episode Rewards over Time")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed Reward")
    ax.legend(fontsize=8, ncol=2)
    
    out_path = os.path.join(OUTPUTS_DIR, "reinforce_convergence.png")
    plt.savefig(out_path, dpi=150)
    print(f"  -> Saved Convergence chart: {out_path}")
    plt.close()

def run_all():
    all_dfs = {}
    
    print("--- Starting PPO Training ---")
    ppo_records = []
    for i, params in enumerate(PPO_PARAMS, 1):
        mean_r, std_r = train_sb3(PPO, i, params, "ppo")
        ppo_records.append({"run": i, "mean_reward": mean_r, "std_reward": std_r, **params})
    df_ppo = pd.DataFrame(ppo_records)
    df_ppo.to_csv(os.path.join(OUTPUTS_DIR, "ppo_results.csv"), index=False)
    all_dfs["PPO"] = df_ppo
    
    print("\n--- Starting A2C Training ---")
    a2c_records = []
    for i, params in enumerate(A2C_PARAMS, 1):
        mean_r, std_r = train_sb3(A2C, i, params, "a2c")
        a2c_records.append({"run": i, "mean_reward": mean_r, "std_reward": std_r, **params})
    df_a2c = pd.DataFrame(a2c_records)
    df_a2c.to_csv(os.path.join(OUTPUTS_DIR, "a2c_results.csv"), index=False)
    all_dfs["A2C"] = df_a2c

    print("\n--- Starting REINFORCE Training ---")
    rf_records = []
    rf_curves = {}
    for i, params in enumerate(REINFORCE_PARAMS, 1):
        mean_r, std_r, ep_rewards = train_reinforce(i, params, FarmManagementEnv)
        rf_records.append({"run": i, "mean_reward": mean_r, "std_reward": std_r, **params})
        rf_curves[i] = ep_rewards
    df_rf = pd.DataFrame(rf_records)
    df_rf.to_csv(os.path.join(OUTPUTS_DIR, "reinforce_results.csv"), index=False)
    all_dfs["REINFORCE"] = df_rf
    
    # Generate charts
    plot_pg_comparisons(all_dfs)
    plot_entropy_convergence(rf_curves)

if __name__ == "__main__":
    run_all()