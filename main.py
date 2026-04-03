"""
Run the best-performing RL agent in the Farm Management Environment.
Loads the specified model from models/[algo]/{run}.[zip/pt]

Usage:
    python main.py                          # auto-select best available model globally
    python main.py --algo ppo --run 3       # specific model
    python main.py --algo a2c --run 1
    python main.py --algo dqn --run 1
    python main.py --algo reinforce --run 2
    python main.py --episodes 5             # number of episodes to simulate
    python main.py --no-render              # headless (terminal only)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import warnings
from stable_baselines3 import DQN, PPO, A2C

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

sys.path.insert(0, BASE_DIR)
from environment.custom_env import FarmManagementEnv

# Import the custom REINFORCE agent class
try:
    from training.pg_training import ReinforceAgent
except ImportError:
    ReinforceAgent = None

# --- Model Loading Helpers ---

def load_sb3_model(algo: str, run_id: int):
    cls_map = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
    cls = cls_map[algo.lower()]
    
    # Path format: models/[algo]/[run_id].zip
    path = os.path.join(MODELS_DIR, algo.lower(), f"{run_id}.zip")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}\nTrain first using the training scripts.")
    
    print(f"  Loading {algo.upper()} Run #{run_id} from {path}")
    return cls.load(path)

def load_reinforce_model(run_id: int):
    if ReinforceAgent is None:
        raise ImportError("Could not import ReinforceAgent from training.pg_training.")
        
    # Path format: models/reinforce/[run_id].pt
    path = os.path.join(MODELS_DIR, "reinforce", f"{run_id}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"REINFORCE model not found: {path}")

    print(f"  Loading REINFORCE Run #{run_id} from {path}")

    env = FarmManagementEnv()
    agent = ReinforceAgent(env.observation_space.shape[0], env.action_space.n, lr=1e-3, gamma=0.99, hidden=128)
    env.close()

    agent.policy.load_state_dict(torch.load(path, map_location="cpu"))
    agent.policy.eval()
    return agent

def find_best_model():
    best_algo, best_run, best_r = "ppo", 1, -np.inf
    
    csv_map = {
        "dqn": os.path.join(OUTPUTS_DIR, "dqn_results.csv"),
        "ppo": os.path.join(OUTPUTS_DIR, "ppo_results.csv"),
        "a2c": os.path.join(OUTPUTS_DIR, "a2c_results.csv"),
        "reinforce": os.path.join(OUTPUTS_DIR, "reinforce_results.csv"),
    }
    
    found_any = False
    for algo, csv_path in csv_map.items():
        if not os.path.exists(csv_path):
            continue
            
        found_any = True
        df = pd.read_csv(csv_path)
        idx = df["mean_reward"].idxmax()
        r = df.loc[idx, "mean_reward"]
        rid = int(df.loc[idx, "run"])
        
        if r > best_r:
            best_r, best_algo, best_run = r, algo, rid
            
    if not found_any:
        print("  Warning: No CSV logs found. Defaulting to PPO Run #1.")
        return "ppo", 1
        
    print(f"  Auto-selected globally best model: {best_algo.upper()} Run #{best_run} (Mean Reward: {best_r:.2f})")
    return best_algo, best_run

# --- Simulation Loop ---

ACTION_LABELS = ["Idle", "Refill Hen Feed", "Refill Pig Feed", "Clean Coop", "Clean Pens", "Inspect Health"]

def run_episode(model, algo: str, env: FarmManagementEnv, episode_num: int):
    obs, info = env.reset()
    done = False
    step = 0
    total_r = 0.0

    print(f"\n{'─'*75}")
    print(f"  EPISODE {episode_num} — 24-Hour Farm Cycle")
    print(f"{'─'*75}")
    print(f"  {'Hour':>4}  {'Action Taken':<18}  {'HenF':>5}  {'PigF':>5}  {'HenH':>5}  {'PigH':>5}  {'Labr':>4}  {'Reward':>7}  {'CumR':>8}")
    print(f"  {'':─>4}  {'':─>18}  {'':─>5}  {'':─>5}  {'':─>5}  {'':─>5}  {'':─>4}  {'':─>7}  {'':─>8}")

    while not done:
        if algo == "reinforce":
            with torch.no_grad():
                action, _ = model.policy.act(obs)
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_r += reward
        step += 1

        if step % 3 == 0 or done:
            print(f"  {info['hour']:>4}  {ACTION_LABELS[action]:<18}  "
                  f"{info['hen_feed']:>5.2f}  {info['pig_feed']:>5.2f}  "
                  f"{info['hen_health']:>5.2f}  {info['pig_health']:>5.2f}  "
                  f"{info['labor']:>4.1f}  {reward:>+7.2f}  {total_r:>8.2f}")

    status = "[OK] Livestock Thriving" if info["alive"] else "[FAIL] Critical Condition Reached"
    print(f"\n  {status}  |  Hours Managed: {step}  |  Final Reward: {total_r:.2f}")
    return total_r, info["alive"]

def print_banner(algo: str, run_id: int):
    print("\n" + "-" * 75)
    print("    AI-INTEGRATED FARM MANAGEMENT — RL SIMULATION")
    print("-" * 75)
    print(f"  Algorithm : {algo.upper()} | Run ID : #{run_id}")
    print("-" * 75)

# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Farm Management RL Simulation")
    parser.add_argument("--algo", default="auto", help="Algorithm: dqn|ppo|a2c|reinforce|auto")
    parser.add_argument("--run", type=int, default=1, help="Run ID to load")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable pygame rendering")
    args = parser.parse_args()

    if args.algo.lower() == "auto":
        algo, run_id = find_best_model()
    else:
        algo, run_id = args.algo.lower(), args.run

    print_banner(algo, run_id)

    try:
        if algo == "reinforce":
            model = load_reinforce_model(run_id)
        else:
            model = load_sb3_model(algo, run_id)
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        return

    render_mode = None if args.no_render else "human"
    try:
        env = FarmManagementEnv(render_mode=render_mode)
    except Exception as e:
        print(f"\n   [GUI Warning] Rendering unavailable ({e}). Falling back to headless mode.\n")
        env = FarmManagementEnv(render_mode=None)

    rewards, outcomes = [], []
    for ep in range(1, args.episodes + 1):
        ep_r, alive = run_episode(model, algo, env, ep)
        rewards.append(ep_r)
        outcomes.append(alive)
        time.sleep(1.0) 

    env.close()

    print("\n" + "-" * 75)
    print("  SIMULATION SUMMARY")
    print("-" * 75)
    print(f"  Algorithm           : {algo.upper()} Run #{run_id}")
    print(f"  Mean Reward         : {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  Successful Days     : {sum(outcomes)}/{args.episodes} ({100*sum(outcomes)//args.episodes}%)")
    print("-" * 75)

if __name__ == "__main__":
    main()