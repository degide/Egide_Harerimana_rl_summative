# Egide_Harerimana_rl_summative

This repository contains a custom, Reinforcement Learning environment (`FarmManagementEnv`) built with Gymnasium. It acts as a digital twin for a mixed-livestock agricultural operation managing **980 laying hens** and **10 pigs**. 

The RL agent acts as the centralized management system, learning to balance limited daily worker hours (labor) and feed resources to maintain optimal health scores and maximize overall productivity without depleting resources.

---

## Environment Architecture

The environment avoids simplified grid-world abstractions, instead modeling continuous resource decay, interrelated state variables, and strict labor constraints over a 24-hour daily cycle.

### Observation Space `Box(6,)`
All values are normalized `[0.0, 1.0]` for stable neural network training.
| Index | Feature | Description |
| :--- | :--- | :--- |
| `0` | **Time of Day** | Current hour normalized (step / 24). |
| `1` | **Hen Feed Level** | Remaining feed for the poultry coop. |
| `2` | **Pig Feed Level** | Remaining feed for the pig pens. |
| `3` | **Hen Health** | Simulated CNN-derived health score. |
| `4` | **Pig Health** | Simulated CNN-derived health score. |
| `5` | **Labor Remaining** | Worker hours left in the current day (max 8 hrs). |

### Action Space `Discrete(6)`
| Action | Task | Labor Cost | Effect |
| :---: | :--- | :---: | :--- |
| `0` | **Idle** | 0 hrs | Conserves labor, allows natural decay. |
| `1` | **Refill Hen Feed** | 1 hr | Increases Hen Feed Level. |
| `2` | **Refill Pig Feed** | 1 hr | Increases Pig Feed Level. |
| `3` | **Clean Coop** | 1 hr | Boosts Hen Health. |
| `4` | **Clean Pens** | 1 hr | Boosts Pig Health. |
| `5` | **Inspect Health** | 1 hr | Provides minor health boost & prevents decay for 1 step. |

### Reward Dynamics
* **+ (Positive):** Continuous reward scaled by current health scores `(hen_health * 2.5) + (pig_health * 2.5)`.
* **+ (Bonus):** `+10` for surviving the full 24-hour cycle.
* **- (Penalty):** `-5` for attempting an action without sufficient labor hours.
* **- (Catastrophic):** `-20` if any livestock health drops below `0.20` (triggers early termination).

---

## Algorithms & Hyperparameter Tuning

Four distinct Reinforcement Learning algorithms were implemented and evaluated. Each algorithm underwent rigorous hyperparameter tuning with **10 unique configurations** (40 total experiments), analyzing the effects of learning rate, gamma, entropy coefficients, and batch sizes.

1. **DQN (Value-Based):** Implemented via Stable-Baselines3.
2. **PPO (Policy Gradient):** Implemented via Stable-Baselines3.
3. **A2C (Actor-Critic):** Implemented via Stable-Baselines3.
4. **REINFORCE (Vanilla PG):** Custom PyTorch implementation using Monte Carlo returns with baseline subtraction.

*All tuning results, including CSVs and performance plots, are generated automatically in the `/outputs` directory.*

---

## Production Integration & API Readiness

To bridge the gap between simulation and real-world application (as per the mission to build an AIoT ecosystem), this RL agent is designed for production integration:
* **State Serialization:** The `_get_info()` dictionary allows the environment's telemetry to be instantly serialized into JSON.
* **Architecture:** In a production pipeline, a **NestJS** backend can serve the trained model's live predictions (from `main.py`) via a REST or WebSocket API to a **Next.js** frontend dashboard, providing farm workers with real-time, AI-driven task scheduling.

---

## Repository Structure

```text
Egide_Harerimana_rl_summative/
├── environment/
│   ├── custom_env.py        # Core Gymnasium environment logic
│   ├── rendering.py         # Advanced Pygame 2D Visualization
│   └── __init__.py
├── training/
│   ├── dqn_training.py      # DQN training loop & grid search
│   └── pg_training.py       # PPO, A2C, and REINFORCE training & grid search
├── models/
│   ├── dqn/                 # Saved DQN model weights (*.zip)
│   ├── ppo/                 # Saved PPO model weights (*.zip)
│   ├── a2c/                 # Saved A2C model weights (*.zip)
│   └── reinforce/           # Saved REINFORCE weights (*.pt)
├── outputs/                 # CSV result tables & matplotlib graphs
├── main.py                  # Entry point: runs best agent with GUI rendering
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Setup

```sh
git clone https://github.com/degide/Egide_Harerimana_rl_summative.git
cd Egide_Harerimana_rl_summative
pip install -r requirements.txt
```

## Training

```sh
# Train DQN models
python training/dqn_training.py

# Train PPO, A2C, and REINFORCE models
python training/pg_training.py
```

## Live Simulation

Once training is complete, run the main script to automatically load the best-performing model (based on highest mean reward) and visualize its decision-making process in the Pygame dashboard.

```sh
python main.py --episodes 5
```

## Lisence

[MIT](LICENSE)