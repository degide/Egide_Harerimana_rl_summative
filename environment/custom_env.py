"""
FarmManagementEnv: A Custom Gymnasium Environment

Agent Mission: Optimize labor scheduling and feed distribution to maximize the health 
of 980 laying hens and 10 pigs within a 24-hour daily cycle. This represents
a simplified digital twin of a mixed-livestock agricultural operation.

Observation Space (6 continuous features, normalized [0, 1]):
  [time_norm, hen_feed, pig_feed, hen_health, pig_health, labor_norm]

Action Space: Discrete(6)
  0 -> Idle (conserves labor hours)
  1 -> Refill Hen Feed & Water
  2 -> Refill Pig Feed & Water
  3 -> Clean Poultry Coop
  4 -> Clean Pig Pens
  5 -> Conduct Health Inspection
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FarmManagementEnv(gym.Env):
    """
    Custom Environment that follows the Gymnasium interface.
    Simulates the daily resource and labor allocation required for a multi-species livestock operation.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    MAX_STEPS = 24  # Total hours in a simulated day
    MAX_LABOR = 8.0 # Total worker hours available per day

    def __init__(self, render_mode=None):
        """
        Initializes the environment, defines action and observation spaces.
        
        Args:
            render_mode (str, optional): The mode to render the environment ('human' or 'rgb_array').
        """
        super().__init__()
        self.render_mode = render_mode

        # Define the discrete action space (6 possible actions)
        self.action_space = spaces.Discrete(6)

        # Define the continuous observation space (all values normalized between 0.0 and 1.0)
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32
        )

        self._renderer = None

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state at the start of a new episode (day).
        
        Args:
            seed (int, optional): Seed for random number generation to ensure reproducibility.
            options (dict, optional): Additional options for reset.
            
        Returns:
            tuple: (observation, info) representing the initial state.
        """
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize state variables with slight randomness for varied starting conditions
        self.hen_feed   = float(self.np_random.uniform(0.6, 0.9))
        self.pig_feed   = float(self.np_random.uniform(0.6, 0.9))
        self.hen_health = float(self.np_random.uniform(0.8, 1.0))
        self.pig_health = float(self.np_random.uniform(0.8, 1.0))
        self.labor      = 1.0 # Start with 100% of the 8 available hours
        
        self.total_reward = 0.0
        self.last_action  = 0
        self.livestock_alive = True

        obs = self._get_obs()
        info = self._get_info()

        # Initialize the renderer if in human mode
        if self.render_mode == "human":
            self._init_renderer()
            
        return obs, info

    def step(self, action: int):
        """
        Executes a single time step within the environment based on the agent's action.
        
        Args:
            action (int): The discrete action chosen by the RL agent.
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        self.last_action = action
        reward = 0.0

        # Calculate the normalized cost of 1 hour of labor
        labor_cost = 1.0 / self.MAX_LABOR 
        
        # Process the action if the agent chose to work (action > 0)
        if action != 0:
            if self.labor >= labor_cost:
                # Deduct labor and apply action effects
                self.labor -= labor_cost
                if action == 1: self.hen_feed = min(1.0, self.hen_feed + 0.4)
                elif action == 2: self.pig_feed = min(1.0, self.pig_feed + 0.4)
                elif action == 3: self.hen_health = min(1.0, self.hen_health + 0.05)
                elif action == 4: self.pig_health = min(1.0, self.pig_health + 0.05)
                elif action == 5: 
                    # Health inspection yields a small immediate health boost
                    self.hen_health = min(1.0, self.hen_health + 0.02)
                    self.pig_health = min(1.0, self.pig_health + 0.02)
            else:
                # Penalty: The agent attempted a task requiring labor, but no hours remain
                reward -= 5.0

        # Advance the simulation clock by 1 hour
        self.current_step += 1
        
        # Apply hourly environmental decay (simulating 980 hens and 10 pigs eating)
        self.hen_feed = max(0.0, self.hen_feed - 0.08)
        self.pig_feed = max(0.0, self.pig_feed - 0.10)
        
        # Apply health decay if feed levels drop below critical thresholds (20%)
        if self.hen_feed < 0.2: self.hen_health -= 0.05
        if self.pig_feed < 0.2: self.pig_health -= 0.05

        # Calculate shaped reward based on maintaining high health scores
        reward += (self.hen_health * 2.5) + (self.pig_health * 2.5)

        # Check terminal conditions
        terminated = False
        if self.hen_health < 0.2 or self.pig_health < 0.2:
            # Episode ends early if livestock falls into critical condition
            self.livestock_alive = False
            reward -= 20.0 
            terminated = True
        elif self.current_step >= self.MAX_STEPS:
            # Episode ends successfully if the 24-hour cycle is completed
            reward += 10.0 
            terminated = True

        truncated = False # Not using explicit time limits outside of standard episode length
        self.total_reward += reward

        obs = self._get_obs()
        info = self._get_info()

        # Update visuals if rendering
        if self.render_mode == "human":
            self._render_frame()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Renders the environment based on the specified render_mode."""
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()

    def close(self):
        """Cleans up rendering resources when the environment is closed."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_obs(self) -> np.ndarray:
        """Helper to construct the normalized observation array."""
        return np.array([
            self.current_step / self.MAX_STEPS,
            self.hen_feed,
            self.pig_feed,
            self.hen_health,
            self.pig_health,
            self.labor
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        """Helper to construct the info dictionary for logging and rendering."""
        return {
            "hour": self.current_step,
            "hen_feed": round(self.hen_feed, 2),
            "pig_feed": round(self.pig_feed, 2),
            "hen_health": round(self.hen_health, 2),
            "pig_health": round(self.pig_health, 2),
            "labor": round(self.labor * self.MAX_LABOR, 1), # Convert back to actual hours
            "total_reward": round(self.total_reward, 2),
            "alive": self.livestock_alive
        }

    def _init_renderer(self):
        """Lazy initialization of the Pygame renderer."""
        from environment.rendering import FarmRenderer
        if self._renderer is None:
            self._renderer = FarmRenderer(self.MAX_STEPS)

    def _render_frame(self):
        """Triggers a single frame update in Pygame."""
        if self._renderer is None:
            self._init_renderer()
        self._renderer.render(self._get_info(), self.last_action)

    def _render_rgb(self):
        """Returns the raw RGB array of the Pygame surface."""
        if self._renderer is None:
            self._init_renderer()
        return self._renderer.get_rgb_array()