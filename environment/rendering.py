"""
Pygame visualization module for FarmManagementEnv.
Renders an interactive, top-down view of the farm using geometric primitives
to represent individual animals and dynamic feed troughs.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import random
import pygame

# UI Color Palette
C = {
    "bg":          (124, 204, 122),  # Grassy green
    "panel":       (45, 45, 55),     # Dark UI panel
    "text":        (240, 240, 240),
    "hen_zone":    (220, 200, 160),  # Dirt/sand coop
    "pig_zone":    (139, 115, 85),   # Muddy pen
    "health_opt":  (50, 205, 50),
    "health_crit": (220, 20, 60),
    "feed_box":    (90, 60, 30),
    "feed":        (255, 215, 0),    # Yellow grain
    "pig":         (255, 182, 193),  # Light pink
    "pig_snout":   (219, 112, 147),  # Darker pink
    "hen":         (255, 250, 240),  # Whiteish
    "hen_comb":    (255, 0, 0),      # Red comb
    "bar_bg":      (80, 80, 90),
    "action_bg":   (30, 30, 40),
    "white":       (255, 255, 255)
}

ACTION_LABELS = ["Idle", "Refill Hen Feed", "Refill Pig Feed", "Clean Coop", "Clean Pens", "Inspect Health"]

class FarmRenderer:
    W, H = 850, 650
    FPS = 5

    def __init__(self, max_steps):
        pygame.init()
        self.max_steps = max_steps
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("AI-Integrated Farm Management Simulator")
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_label = pygame.font.SysFont("Arial", 15)
        self.frame_surface = None
        
        # Initialize fixed baseline coordinates for the animals
        # 40 graphical hens represent the flock of 980
        self.hen_bases = [(random.randint(60, 320), random.randint(110, 280)) for _ in range(40)]
        # 10 individual pigs
        self.pig_bases = [(random.randint(70, 310), random.randint(390, 520)) for _ in range(10)]

    def render(self, info: dict, action: int):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        self.screen.fill(C["bg"])

        # Draw farm zones
        # Poultry coop
        pygame.draw.rect(self.screen, C["hen_zone"], (40, 50, 320, 260), border_radius=8)
        pygame.draw.rect(self.screen, (100, 80, 50), (40, 50, 320, 260), width=4, border_radius=8) 
        
        # Pig Pens
        pygame.draw.rect(self.screen, C["pig_zone"], (40, 330, 320, 220), border_radius=8)
        pygame.draw.rect(self.screen, (80, 60, 40), (40, 330, 320, 220), width=4, border_radius=8)

        # Labels
        self.screen.blit(self.font_title.render("Poultry Coop (980 Hens)", True, (40,40,40)), (50, 60))
        self.screen.blit(self.font_title.render("Pig Pens (10 Pigs)", True, (40,40,40)), (50, 340))

        # Draw troughs
        self._draw_trough(180, 85, 150, 20, info["hen_feed"])
        self._draw_trough(180, 365, 150, 20, info["pig_feed"])

        # Draw animals (with slight random jitter to simulate life)
        for bx, by in self.hen_bases:
            jx = bx + random.randint(-3, 3)
            jy = by + random.randint(-3, 3)
            pygame.draw.circle(self.screen, C["hen"], (jx, jy), 6) # Body
            pygame.draw.circle(self.screen, C["hen_comb"], (jx+4, jy-3), 2) # Comb

        for bx, by in self.pig_bases:
            jx = bx + random.randint(-2, 2)
            jy = by + random.randint(-2, 2)
            pygame.draw.ellipse(self.screen, C["pig"], (jx, jy, 24, 16)) # Body
            pygame.draw.ellipse(self.screen, C["pig_snout"], (jx+18, jy+4, 8, 8)) # Snout

        # Action indicators
        if action == 1: # Refill Hen
            pygame.draw.rect(self.screen, C["health_opt"], (40, 50, 320, 260), width=6, border_radius=8)
        elif action == 2: # Refill Pig
            pygame.draw.rect(self.screen, C["health_opt"], (40, 330, 320, 220), width=6, border_radius=8)
        elif action == 3: # Clean Hen
            pygame.draw.circle(self.screen, C["white"], (330, 280), 15)
        elif action == 4: # Clean Pig
            pygame.draw.circle(self.screen, C["white"], (330, 520), 15)

        # Stats panel
        pygame.draw.rect(self.screen, C["panel"], (400, 50, 410, 500), border_radius=10)
        self.screen.blit(self.font_title.render("AI Management Telemetry", True, C["text"]), (430, 70))

        y = 130
        self._draw_gauge(430, y, 350, "Hen Health", info["hen_health"])
        self._draw_gauge(430, y+70, 350, "Hen Feed Level", info["hen_feed"], color=C["feed"])
        self._draw_gauge(430, y+140, 350, "Pig Health", info["pig_health"])
        self._draw_gauge(430, y+210, 350, "Pig Feed Level", info["pig_feed"], color=C["feed"])
        
        self.screen.blit(self.font_label.render(f"Labor Available: {info['labor']} hrs", True, C["text"]), (430, y+300))
        self.screen.blit(self.font_label.render(f"Cumulative Reward: {info['total_reward']}", True, C["text"]), (430, y+330))

        # --- Draw Bottom Bar ---
        pygame.draw.rect(self.screen, C["action_bg"], (0, 590, 850, 60))
        status = f"Hour {info['hour']:02d}/24   |   Action: {ACTION_LABELS[action]}   |   Status: {'ALIVE' if info['alive'] else 'CRITICAL'}"
        self.screen.blit(self.font_title.render(status, True, C["white"]), (30, 605))

        self.frame_surface = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def _draw_trough(self, x, y, w, h, fill_pct):
        pygame.draw.rect(self.screen, C["feed_box"], (x, y, w, h))
        if fill_pct > 0:
            fill_w = int(w * fill_pct)
            pygame.draw.rect(self.screen, C["feed"], (x+2, y+2, fill_w-4, h-4))

    def _draw_gauge(self, x, y, w, label, value, color=None):
        if color is None:
            color = C["health_opt"] if value > 0.5 else C["health_crit"]
            
        self.screen.blit(self.font_label.render(f"{label}: {value:.2f}", True, C["text"]), (x, y))
        pygame.draw.rect(self.screen, C["bar_bg"], (x, y+25, w, 20), border_radius=5)
        pygame.draw.rect(self.screen, color, (x, y+25, int(w * value), 20), border_radius=5)

    def get_rgb_array(self):
        return self.frame_surface

    def close(self):
        pygame.display.quit()
        pygame.quit()

# --- Standalone Random Agent Demo ---

def run_random_demo():
    """
    Creates a static visualization of the environment with random actions.
    No model is loaded. This demonstrates the GUI and environment physics.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environment.custom_env import FarmManagementEnv

    print("=" * 60)
    print("  RANDOM AGENT DEMO  (no model, pure environment test)")
    print("=" * 60)

    # Initialize environment in human render mode
    env = FarmManagementEnv(render_mode="human")

    # Simulate 3 complete 24-hour cycles (days)
    for ep in range(3):
        obs, info = env.reset()
        done = False
        step = 0
        total_r = 0.0

        print(f"\n--- Episode {ep + 1} ---")
        while not done:
            # The agent takes a purely random action (0 to 5)
            action = env.action_space.sample()   
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += reward
            step += 1

            # Print terminal logs every 4 hours to match the visual
            if step % 4 == 0 or done:
                print(f"  Hour {info['hour']:02d} | Action: {action} "
                      f"| HenHealth={info['hen_health']:.2f} "
                      f"| PigHealth={info['pig_health']:.2f} "
                      f"| Reward={reward:+.2f}")

        print(f"  Day ended. Total reward: {total_r:.1f} | Livestock Alive: {info['alive']}")

    env.close()
    print("\nDemo complete.")

if __name__ == "__main__":
    run_random_demo()