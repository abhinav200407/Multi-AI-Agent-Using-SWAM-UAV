import pygame
import numpy as np

# --- Constants ---
WIDTH, HEIGHT = 800, 600
UAV_RADIUS = 10
TARGET_RADIUS = 8
COLLISION_DISTANCE = 2 * UAV_RADIUS
TARGET_REACH_DISTANCE = UAV_RADIUS + TARGET_RADIUS

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class UAV:
    """Represents a single UAV agent."""
    def __init__(self, uav_id, x, y):
        self.id = uav_id
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.zeros(2)
        self.max_speed = 3
        self.color = (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200))

    def update(self, action):
        action = np.squeeze(action)
        self.vel += action
        
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed
            
        self.pos += self.vel

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.pos.astype(int), UAV_RADIUS)
        end_pos = self.pos + self.vel * 5
        pygame.draw.line(screen, WHITE, self.pos.astype(int), end_pos.astype(int), 2)

class Target:
    """Represents a target."""
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=np.float64)
        self.color = GREEN

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.pos.astype(int), TARGET_RADIUS)

class SwarmEnv:
    """The main simulation environment."""
    def __init__(self, num_uavs=3, num_targets=3, render_mode=False):
        self.num_uavs = num_uavs
        self.num_targets = num_targets
        
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("UAV Swarm Simulation")
            self.font = pygame.font.SysFont(None, 24)

    def reset(self):
        """Resets the environment."""
        self.uavs = [UAV(i, *np.random.uniform([50, 50], [150, HEIGHT - 50])) for i in range(self.num_uavs)]
        self.targets = [Target(*np.random.uniform([WIDTH - 150, 50], [WIDTH - 50, HEIGHT - 50])) for _ in range(self.num_targets)]
        self.visited_targets = [False] * self.num_targets
        return self._get_states()

    def step(self, actions):
        """Takes a step in the environment."""
        dist_before = self._get_min_distances_to_targets()
        
        for i, uav in enumerate(self.uavs):
            uav.update(actions[i])
        
        dist_after = self._get_min_distances_to_targets()
        
        rewards = self._calculate_rewards(dist_before, dist_after)
        next_states = self._get_states()
        done = all(self.visited_targets)
        
        return next_states, rewards, done, {}

    def _get_states(self):
        """Gets the state observation for each agent."""
        all_states = []
        for i, agent in enumerate(self.uavs):
            own_state = [agent.pos[0] / WIDTH, agent.pos[1] / HEIGHT, agent.vel[0] / agent.max_speed, agent.vel[1] / agent.max_speed]
            
            other_uav_states = []
            for j, other_agent in enumerate(self.uavs):
                if i == j: continue
                other_uav_states.extend((other_agent.pos - agent.pos) / WIDTH)

            target_states = []
            for k, target in enumerate(self.targets):
                target_states.extend((target.pos - agent.pos) / WIDTH)
                target_states.append(1.0 if self.visited_targets[k] else 0.0)

            all_states.append(np.array(own_state + other_uav_states + target_states))
        return all_states

    def _get_min_distances_to_targets(self):
        """Calculates distance from each UAV to its nearest unvisited target."""
        min_dists = np.full(self.num_uavs, float('inf'))
        unvisited_targets = [t.pos for i, t in enumerate(self.targets) if not self.visited_targets[i]]
        if not unvisited_targets:
            return np.zeros(self.num_uavs)
            
        for i, uav in enumerate(self.uavs):
            distances = [np.linalg.norm(uav.pos - t_pos) for t_pos in unvisited_targets]
            if distances:
                min_dists[i] = np.min(distances)
        return min_dists

    def _calculate_rewards(self, dist_before, dist_after):
        """--- KEY CHANGE: A more robust reward function ---"""
        rewards = np.zeros(self.num_uavs)

        # 1. Strong "getting warmer" signal (Potential-Based Reward Shaping)
        distance_improvement = dist_before - dist_after
        rewards += distance_improvement * 0.3  # Increased scaling factor

        for i, uav in enumerate(self.uavs):
            # 2. Penalty for being out of bounds
            if not (0 < uav.pos[0] < WIDTH and 0 < uav.pos[1] < HEIGHT):
                rewards[i] -= 2.0

            # 3. Small "anti-lazy" penalty to encourage movement
            if np.linalg.norm(uav.vel) < 0.1:
                rewards[i] -= 0.05

        # 4. Collision penalty
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                if np.linalg.norm(self.uavs[i].pos - self.uavs[j].pos) < COLLISION_DISTANCE:
                    rewards[i] -= 1.5
                    rewards[j] -= 1.5

        # 5. Massive, shared team reward for reaching a target
        for j, target in enumerate(self.targets):
            if not self.visited_targets[j]:
                for uav in self.uavs:
                    if np.linalg.norm(uav.pos - target.pos) < TARGET_REACH_DISTANCE:
                        self.visited_targets[j] = True
                        target.color = RED
                        rewards += 50.0  # Huge reward for the whole team
                        break # Prevent multiple rewards for the same target
        return rewards

    def render(self, episode=None, step=None):
        if not self.render_mode: return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self.screen.fill(BLACK)
        for uav in self.uavs: uav.draw(self.screen)
        for target in self.targets: target.draw(self.screen)
        if episode is not None:
            self.screen.blit(self.font.render(f"Episode: {episode}", True, WHITE), (10, 10))
        if step is not None:
            self.screen.blit(self.font.render(f"Step: {step}", True, WHITE), (10, 30))
        pygame.display.flip()
        pygame.time.delay(10)