import torch
import time
import pygame
import os

# --- Make sure to import from the final versions of your files ---
from uavsimulation import SwarmEnv
from MADDPG import MADDPGAgent

# --- Constants ---
MODEL_DIR = "models"
# These values MUST match the constants used during training.
N_UAVS = 3
N_TARGETS = 3

def visualize():
    """Load trained models and visualize the swarm's behavior."""
    env = SwarmEnv(num_uavs=N_UAVS, num_targets=N_TARGETS, render_mode=True)
    
    # Get state and action sizes from the environment
    initial_states = env.reset()
    state_size = initial_states[0].shape[0]
    action_size = 2
    
    # Initialize agents
    agents = [MADDPGAgent(state_size, action_size, i, N_UAVS, random_seed=0) for i in range(N_UAVS)]

    # Load the trained weights from the 'models' directory
    try:
        for i, agent in enumerate(agents):
            actor_path = os.path.join(MODEL_DIR, f'actor_{i}.pth')
            agent.actor_local.load_state_dict(torch.load(actor_path))
        print("Models loaded successfully from the 'models' directory.")
    except FileNotFoundError:
        print(f"Error: Model files not found in the '{MODEL_DIR}' directory.")
        print("Please make sure you have run 'train_final.py' and that the model files were saved.")
        return

    states = env.reset()
    step = 0
    max_steps_per_run = 1000

    running = True
    while running:
        # Check if the user has clicked the close button
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get actions from the TRAINED agents (no noise during visualization)
        actions = [agents[i].act(states[i], add_noise=False) for i in range(N_UAVS)]
        
        # Take a step in the environment
        next_states, _, done, _ = env.step(actions)
        
        # Render the current state of the simulation
        env.render(episode=1, step=step)
        
        states = next_states
        step += 1
        
        # Reset the simulation if the episode is over or runs for too long
        if done or step > max_steps_per_run:
            print("All targets visited or max steps reached. Resetting simulation in 3 seconds...")
            time.sleep(3)
            states = env.reset()
            step = 0

    pygame.quit()

if __name__ == '__main__':
    visualize()
