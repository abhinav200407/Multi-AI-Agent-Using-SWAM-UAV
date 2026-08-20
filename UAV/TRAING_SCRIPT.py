import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os

# --- Use the final, improved versions of the files ---
from uavsimulation import SwarmEnv
from MADDPG import MADDPGAgent, ReplayBuffer, BATCH_SIZE

# --- Constants ---
N_UAVS = 3
N_TARGETS = 3
N_EPISODES = 2000
MAX_T = 500 
PRINT_EVERY = 100
MODEL_DIR = "models"
GAMMA = 0.99

def plot_metrics(all_scores, actor_losses, critic_losses):
    """Generates and displays plots for scores and network losses."""
    moving_avg = [np.mean(all_scores[max(0, i-100):i+1]) for i in range(len(all_scores))]
            
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Training Performance Metrics', fontsize=16)

    # Plot 1: Scores
    axs[0].plot(all_scores, label='Score per Episode', alpha=0.5)
    axs[0].plot(moving_avg, label='100-Episode Moving Average', color='orange', linewidth=2)
    axs[0].set_ylabel('Score')
    axs[0].set_title('Agent Scores')
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Actor Loss
    axs[1].plot(actor_losses, color='tab:red')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Actor Network Loss (Policy)')
    axs[1].grid(True)

    # Plot 3: Critic Loss
    axs[2].plot(critic_losses, color='tab:blue')
    axs[2].set_ylabel('Loss')
    axs[2].set_xlabel('Training Step #')
    axs[2].set_title('Critic Network Loss (Value)')
    axs[2].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

def train():
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

    env = SwarmEnv(num_uavs=N_UAVS, num_targets=N_TARGETS)
    
    initial_states = env.reset()
    state_size = initial_states[0].shape[0]
    action_size = 2
    
    agents = [MADDPGAgent(state_size, action_size, i, N_UAVS, random_seed=0) for i in range(N_UAVS)]
    memory = ReplayBuffer(action_size, int(1e6), BATCH_SIZE, seed=0)
    
    all_scores, scores_deque = [], deque(maxlen=PRINT_EVERY)
    actor_losses, critic_losses = [], []

    for i_episode in range(1, N_EPISODES + 1):
        states = env.reset()
        episode_scores = np.zeros(N_UAVS)
        
        for t in range(MAX_T):
            actions = [agents[i].act(states[i]) for i in range(N_UAVS)]
            next_states, rewards, done, _ = env.step(actions)
            
            memory.add(np.concatenate(states), np.concatenate(actions), rewards, np.concatenate(next_states), done)
            
            if len(memory) > BATCH_SIZE:
                for i in range(N_UAVS):
                    experiences = memory.sample()
                    with torch.no_grad():
                         all_next_actions = torch.cat([agents[j].actor_target(experiences[3][:, j*state_size:(j+1)*state_size]) for j in range(N_UAVS)], dim=1)
                    
                    all_actions = experiences[1]
                    agent_experiences = (experiences[0], experiences[1], experiences[2][:,i].unsqueeze(1), experiences[3], experiences[4])
                    
                    actor_loss, critic_loss = agents[i].learn(agent_experiences, GAMMA, all_next_actions, all_actions)
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
            
            states = next_states
            episode_scores += rewards
            if done: break
        
        # --- KEY CHANGE: Decay noise after each episode ---
        for agent in agents:
            agent.decay_noise()

        avg_score = np.mean(episode_scores)
        scores_deque.append(avg_score)
        all_scores.append(avg_score)
        
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}', end="")
        if i_episode % PRINT_EVERY == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}')
            for i, agent in enumerate(agents):
                torch.save(agent.actor_local.state_dict(), os.path.join(MODEL_DIR, f'actor_{i}.pth'))
                torch.save(agent.critic_local.state_dict(), os.path.join(MODEL_DIR, f'critic_{i}.pth'))

    plot_metrics(all_scores, actor_losses, critic_losses)

if __name__ == '__main__':
    train()
