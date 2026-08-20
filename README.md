# Multi-AI Agent for Swarm UAV Operation
**Major Project Summary & Executive Technical Report**

---

## PAGE 1: PROJECT OVERVIEW, MATHEMATICAL MODEL & ARCHITECTURE

### 1. Executive Summary & Problem Formulation
The autonomous coordination of Unmanned Aerial Vehicle (UAV) swarms presents substantial advantages over single-UAV systems in terms of scalability, operational redundancy, and spatial area coverage. However, establishing decentralized, reliable control for multi-agent systems is challenging due to the dynamic nature of operational environments and the issue of **non-stationarity**. In a multi-drone setup, simultaneous learning causes the environment to shift continuously from the perspective of each individual agent, causing standard single-agent algorithms (e.g., standard DDPG or DQN) to fail or diverge.

This project implements a Multi-Agent Deep Reinforcement Learning (MARL) framework based on the **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** algorithm operating under the **Centralized Training, Decentralized Execution (CTDE)** paradigm[cite: 1, 4]. The framework enables a swarm of 3 UAVs to autonomously allocate targets, coordinate continuous flight trajectories, and prevent inter-agent collisions within a continuous 2D operational space[cite: 4, 5, 6].

---

### 2. Theoretical Framework & State-Action Representation
The swarm coordination problem is formulated as a multi-agent Markov Game defined by the tuple $\langle \mathcal{S}, \{\mathcal{A}_i\}_{i=1}^N, \mathcal{P}, \{\mathcal{R}_i\}_{i=1}^N, \gamma \rangle$[cite: 1, 4]:
* **Observation Space ($\mathcal{S}_i \in \mathbb{R}^{17}$):** Each UAV agent receives a normalized 17-dimensional local observation vector[cite: 6]:
  * **Self Kinematics (4D):** Normalized position $(x/W, y/H)$ and velocities $(v_x/v_{max}, v_y/v_{max})$[cite: 6].
  * **Peer Telemetry (4D):** Relative spatial distance vectors to peer drones $\Delta p_{ij} = (p_j - p_i) / W$[cite: 6].
  * **Target Tracking (9D):** Relative positional vectors to each target and a binary indicator of target visitation status[cite: 6].
* **Continuous Action Space ($\mathcal{A}_i \in [-1, 1]^2$):** Each agent outputs continuous steering accelerations $(\Delta v_x, \Delta v_y)$ mapped through a hyperbolic tangent (`Tanh`) activation layer[cite: 4, 6].

---

### 3. Neural Network Architecture & CTDE Paradigm
To balance computational training complexity with real-time onboard inference, the system divides processing into an asymmetric Actor-Critic design[cite: 1, 4]:
* **Decentralized Actor Network ($\mu_{\theta_i}$):** A Multi-Layer Perceptron (MLP) consisting of an input layer ($17$ nodes), two hidden layers ($256$ and $128$ units) with `ReLU` activations and layer initialization, and a `Tanh` output layer ($2$ continuous actions)[cite: 4]. During runtime, each UAV executes inference purely using its local observation[cite: 1, 4].
* **Centralized Critic Network ($Q_{\phi_i}$):** Evaluates joint state-action pairs using global information ($51$ state dimensions $+ 6$ action dimensions across all $3$ agents)[cite: 4, 5]. The global state is processed through linear layers ($256$ units) concatenated with joint actions, passing through a second hidden layer ($128$ units) to output a scalar $Q$-value approximation[cite: 4].

---

### 4. Reward Shaping & Objective Functions
To overcome sparse reward exploration bottlenecks, a potential-based reward function $\mathcal{R}_i$ is utilized[cite: 1, 6]:
$$\mathcal{R}_i = 0.3 \cdot (d_{t-1} - d_t) + R_{\text{team}} - P_{\text{collision}} - P_{\text{boundary}} - P_{\text{idle}}$$
* **Distance Gradient:** Potential-based reward shaping provides continuous feedback proportional to spatial progress toward the nearest unvisited target ($+0.3 \Delta d$)[cite: 6].
* **Team Acquisition Reward ($R_{\text{team}}$):** A shared $+50.0$ team bonus upon visiting an active target radius ($18\text{ px}$)[cite: 6].
* **Safety Penalties:** Inter-agent collision penalty ($-1.5$ per colliding pair), out-of-bounds boundary penalty ($-2.0$), and low-velocity anti-idling penalty ($-0.05$)[cite: 6].

\newpage

## PAGE 2: IMPLEMENTATION, RESULTS, AND SYSTEM VALIDATION

### 5. Implementation Stack & Training Infrastructure
The core software components are developed natively in Python using PyTorch, Pygame, and NumPy[cite: 1, 4, 6]:
* **Algorithm Core (`MADDPG.py`):** Encapsulates experience replay sampling ($1,000,000$ buffer capacity, batch size of $256$), decaying Gaussian exploration noise ($\sigma_{\text{start}}=1.0$, decay rate $=0.9999$, $\sigma_{\text{min}}=0.01$), and Polyak soft target updates ($\tau = 0.001$)[cite: 4].
* **Physics & Environment Simulation (`uavsimulation.py`):** Computes continuous kinematic updates ($\mathbf{v} \leftarrow \mathbf{v} + \mathbf{a}$, $\vert{}\mathbf{v}\vert{} \le v_{\text{max}}$), circular hitbox collision detections, and real-time visualization routines[cite: 6].
* **Training Orchestration (`TRAING_SCRIPT.py`):** Runs a 2000-episode pipeline ($500$ maximum steps per episode), logs cumulative returns using moving averages, updates networks via the Adam optimizer ($\alpha_{\text{actor}} = 10^{-4}, \alpha_{\text{critic}} = 3\times 10^{-4}$), and saves trained model weights[cite: 4, 5].
* **Inference Visualizer (`VISUALIZATION.PY`):** Deploys saved neural weights to evaluate deterministic, real-time swarm execution[cite: 7].

```text
       CENTRALIZED TRAINING (Ground Station / Server)
 ┌─────────────────────────────────────────────────────────────┐
 │  Global Replay Buffer: 1,000,000 Transitions (Batch = 256)  │[cite: 4]
 │  Centralized Critic Loss: L(θ_i) = E[(Q(S,A) - y)^2]        │[cite: 1, 4]
 └──────────────────────────────┬──────────────────────────────┘
                                │ Soft Updates (τ = 0.001)[cite: 4]
                                ▼
       DECENTRALIZED EXECUTION (Edge UAV Nodes)
 ┌──────────────────────┐┌──────────────────────┐┌──────────────────────┐
 │ UAV 1: Actor Policy  ││ UAV 2: Actor Policy  ││ UAV 3: Actor Policy  │[cite: 4, 5]
 │ Input: o_1 -> a_1    ││ Input: o_2 -> a_2    ││ Input: o_3 -> a_3    │[cite: 1, 4]
 └──────────────────────┘└──────────────────────┘└──────────────────────┘
