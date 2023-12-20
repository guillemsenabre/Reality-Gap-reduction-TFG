# Enhancing MARL for Reality Gap reduction

Welcome to my Final Bachelor Thesis!

### Scope and Goals

This project is divided in two main parts, simulation and implementation. The goal is to apply some techniques to diminish the reality gap when transfering trained MARL (Multi-Agent RL) from the simulation to the real-world application.

### Simulation

Gazebo simulator (https://gazebosim.org/home) is used to cope with the simulation used in the paper "Distributed RL for Cooperative Multi-Robot Object Manipulation". Specifically, Gazebo Fortress due to its compatibility with ROS 2 Humble, for robot control.


![image](https://github.com/BakiRhina/Reality-Gap-reduction-TFG/assets/108484177/67cd12ea-3b7f-4cdb-a8ec-ab4472240e2e)



Both in simulation and in real-life applications, two robotic arms composed by 5 joints and a gripper as end-effector will be used. The environment has both robots on a desktop table facing each other. In between them there is an object to be lifted, carried and placed in a specific area. The following screenshot shows the environment in a visual way:

<img width="706" alt="image" src="https://github.com/BakiRhina/Reality-Gap-reduction-TFG/assets/108484177/db0fb80f-8833-45d1-93c4-047db9460709">



Both robots are learning through a custom algorithm based on DDPG (Deep Deterministic Policy Gradient) agent. The structure of it is explained below:

### DDPG Agent Structure and Operation

The DDPG agent, or Deep Deterministic Policy Gradient agent, is a powerful algorithm employed in this project to facilitate the learning process of the robotic arms in a multi-agent reinforcement learning (MARL) setting. Below is a breakdown of its structure and operation:

#### Actor-Critic Architecture:

The DDPG agent follows an actor-critic architecture, consisting of two main components:

1. **Actor Network:**
   - The actor network is responsible for learning and representing the policy function. In the context of robotic control, the policy defines the optimal actions the robotic arms should take in different states to achieve the desired objectives.
   - It takes the current state of the environment as input and outputs a continuous action space, which represents the joint angles and gripper position of the robotic arms.

2. **Critic Network:**
   - The critic network evaluates the actions chosen by the actor by estimating the expected cumulative reward. It helps the agent understand the value of its chosen actions in a given state.
   - The critic network takes both the current state and the action as inputs and outputs a Q-value, representing the expected cumulative reward.

#### Training Process:

The DDPG agent is trained through an iterative process involving the following steps:

1. **Experience Replay:**
   - To enhance learning stability, the agent employs experience replay, also called buffer replay (the class containing this file in the project has this name). It stores and randomly samples past experiences (state, action, reward, next state) from its replay buffer during training.

2. **Target Networks:**
   - The agent utilizes target networks for both the actor and critic. These target networks slowly track the learned networks to provide more stable target values during the training process.

3. **Bellman Equation:**
   - The agent optimizes its policy by minimizing the temporal difference error, computed using the Bellman equation. This guides the agent towards actions that maximize expected cumulative reward over time.
  
Bellman equation definition:

      Q(s,a)=E[r+γ⋅max a′Q(s′,a′)∣s,a]

Python Implementation:

```python
Q_sa = reward + gamma * torch.max(Q_s_prime_a_prime)
```

4. **Gradient Ascent:**
   - The actor network is updated through gradient ascent, aiming to increase the expected cumulative reward. The critic network is updated to minimize the temporal difference error.

#### Why DDPG:

DDPG is chosen for its suitability in continuous action spaces, making it well-suited for robotic control tasks where actions are often parameterized. Its ability to handle high-dimensional action spaces and continuous state spaces aligns with the requirements of the multi-agent robotic arms scenario.

By leveraging the DDPG algorithm, this project aims to bridge the gap between simulation and real-world application, enabling a smoother transfer of knowledge from simulated environments to actual robotic systems.
