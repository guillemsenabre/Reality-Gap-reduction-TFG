# Enhancing MARL for Reality Gap reduction

Welcome to my Final Bachelor Project!

### Scope and Goals

This project is divided in two main parts, simulation and implementation. The goal is to apply some techniques to diminish the reality gap when transfering trained MARL (Multi-Agent RL) from the simulation to the real-world application.

### Simulation

Gazebo simulator (https://gazebosim.org/home) is used to cope with the simulation used in the paper "Distributed RL for Cooperative Multi-Robot Object Manipulation". Specifically, Gazebo Fortress due to its compatibility with ROS 2 Humble, for robot control.


![image](https://github.com/BakiRhina/Reality-Gap-reduction-TFG/assets/108484177/67cd12ea-3b7f-4cdb-a8ec-ab4472240e2e)



Both in simulation and in real-life applications, two robotic arms composed by 4 joints and a gripper as end-effector will be used. The environment has both robots on a desktop table facing each other. In between them there is an object to be lifted, carried and placed in a specific area. The following screenshot shows the environment in a visual way:

<img width="706" alt="image" src="https://github.com/BakiRhina/Reality-Gap-reduction-TFG/assets/108484177/db0fb80f-8833-45d1-93c4-047db9460709">

The structure of the reinforcement learning algorithm contains the environment states, a distribution probability of possible actions to be taken by the agents in a certain state, the reward function, the policy and the value function. They will all be explained below!
