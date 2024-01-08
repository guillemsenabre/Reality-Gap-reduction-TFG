# Enhancing MARL for Reality Gap reduction

Welcome to my Final Bachelor Thesis!

---

## Introduction

Reinforcement Learning (RL) is a computational approach that enables agents to learn optimal behaviors through interaction with an environment. Multi-Agent Reinforcement Learning (MARL) extends this concept to multiple interacting agents, allowing them to learn and adapt collectively. However, one significant challenge in deploying RL agents in the real world is the "reality gap." The reality gap refers to the mismatch between the training environment (simulation) and the real-world conditions, which can hinder the performance of learned policies when applied to the actual scenario. This project aims to address the reality gap in MARL for more effective real-world deployment.

---

### Scope and Goals

This project is divided in two main parts, simulation and implementation. The goal is to apply some techniques to diminish the reality gap when transfering trained MARL (Multi-Agent RL) from the simulation to the real-world application.

---

## Replicating the Project

To replicate this project, follow the steps outlined below:

### Simulation

Gazebo simulator (https://gazebosim.org/home) is used to cope with the simulation used in the paper "Distributed RL for Cooperative Multi-Robot Object Manipulation". Specifically, Gazebo Fortress due to its compatibility with ROS 2 Humble, for robot control.


![image](https://github.com/BakiRhina/Reality-Gap-reduction-TFG/assets/108484177/67cd12ea-3b7f-4cdb-a8ec-ab4472240e2e)



Both in simulation and in real-life applications, two robotic arms composed by 4 joint. The environment has both robots on a desktop table facing each other. In between them there is an object to be lifted, carried and placed in a specific area. The following screenshot shows the environment in a visual way:

<img width="706" alt="image" src="https://github.com/BakiRhina/Reality-Gap-reduction-TFG/assets/108484177/db0fb80f-8833-45d1-93c4-047db9460709">

### Step by step tutorial to launch the simulation

1. **Software Requirements:**
   - Install ROS 2 Humble and Gazebo Fortress (Ignition gazebo).
   - Alternatively, consider using other simulators like NVIDIA's Isaac Gym for ease of use and if full customization is not required.

2. **Programming Languages:**
   - Utilize Python, preferably versions greater than 3.10.

3. **Setting up the Simulation:**
   - Navigate to the Simulation folder containing a ROS workspace with a package named `arm_pkg` that encompasses the simulation.
   - Open a terminal and build the workspace locally with the command: `colcon build`.

4. **Running the Simulation:**
   - Execute the following command to launch the necessary packages from the custom launch file: `ros2 launch arm_pkg launch`.
   - Simultaneously, a bridge is required to facilitate data transfer from Gazebo to ROS2. Refer to `/bridge_commands/rosgzbridge.txt` and copy the last command specified, which includes the necessary data for simulating two robots with 5 joints each.

5. **Training the Agents:**
   - Once the bridge is operational, run the launch file. This action will open a Gazebo window displaying the simulation, and after a brief initialization period, the training process will commence.
   - The simulation will continue until a predefined terminal condition is met, at which point the results will be automatically presented.

By carefully following these steps, one can replicate the project and delve into the efforts aimed at enhancing MARL to mitigate the challenges posed by the reality gap.