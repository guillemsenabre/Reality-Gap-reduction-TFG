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
   - Clone the Simulation repository into your local folder with `git clone https://github.com/BakiRhina/Reality-Gap-reduction-TFG.git`
   - Navigate to the Simulation folder containing the ROS workspace with a package named `arm_pkg` that encompasses the simulation.
   - Open a terminal and build the workspace with the command: `colcon build`. This will install necessary dependencies among other files.

4. **Running the Simulation:**
   - Execute the following command to launch the necessary packages from the custom launch file: `ros2 launch arm_pkg launch`.

5. **Bridge Gazebo and ROS 2 with ros_gz_bridge**
   - Simultaneously, a bridge is required to facilitate data transfer from Gazebo to ROS2. Refer to `/Simulation/src/bridge_commands/rosgzbridge.txt`. Copy and run in a separate terminal the last command specified, which includes the necessary data for simulating two robots with 5 joints each.

5. **Training the Agent:**
   - Once the bridge is operational, run the launch file with `/Simulation/src/bridge_commands/rosgzbridge.txt`. This action will open a Gazebo window displaying the simulation, and after a brief initialization period, the training process will commence.
   - The simulation will continue until a predefined terminal condition is met, at which point the results will be automatically presented.

6. **Saving the model**
   - It is possible to save the model after each episode or after an 'n' number of episodes. The model will automatically be saved in `models/`.


---

# Implementation

To implement the project and interface with physical robots, certain components and steps need to be considered.

## Components Required

- **Microcontroller:**
  - Choose between ESP32, Arduino UNO, or Raspberry Pi. The project uses ESP32.

- **PCA9685:**
  - Although not mandatory, using PCA9685 simplifies hardware complexities. 
  - 1 module can control up to 12 motors. Can upgrade adding more modules in cascade if needed.
  - Uses I2c with a default address in 0x40. 

- **Sensors:**
  - MPU6050
  - HC-SR04 

- **Actuators:**
  - Compatible with both MG995 and MG996R. MG996R is recommended for its upgraded torque and precision.

- **Breadboard:**
  - Useful for testing connections.

- **Robot Structure:**
  - Utilize aluminum parts from a robotic kit, allowing customization for building a robot with up to 6 degrees of freedom.

- **Power Supply:**
  - When both robots and sensors are connected simultaneously, approximately 12 Amperes and 5 to 6 Volts are needed. The HS-75-5 power supply is recommended due to its cost-effectiveness and suitability for project requirements.

## Connection schematics

These are the schematics for the electronic components used in this project. Up to 12 servomotoros can be controlled by one PCA9685. If more motors are used, more PCA9685 can be added in cascade using the I2C bus and shortcircuiting the 12 bit adderesses.

- **Power Supply** to **ESP32** to **PCA9685**


<img width="327" alt="image-1" src="https://github.com/BakiRhina/Reality-Gap-reduction-TFG/assets/108484177/be053d31-73de-4654-a551-7b66016756b0">


- **ESP32** to **Sensors**
- 

<img width="131" alt="image-2" src="https://github.com/BakiRhina/Reality-Gap-reduction-TFG/assets/108484177/6dfd8d5e-1856-4708-abdd-f6806e34727a">


- **PCA9685** to **MG996R**


<img width="200" alt="image-3" src="https://github.com/BakiRhina/Reality-Gap-reduction-TFG/assets/108484177/3cb3eb04-ec9e-43c6-9fef-c710f58d9a2e">


## Communication

- **ESP32 Communication:**
  - Establishes communication with the main computer (laptop, PC, or Raspberry Pi) through the serial port using UART. A micro-USB cable is sufficient.

- **Sensor Communication:**
  - MPU6050 and PCA9685 communicate with the ESP32 using I2C, customizable to addresses 0x68 and 0x40, respectively.
  - HCSR04 uses GPIO pins for sending distance values.

- **PCA9685 Operation:**
  - Utilizes PWM signals and "ticks" to control motors as needed.

- **Main Computer:**
  - Contains the agent/algorithm processing state data from ESP32.
  - Returns float values representing torque forces, mapped to angles and then to PWM signals for PCA9685.

## Running the Implementation

Once the hardware setup is complete, follow these steps to run the implementation:

1. **Clone the Implementation Repository:**
   - Clone the Implementation repository or use the same directory where the simulation repo was cloned.

2. **Run main.py:**
   - Execute `main.py` in `Implementation/Inference/main.py`.

3. **Provide Port Information:**
   - Enter the port where your ESP32 is connected
   - In Windows usually `COM6`, in Linux its usually `/dev/ttyUSB0`
4. **Specify Motors and Sensors:**
   - Input the number of motors and sensors in your setup. The software is designed to handle any number, but a range of 5 to 10 motors is recommended.

5. **Choose Training Option:**
   - Decide whether to train the model from scratch or use transfer learning (a model previously trained in simulation).
  
6. **Initiate Training:**
   - If training is selected, the training process will commence. Ensure that the robots are securely fixed to the ground to prevent falls.


## Recommandations and notes

- Gazebo simulator is more supported in Linux OS than Windows. If you are not familiar with this OS, we recommend using another simulator such as PyBullet or Isaac Gym as a replacement. Isaac Gym provides Python APIs to communicate with and works directly with the GPU. When escalating this project, the simulation section would have been shifted to this simulator instead of Gazebo.

- Start more simple. Simplicity will be your friend and makes things more organized, clear ,and scalable with less errors in the long run. Choose a simpler robots in a simpler environment. Even if it seems it is toot simple, when the project starts growing it will be more efficient and clear to have simpler modular areas than a confusing and complex system.

- For deeper information about the project, refer to the Thesis in `thesis_docs/Thesis.docx`


