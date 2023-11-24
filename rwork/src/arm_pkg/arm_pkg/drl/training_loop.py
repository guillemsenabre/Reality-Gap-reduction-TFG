import rclpy
import subprocess
import os
from rclpy.node import Node
from ros_gz_interfaces.msg import Float32Array
from std_msgs.msg import Float32


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim






#SECTION - POLICY DRL ALGORITHM
#SECTION - ACTOR NETWORK


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) # normalise [-1, 1]
        return action

#!SECTION

#SECTION - CRITIC NETWORK


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1) 

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x) # Estimated Q-Value for a given state-action pair
        return value

#!SECTION



#SECTION - DDPG AGENT


class DDPGAgent:
    def __init__(self, state_dim, action_dim):

        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim) # Has the same architecture as the main actor network but it's updated slowly --> provides training stability
        self.actor_target.load_state_dict(self.actor.state_dict()) # Get parameters from main actor network and synchronize with acto_target

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3) # Adam optimizer To update the weights during training
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
    
    #SECTION - Select action

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action = self.actor(state)

        # remove gradients from tensor and convert it to numpy array
        return action.detach().numpy() 
    
    #SECTION - Update 

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor([reward])
        next_state = torch.FloatTensor(next_state)

        # Critic loss
        value = self.critic(state, action)
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action.detach())
        target_value = reward + 0.99 * next_value * (1 - done)
        critic_loss = F.mse_loss(value, target_value)

        # Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update target networks with soft updates
        self.soft_update(self.actor, self.actor_target, 0.01)
        self.soft_update(self.critic, self.critic_target, 0.01)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * local_param.data)

#!SECTION
#!SECTION


#SECTION - RESET CLASS




class Reset(Node):
    def __init__(self):
        super().__init__('joint_torque_controller')

    def reset(self):
        self.get_logger().info("Resetting simulation...")
        self.kill_gazebo_process()
        self.run_gazebo()
        self.unpause()

    def kill_gazebo_process(self):
        # Find and kill the Gazebo process
        try:
            subprocess.run(['pkill', '-f', 'gazebo'], check=True)
        except subprocess.CalledProcessError:
            self.get_logger().warning("Failed to kill Gazebo process.")

    def run_gazebo(self):
        self.get_logger().info("starting gazebo simulator...")
        home_directory = os.path.expanduser("~")
        sdf_file_path = os.path.join(home_directory, 'tfg', 'rwork', 'src', 'sdf_files', 'full_env_simpler.sdf')

        try:
            subprocess.Popen(['ign', 'gazebo', sdf_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            self.get_logger().error("Failed to start Gazebo process.")

    def unpause(self):
        # Use subprocess to execute the ros2 service call command
        command = 'ros2 service call /world/full_env_simpler/control ros_gz_interfaces/srv/ControlWorld "{world_control: {pause: false}}"'
        try:
            subprocess.run(command, shell=True, check=True)
            self.get_logger().info("Simulation unpaused successfully.")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Failed to unpause simulation. Error: {e}")

#!SECTION


#SECTION - RECEIVE AND PROCESS DATA


class RosData(Node):
    def __init__(self):    
        super().__init__('ros_data')
        
        self.state = np.array([])
        self.reward_value = 0.0

        # Subsribing to topics data

        self.state_subscription = self.create_subscription(
            Float32Array,
            'packed/state/data',
            self.process_state_data,
            1
        )

        self.reward_subscription = self.create_subscription(
            Float32,
            'reward/data',
            self.process_reward_data,
            1
        )

        # Publishers for the joints

        self.joint_publishers = []
        self.joint_names = [
                            'joint0_1', 
                            'joint1_1', 
                            'joint2_1',
                            'joint3_1',
                            'joint0_2', 
                            'joint1_2', 
                            'joint2_2',
                            'joint3_2',
                            ]
        
        for joint_name in self.joint_names:
            publisher = self.create_publisher(Float32, f'/arm/{joint_name}/wrench', 1)
            self.joint_publishers.append(publisher)

        # Initialize the DDPG agent
        state_dim = 12  
        action_dim = 8
        self.agent = DDPGAgent(state_dim, action_dim)

    def process_state_data(self, msg: Float32Array):

        data = msg.data
        
        # Extract gripper and object positions

        gripper_1_pos = np.array(data[4:7])

        gripper_2_pos = np.array(data[15:18])
        object_pos = np.array(data[22:25])

        object_1_pos = np.array([object_pos[0] - 0.125, object_pos[1], object_pos[2]])
        object_2_pos = np.array([object_pos[0] + 0.125, object_pos[1], object_pos[2]])

        self.state = np.concatenate([gripper_1_pos, gripper_2_pos, object_1_pos, object_2_pos])

    def process_reward_data(self, msg: Float32):
        self.reward_value = msg.data

    
    def move_joints(self, action):

        for idx, publisher in enumerate(self.joint_publishers):
            msg = Float32()
            msg.data = float(action[idx])

            publisher.publish(msg)

#!SECTION



#SECTION - TRAINING LOOP



def main(args=None):
    rclpy.init(args=args)
    
    ros_data = RosData()
    reset = Reset()

    num_episodes = 100
    max_steps = 1000
    for episode in range(num_episodes):

        print(f'Running poch: {episode}')
        # Reset environment and get initial state
        reset.reset()
        
        # Waiting for the first state message to be received
        while not ros_data.state.any():
            print("Waiting for state data ...")
            rclpy.spin_once(ros_data)

        print("State data is here!!")
        state = ros_data.state


        for step in range(max_steps):
            # Select action from the agent's policy
            action = ros_data.agent.select_action(state)

            print(action)

            # Execute actions
            ros_data.move_joints(action)

            # observe next state and reward
            next_state = ros_data.state
            reward = ros_data.reward_value

            #FIXME - Remove env and figure what done is

            next_state, reward, done, _ = env.step(action)

            # Update agent
            ros_data.agent.update(state, action, reward, next_state, done)

            if done:
                break


    rclpy.spin(ros_data)

    ros_data.destroy_node()

    rclpy.shutdown()

#!SECTION

if __name__ == '__main__':
    main()