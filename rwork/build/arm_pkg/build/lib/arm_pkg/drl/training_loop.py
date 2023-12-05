import rclpy
import subprocess
import os
import time
import random
import psutil
from rclpy.node import Node
from ros_gz_interfaces.msg import Float32Array
from std_msgs.msg import Float64, Float32


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


#TODO - Incorporate buffer data
#TODO - Tune hyperparameters (lr)
#TODO - Save model 


#SECTION - POLICY DRL ALGORITHM
#SECTION - ACTOR NETWORK

    #FIXME - How to save the model? When?

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
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x) # Estimated Q-Value for a given state-action pair
        return value

#!SECTION



#SECTION - DDPG AGENT


class DDPGAgent:
    def __init__(self, state_dim, action_dim, buffer_size = 10000):

        self.actor_losses = []
        self.critic_losses = []

        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim) # Has the same architecture as the main actor network but it's updated slowly --> provides training stability
        self.actor_target.load_state_dict(self.actor.state_dict()) # Get parameters from main actor network and synchronize with acto_target

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3) # Adam optimizer To update the weights during training
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_bufer = ReplayBuffer(buffer_size)
    
    #SECTION - Select action

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action = self.actor(state)

        # remove gradients from tensor and convert it to numpy array
        return action.detach().numpy() 
    
    #SECTION - Update 

    def update(self, state, action, reward, next_state, terminal_condition, batch_size = 64):
        # Add the real-time experience to the replay buffer    
        self.replay_bufer.add((state, action, reward, next_state, terminal_condition))
        
        # Sample a batch from the replay buffer
        buffer_batch = self.replay_bufer.sample(batch_size)

        # Unpacking buffer_batch into separate lists for each variable
        buffer_states, buffer_actions, buffer_rewards, buffer_next_states, buffer_terminal_condition = zip(*buffer_batch)

        # Convert lists to PyTorch tensors
        buffer_states = torch.FloatTensor(buffer_states)
        buffer_actions = torch.FloatTensor(buffer_actions)
        buffer_rewards = torch.FloatTensor(buffer_rewards).view(-1, 1)
        buffer_next_states = torch.FloatTensor(buffer_next_states)
        buffer_terminal_condition = torch.FloatTensor(buffer_terminal_condition).view(-1, 1)

        # Critic loss for buffer data
        buffer_values = self.critic(buffer_states, buffer_actions)
        buffer_next_actions = self.actor_target(buffer_next_states)
        buffer_next_values = self.critic_target(buffer_next_states, buffer_next_actions.detach())
        buffer_target_values = buffer_rewards + 0.99 * buffer_next_values * (1 - buffer_terminal_condition)
        critic_loss = F.mse_loss(buffer_values, buffer_target_values)

        # Actor loss for buffer data
        actor_loss = -self.critic(buffer_states, self.actor(buffer_states)).mean()

        # Append losses to the history
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())

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

#SECTION - Replay buffer class

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, experience):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size) if len(self.buffer) >= batch_size else self.buffer

#!SECTION



#SECTION - RESET CLASS


class Reset(Node):
    def __init__(self):
        super().__init__('reset')

    def reset(self):
        self.get_logger().info("Resetting simulation...")
        self.kill_gazebo_process()
        time.sleep(3)
        self.run_gazebo()
        time.sleep(7)
        self.unpause()

    def kill_gazebo_process(self):
        # Find and kill the Gazebo process
        try:
            subprocess.run(['pkill', '-f', 'gazebo'], check=True)
        except subprocess.CalledProcessError:
            self.get_logger().warning("Failed to kill Gazebo process.")

    def run_gazebo(self):
        self.get_logger().info("Check if gazebo is dead...")
        print(not self.is_gazebo_running())

        if not self.is_gazebo_running():
            self.get_logger().info("starting gazebo simulator...")
            home_directory = os.path.expanduser("~")
            sdf_file_path = os.path.join(home_directory, 'tfg', 'rwork', 'src', 'sdf_files', 'full_env_simpler.sdf')

            try:
                subprocess.Popen(['ign', 'gazebo', sdf_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                self.get_logger().error("Failed to start Gazebo process.")
        
        else:
            self.get_logger().error("Gazebo is still running...")
            time.sleep(1)
            self.get_logger().error("Trying again...")
            self.reset()

    def unpause(self):
        # Use subprocess to execute the ros2 service call command
        command = 'ros2 service call /world/full_env_simpler/control ros_gz_interfaces/srv/ControlWorld "{world_control: {pause: false}}"'
        try:
            subprocess.run(command, shell=True, check=True)
            self.get_logger().info("Simulation unpaused successfully.")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Failed to unpause simulation. Error: {e}")

    def is_gazebo_running(self):
        for process in psutil.process_iter(['pid', 'name']):
            if 'gazebo' in process.info['name']:
                return True
        return False

#!SECTION


#SECTION - RECEIVE AND PROCESS DATA


class RosData(Node):
    def __init__(self):    
        super().__init__('ros_data')

        self.get_logger().info(f'Starting training...')

        
        self.state = np.array([])
        self.reward_list = []
        self.reward_value = 0.0
        self.margin_value = 0.01
        
        self.maximum_accumulative_reward = 100

        state_dim = 12  
        action_dim = 8
        self.agent = DDPGAgent(state_dim, action_dim)


        # Subscribing to topics data
        self.state_subscription = self.create_subscription(
            Float32Array,
            'packed/state/data',
            self.process_state_data,
            10
        )

        self.reward_subscription = self.create_subscription(
            Float32,
            'reward/data',
            self.process_reward_data,
            10
        )


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
            publisher = self.create_publisher(Float64, f'/arm/{joint_name}/wrench', 1)
            self.joint_publishers.append(publisher)

    def process_state_data(self, msg: Float32Array):
        data = msg.data
        
        # Extract gripper and object positions

        gripper_1_pos = np.array(data[4:7])

        gripper_2_pos = np.array(data[15:18])
        object_pos = np.array(data[22:25])

        #self.get_logger().info(f"OBJECT POS: {object_pos}")

        object_1_pos = np.array([object_pos[0] - 0.125, object_pos[1], object_pos[2]])
        object_2_pos = np.array([object_pos[0] + 0.125, object_pos[1], object_pos[2]])

        self.state = np.concatenate([gripper_1_pos, gripper_2_pos, object_1_pos, object_2_pos])

    def process_reward_data(self, msg: Float64):
        self.reward_value = msg.data

    def terminal_condition(self):
        self.reward_list.append(self.reward_value)

        if self.maximum_accumulative_reward == len(self.reward_list):
            margin = abs(self.margin_value * self.reward_list[0])
            difference = abs(self.reward_list[0] - self.reward_list[-1]) 
            self.reward_list = []

            if difference <= margin:
                print(f'Reached local minimum!')
                print(f'Difference: {round(difference, 4)}')
                print(f'Margin: {round(margin, 4)}')
                return True
                        
        elif (self.state[11] or self.state[8]) < 1.2:
            print(f'Oops, object dropped')
            return True
        
        else:
            return False
        
        return False
            
    def move_joints(self, action):
        for idx, publisher in enumerate(self.joint_publishers):
            msg = Float64()
            msg.data = float(action[idx])
            publisher.publish(msg)
            #self.get_logger().info(f'Joint {idx} action: {action[idx]}, torque: {msg.data}')

        time.sleep(0.01)

#!SECTION

#SECTION - Plotting

def plot_results(episode_rewards, actor_losses, critic_losses):
    # Flatten the list of lists into a single list
    flattened_rewards = [reward for episode in episode_rewards for reward in episode]

    # Plot episode rewards
    plt.figure(figsize=(12, 6))

    plt.plot(flattened_rewards, label='Episode Total Reward')
    plt.title('Episode Rewards')
    plt.xlabel('Step')
    plt.ylabel('Total Reward')
    plt.legend()

    # Plot actor and critic losses
    plt.plot(actor_losses, label='Actor Loss', alpha=0.7)
    plt.plot(critic_losses, label='Critic Loss', alpha=0.7)
    plt.title('Actor and Critic Losses')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

#!SECTION

#SECTION - TRAINING LOOP -



def main(args=None):
    rclpy.init(args=args)
    ros_data = RosData()
    reset = Reset()
    num_episodes = 100

    episode_rewards = []

    for episode in range(num_episodes):

        print(f'Running poch: {episode}')

        reset.reset()
        
        # Waiting for the first state message to be received
        while not ros_data.state.any():
            print("Waiting for state data ...")
            rclpy.spin_once(ros_data)

        print("Training!")

        episode_reward_list = []

        while True:
            state = ros_data.state
            action = ros_data.agent.select_action(state)
            ros_data.move_joints(action)
            next_state = ros_data.state
            reward = ros_data.reward_value

            terminal_condition = ros_data.terminal_condition()

            # Update agent
            ros_data.agent.update(state, action, reward, next_state, terminal_condition)

            # Collect data for the current episode
            episode_reward_list.append(reward)

            rclpy.spin_once(ros_data)

            if terminal_condition:
                print(f'Terminal condition reached!')
                break

        # Store episode data for plotting
        episode_rewards.append(episode_reward_list)

        # Plot at the end of each episode
        plot_results(episode_rewards, ros_data.agent.actor_losses, ros_data.agent.critic_losses)


    ros_data.destroy_node()
    rclpy.shutdown()

#!SECTION
#!SECTION

if __name__ == '__main__':
    main() 