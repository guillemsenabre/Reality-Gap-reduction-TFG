import rclpy
import os
import time
from rclpy.node import Node
from ros_gz_interfaces.msg import Float32Array
from std_msgs.msg import Float64, Float32

from .ddpg import DDPGAgent
from .reset import Reset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


#TODO - Tune hyperparameters (lr)
#TODO - Understanding losses behaviour
#TODO - Save model 


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