import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import Float32Array
from std_msgs.msg import Float32


################ ACTOR NETWORKS ####################


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


################ CRITIC NETWORKS ###################


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


################## DDPG AGENT ######################
class RosData(Node):
    def __init__(self, state_dim, action_dim):    
        super().__init__('ros_data')

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

    def process_state_data(self, msg: Float32Array):

        data = msg.data

        # Extract gripper and object positions
        gripper_1_pos = data[4:7]
        gripper_2_pos = data[15:18]
        object_pos = data[22:25]

        self.state = gripper_1_pos + gripper_2_pos + object_pos

    def process_reward_data(self, msg: Float32):
        self.reward_value = msg.data

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
    
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        action = self.actor(state)
        return action.detach().numpy()

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







# Initialize the DDPG agent
state_dim = 18  # Modify based on your state space
action_dim = 6  # Modify based on your action space
agent = DDPGAgent(state_dim, action_dim)

# Training loop
for episode in range(num_episodes):
    # Reset environment and get initial state
    state = env.reset()

    for step in range(max_steps):
        # Select action from the agent's policy
        action = agent.select_action(state)

        # Execute action and observe next state and reward
        next_state, reward, done, _ = env.step(action)

        # Update agent
        agent.update(state, action, reward, next_state, done)

        # Move to the next state
        state = next_state

        if done:
            break








def main(args=None):
    rclpy.init(args=args)
    ros_data = RosData()
    rclpy.spin(ros_data)
    ros_data.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()