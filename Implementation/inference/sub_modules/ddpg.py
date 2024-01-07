import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sub_modules.rbuffer import ReplayBuffer

#SECTION - POLICY DRL ALGORITHM -


#SECTION - ACTOR NETWORK

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, actor_dropout_p):
        super(Actor, self).__init__()

        self.dropout = nn.Dropout(p=actor_dropout_p)
        self.fc1 = nn.Linear(state_dim, 256)
        print(self.fc1)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.dropout(self.fc1(state)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = F.relu(self.dropout(self.fc3(x)))
        action = torch.tanh(self.fc4(x)) # normalise [-1, 1]
        return action

#!SECTION

#SECTION - CRITIC NETWORK


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, critic_dropout_p):
        super(Critic, self).__init__()

        self.dropout = nn.Dropout(p=critic_dropout_p)
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1) 

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = F.relu(self.dropout(self.fc3(x)))
        value = self.fc4(x) # Estimated Q-Value for a given state-action pair
        return value

#!SECTION



#SECTION - DDPG AGENT


class DDPGAgent:
    def __init__(self, state_dim, action_dim, buffer_size = 10000):

        self.actor_lr = 0.5e-4
        self.critic_lr = 1e-4
        self.discount_factor = 0.95
        self.soft_update_rate = 0.01
        self.actor_dropout_p = 0.5
        self.critic_dropout_p = 0.5
        self.batch_size = 64

        self.replay_bufer = ReplayBuffer(buffer_size)

        self.actor_losses = []
        self.critic_losses = []

        self.state_dim_test = state_dim
        self.action_dim_test = action_dim

        self.actor = Actor(state_dim, action_dim, self.actor_dropout_p)
        self.actor_target = Actor(state_dim, action_dim, self.actor_dropout_p) # Has the same architecture as the main actor network but it's updated slowly --> provides training stability
        self.actor_target.load_state_dict(self.actor.state_dict()) # Get parameters from main actor network and synchronize with acto_target

        self.critic = Critic(state_dim, action_dim, self.critic_dropout_p)
        self.critic_target = Critic(state_dim, action_dim, self.critic_dropout_p)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
    
    #SECTION - Select action

    def select_action(self, state):

        state = torch.FloatTensor(state)
        action = self.actor(state)

        # remove gradients from tensor and convert it to numpy array
        return action.detach().numpy() 
    
    #SECTION - Update 

    def update(self, state, action, reward, next_state, terminal_condition):
        # Add the real-time experience to the replay buffer    
        self.replay_bufer.add((state, 
                               action,
                               reward, 
                               next_state, 
                               terminal_condition)
        )

        # Sample a batch from the replay buffer
        batch_size = self.batch_size
        buffer_batch = self.replay_bufer.sample(batch_size)

        # Unpacking buffer_batch into separate lists for each variable
        buffer_states, buffer_actions, buffer_rewards, buffer_next_states, buffer_terminal_condition = zip(*buffer_batch)

        # Convert lists to NumPy arrays for efficency
        buffer_states = np.array(buffer_states)
        buffer_actions = np.array(buffer_actions)
        buffer_rewards = np.array(buffer_rewards).reshape(-1, 1)
        buffer_next_states = np.array(buffer_next_states)
        buffer_terminal_condition = np.array(buffer_terminal_condition).reshape(-1, 1)

        # Convert lists to PyTorch tensors
        buffer_states = torch.FloatTensor(buffer_states)
        buffer_actions = torch.FloatTensor(buffer_actions)
        buffer_rewards = torch.FloatTensor(buffer_rewards)
        buffer_next_states = torch.FloatTensor(buffer_next_states)
        buffer_terminal_condition = torch.FloatTensor(buffer_terminal_condition)

        # Buffer data
        buffer_values = self.critic(buffer_states, buffer_actions)
        buffer_next_actions = self.actor_target(buffer_next_states)
        buffer_next_values = self.critic_target(buffer_next_states, buffer_next_actions.detach())

        # BELLMAN EQUATION
        buffer_target_values = buffer_rewards + self.discount_factor * buffer_next_values * (1 - buffer_terminal_condition)
        
        # Critic loss for buffer data
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
        self.soft_update(self.actor, self.actor_target, self.soft_update_rate)
        self.soft_update(self.critic, self.critic_target, self.soft_update_rate)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * local_param.data)

#!SECTION
#!SECTION