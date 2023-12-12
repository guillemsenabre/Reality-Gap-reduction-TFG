import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # First fully connected layer from state dimension to 256 units
        self.fc1 = nn.Linear(state_dim, 256)
        # Second fully connected layer with 256 units
        self.fc2 = nn.Linear(256, 256)
        # Output layer for mean of action distribution, size equals action dimension
        self.mean = nn.Linear(256, action_dim)
        # Output layer for standard deviation of action distribution, size equals action dimension
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        # Forward pass through the first layer with ReLU activation
        x = F.relu(self.fc1(state))
        # Forward pass through the second layer with ReLU activation
        x = F.relu(self.fc2(x))
        # Compute mean of the action distribution
        mean = self.mean(x)
        # Compute log standard deviation of the action distribution
        log_std = self.log_std(x)
        # Standard deviation must be positive; use exponential function
        std = torch.exp(log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Fully connected layer from state and action dimensions combined to 256 units
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        # Second fully connected layer with 256 units
        self.fc2 = nn.Linear(256, 256)
        # Output layer for single Q-value
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        # Concatenate state and action as input
        x = torch.cat([state, action], 1)
        # Forward pass through the first layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Forward pass through the second layer with ReLU activation
        x = F.relu(self.fc2(x))
        # Compute Q-value
        q_value = self.fc3(x)
        return q_value

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # Maximum size of the buffer
        self.buffer = []  # Internal storage for experiences
        self.position = 0  # Current position in the buffer

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # Expand buffer if not at full capacity
        # Store experience tuple at the current position
        self.buffer[self.position] = (state, action, reward, next_state, done)
        # Update position with wrap-around
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Randomly sample a batch of experiences from the buffer
        batch = random.sample(self.buffer, batch_size)
        # Unzip the batch into separate components
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
