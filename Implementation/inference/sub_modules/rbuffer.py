import random

class ReplayBuffer:
    # The bigger is the buffer_size, the more data it can store --> more learning, more computational expensive
    def __init__(self, buffer_size):
        # Maximum number of experiences to store in the buffer
        self.buffer_size = buffer_size  
        # List to store experiences
        self.buffer = [] 

    # Method to add a new experience to the replay buffer
    def add(self, experience):
        # Check if the buffer is already full
        if len(self.buffer) >= self.buffer_size:
            # If full, remove the oldest experience (FIFO) to make space for the new one
            self.buffer.pop(0)
        # Add the new experience to the end of the buffer
        self.buffer.append(experience)
    
    # Method to sample a batch of experiences from the replay buffer
    def sample(self, batch_size):
        # Use random.sample to select a random subset of experiences from the buffer
        # If the buffer size is less than the batch size, return the entire buffer
        return random.sample(self.buffer, batch_size) if len(self.buffer) >= batch_size else self.buffer
