from sub_modules.reward import Reward
from sub_modules.states import States
from sub_modules.configuration import Configuration

class AbortOrSave():
    def __init__(self):

        print("Initializing Abort or Save yeee")
        self.reward = Reward()
        self.states = States()
        self.config = Configuration()
        self.angles = []
        self.velocity_history = []
        self.reward_history = []

    def _null_velocity(self):
        # Calculate the average angle
        self.angles = self.states.read_sensor_data[:10]
        average_angle = sum(self.angles) / 10
        
        # Append the average angle to the history
        self.velocity_history.append(average_angle)

        # Check if the history has reached the desired length
        if len(self.velocity_history) == self.config.number_of_velocity_values:
            # Check if the first, middle, and last items are the same
            if self.velocity_history[0] == self.velocity_history[len(self.velocity_history)//2] == self.velocity_history[-1]:
                return True

            # Reset the history for the next round
            self.velocity_history = []

        return False 

    def _local_minimum(self):
        
        self.reward_history.append(self.reward.reward)

        if len(self.reward_history) == self.config.number_of_reward_values:
            if self.reward_history[0] == self.reward_history[len(self.reward_history)//2] == self.reward_history[-1]:
                return True
            
            self.reward_history = []

        return False
    
    def _unwanted_joint_angles(self):
        pass

    def terminal_condition(self):
        if self._null_velocity() or self._local_minimum():
            return True
        else:
            return False