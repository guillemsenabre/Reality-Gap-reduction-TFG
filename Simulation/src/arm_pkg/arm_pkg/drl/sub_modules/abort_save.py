from reward import Reward
from states import States
from ..configuration import Configuration

class AbortOrSave():
    def __init__(self):
        self.reward = Reward()
        self.states = States()
        self.config = Configuration()
        self.angles = self.states.read_sensor_data[:10]
        self.velocity_history = []

    def _null_velocity(self):
        # Calculate the average angle
        average_angle = sum(self.angles) / 10
        
        # Append the average angle to the history
        self.velocity_history.append(average_angle)

        # Check if the history has reached the desired length
        if len(self.velocity_history) == self.config.number_of_values:
            # Check if the first, middle, and last items are the same
            if self.velocity_history[0] == self.velocity_history[len(self.velocity_history)//2] == self.velocity_history[-1]:
                return True

            # Reset the history for the next round
            self.velocity_history = []

        return False 

    def _local_minimum(self):
        pass

    def terminal_condition(self):
        if self._null_velocity() or self._local_minimum():
            return True
        else:
            return False