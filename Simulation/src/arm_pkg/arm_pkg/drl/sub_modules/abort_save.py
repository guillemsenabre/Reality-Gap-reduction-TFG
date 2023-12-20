from reward import Reward
from states import States
from ..configuration import Configuration

class AbortOrSave():
    def __init__(self):
        self.reward = Reward()
        self.states = States()
        self.config = Configuration()
        self.angles = self.states.read_sensor_data[:10]

        self.velocity = []

    def reached_goal(self):

        average_angle = sum(self.angles)/len(self.angles)
        count = 0

        while count <= self.config.number_of_values:
            # Append average values until a certain threshold,
            #Run out of brain today
            self.angles.append


            count += 1
            average_angle = 0
        
        # Velocity ~ 0
        # Distance < threshold

    def local_minimum(self):
        pass