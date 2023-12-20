from states import States
from ..configuration import Configuration

class Reward():
    def __init__(self):
        self.states = States()
        self.config = Configuration()
        self.distanceRB1 = self.states.read_sensor_data[10]
        self.distanceRB2 = self.states.read_sensor_data[11]
        self.object_orientation = self.states.read_sensor_data[12:14]
    
    def _distance_reward(self):

        # The smaller the distance the greater the reward (using f(x)=1/x, x>0)
        reward1 = 1/self.distanceRB1 
        reward2 = 1/self.distanceRB2

        distance_value = reward1 + reward2

        return distance_value

    def _drop_velocity_reward(self):

        # This function gets higher values as the velocity of the joints (
        #or, what is the same, the how much the angle of a joint has changed per time)
        #decreases when the distanceRB1 and distanceRB2 also decreases, to avoid 
        #coliding with the object at high speed and, ideally, to stop the robots when 
        #the distances are 0.
        pass

    def reward(self):
        pass