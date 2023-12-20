from states import States
from ..configuration import Configuration

class Reward():
    def __init__(self):
        self.states = States()
        self.config = Configuration()
        self.angles = self.states.read_sensor_data[:10]
        self.distanceRB1 = self.states.read_sensor_data[10]
        self.distanceRB2 = self.states.read_sensor_data[11]
        self.object_orientation = self.states.read_sensor_data[12:14]

        self.scaling_factor_velocity = self.config.scaling_factor_velocity
    
    def _distance_reward(self):

        # The smaller the distance the greater the reward (using f(x)=1/x, x>0)
        reward1 = 1/self.distanceRB1 
        reward2 = 1/self.distanceRB2

        distance_value = reward1 + reward2

        return distance_value

    def _drop_velocity_reward(self):

            # Calculate the change in joint angles
        delta_angles = [current - prev for current, prev in zip(self.angles, self.prev_angles)]

        # Calculate the sum of absolute changes as a measure of velocity
        velocity = sum(map(abs, delta_angles))

        # Apply a reward function (e.g., inverse of velocity)
        scaled_velocity_reward = velocity / (self.distanceRB1 + 
                                             self.distanceRB2 + 
                                             self.scaling_factor_velocity)

        
        # This function gets higher values as the velocity of the joints (
        #or, what is the same, the how much the angle of a joint has changed per time)
        #decreases when the distanceRB1 and distanceRB2 also decreases, to avoid 
        #coliding with the object at high speed and, ideally, to stop the robots when 
        #the distances are 0.
        pass

    def reward(self):
        pass