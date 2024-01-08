import time
import random

class Reward():
    def __init__(self):
        print("Initializing Reward module skiii")
        self.angles = []

        # The bigger, the more important will be the reward
        # To not affect the function, set to 1
        # Don't set to 0, may divide by 0 at some point

        self.scaling_factor_velocity_1 = 1
        self.scaling_factor_velocity_2 = 1
        self.scaling_distance_reward = 1

        time.sleep(0.3)

    def _distance_reward(self, state, number_motors):
        self.angles = state[:number_motors] # LIST
        self.distanceRB1 = state[number_motors] + random.uniform(0.2, 0.3) # Bias is added (FLOAT)
        self.distanceRB2 = state[number_motors + 1] + random.uniform(0.2, 0.3) #EX: with 10 motors, this would be position 10 (index) and motors would be 0-9
        self.object_orientation = state[number_motors + 2] # List of floats

        # The smaller the distance the greater the reward (using f(x)=1/x, x>0)
        reward1 = 1/(self.distanceRB1 + 0.1) # Some BIAS are added in case of null values
        reward2 = 1/(self.distanceRB2 + 0.1)

        distance_value = reward1 + reward2

        return distance_value

    def _drop_velocity_reward(self, prev_angles):

            # Calculate the change in joint angles
        delta_angles = [current - prev for current, prev in zip(self.angles, prev_angles)]

        # Calculate the sum of absolute changes as a measure of velocity
        velocity = sum(map(abs, delta_angles))

        # Apply a reward function (e.g., inverse of velocity)
        scaled_velocity_reward = velocity / (self.distanceRB1 + 
                                             self.distanceRB2 + 
                                             self.scaling_factor_velocity_1)

        
        # This function gets higher values as the velocity of the joints (
        #or, what is the same, the how much the angle of a joint has changed per time)
        #decreases when the distanceRB1 and distanceRB2 also decreases, to avoid 
        #coliding with the object at high speed and, ideally, to stop the robots when 
        #the distances are 0.
        
        return scaled_velocity_reward

    def reward(self, prev_angles, states, number_motors=10):
        total_reward = (self._distance_reward(states, number_motors) * self.scaling_distance_reward +
                        self._drop_velocity_reward(prev_angles) * self.scaling_factor_velocity_2)
        return total_reward