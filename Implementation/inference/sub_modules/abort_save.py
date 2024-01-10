import time

class AbortOrSave():
    def __init__(self):
        print("Initializing Terminal condition...")

        self.number_of_velocity_values = 20
        self.number_of_reward_values = 30
        self.velocity_history = []
        self.reward_history = []
        time.sleep(0.1)

    def _null_velocity(self, angles):
        # Calculate the average angle
        average_angle = sum(angles) / 10
        
        # Append the average angle to the history
        self.velocity_history.append(average_angle)

        # Check if the history has reached the desired length
        if len(self.velocity_history) == self.number_of_velocity_values:
            # Check if the first, middle, and last items are the same
            if self.velocity_history[0] == self.velocity_history[len(self.velocity_history)//2] == self.velocity_history[-1]:
                return True

            # Reset the history for the next round
            self.velocity_history = []

        return False 

    def _local_minimum(self, reward_value):
        
        '''
        This method stores rewards in a list until it's filled (reaches number_of_reward_values threshold). When
        this happens it checks if the reward values have changed, if not, it assumes that it has reached a local minima
        or it cannot improve more. A bigger threshold means giving more time to find a way out of the local minima 
        although it may not be worth it.
        '''

        # Append reward values to a list
        self.reward_history.append(reward_value)

        # When the reward list is full, check if the values are still changing
        if len(self.reward_history) == self.number_of_reward_values:
            if self.reward_history[0] == self.reward_history[len(self.reward_history)//2] == self.reward_history[-1]:

                # If they are indeed not changing, terminate the episode
                return True
            
            # If not, reset the list and continue
            self.reward_history = []

        return False
    
    def _unwanted_joint_angles(self):
        pass

    def terminal_condition(self, angles, reward_value):
        if self._null_velocity(angles) or self._local_minimum(reward_value):
            return True
        else:
            return False