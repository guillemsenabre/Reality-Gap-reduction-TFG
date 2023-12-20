from states import States

class Reward():
    def __init__(self):
        self.states = States()
        self.distanceRB1 = self.states.read_sensor_data[10]
        self.distanceRB2 = self.states.read_sensor_data[11]
        self.object_orientation = self.states.read_sensor_data[12:14]
    
    def distance_reward(self):
        pass

    def orientation_reward(self):
        pass

    def reward(self):
        pass