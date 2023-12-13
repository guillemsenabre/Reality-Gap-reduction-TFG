class Configuration:
    def __init__(self):
        
        # DDPG AGENT VARIABLES #

        # TRAINING VARIABLES #

        self.margin_value = 0.01
        self.maximum_accumulative_reward = 100
        
        self.state_dim = 12
        self.action_dim = 8