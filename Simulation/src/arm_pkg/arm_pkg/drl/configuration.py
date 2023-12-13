class Configuration:
    def __init__(self):

        #SECTION ROS DATA VARIABLES #
        
        #SECTION DDPG AGENT VARIABLES #

        self.margin_value = 0.01
        self.maximum_accumulative_reward = 100

        self.state_dim = 12
        self.action_dim = 8

        self.joint_names = [
                            'joint0_1', 
                            'joint1_1', 
                            'joint2_1',
                            'joint3_1',
                            'joint0_2', 
                            'joint1_2', 
                            'joint2_2',
                            'joint3_2',
                            ]

        #SECTION TRAINING VARIABLES #

        self.num_episodes = 100

        self.reward_count_to_save_model = 50

        