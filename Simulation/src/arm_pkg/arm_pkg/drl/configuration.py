class Configuration:
    def __init__(self):

        #SECTION DDPG AGENT VARIABLES #

        self.actor_lr = 0.5e-4
        self.critic_lr = 1e-4
        self.discount_factor = 0.95
        self.soft_update_rate = 0.01
        self.actor_dropout_p = 0.5
        self.critic_dropout_p = 0.5
        self.batch_size = 64

        #SECTION ROS DATA VARIABLES #

        self.reward_init_value = 0.0

        self.margin_value = 0.01
        self.maximum_accumulative_reward = 300

        self.action_dim = int(input("Select action dimensions"))
        self.state_dim = int(input("Select state dim"))

        self.joint_names = [
            'joint0_1', 'joint1_1', 'joint2_1', 'joint3_1',
            'joint0_2', 'joint1_2', 'joint2_2', 'joint3_2',
        ]

        self.after_moving_joints_time = 0.01

        #SECTION TRAINING VARIABLES #

        self.num_episodes = 10
        self.reward_count_to_save_model = 50
        self.avg_reward_threshold_to_save_model = 0.5

        #SECTION RESET VARIABLES #

        self.deviation_threshold = 1.2
        self.after_kill_time = 3
        self.after_run_time = 7

        #SECTION REWARD VARIABLES #

        self.deviation_multiplier = 2

        #SECTION INFERENCE VARIABLES #

        self.model_name = "ddpg_model.pth"