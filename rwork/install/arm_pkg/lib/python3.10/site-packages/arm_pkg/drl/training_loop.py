import rclpy
import subprocess
import os
from rclpy.node import Node
from ros_gz_interfaces.msg import Float32Array
from std_msgs.msg import Float32


##########################################################
###################### RESET CLASS #######################
##########################################################

class Reset(Node):
    def __init__(self):
        super().__init__('joint_torque_controller')

    def reset(self):
        self.get_logger().info("Resetting simulation...")
        self.kill_gazebo_process()
        self.run_gazebo()
        self.unpause()

    def kill_gazebo_process(self):
        # Find and kill the Gazebo process
        try:
            subprocess.run(['pkill', '-f', 'gazebo'], check=True)
        except subprocess.CalledProcessError:
            self.get_logger().warning("Failed to kill Gazebo process.")

    def run_gazebo(self):
        self.get_logger().info("starting gazebo simulator...")
        home_directory = os.path.expanduser("~")
        sdf_file_path = os.path.join(home_directory, 'tfg', 'rwork', 'src', 'sdf_files', 'full_env_simpler.sdf')

        try:
            subprocess.Popen(['ign', 'gazebo', sdf_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            self.get_logger().error("Failed to start Gazebo process.")

    def unpause(self):
        # Use subprocess to execute the ros2 service call command
        command = 'ros2 service call /world/full_env_simpler/control ros_gz_interfaces/srv/ControlWorld "{world_control: {pause: false}}"'
        try:
            subprocess.run(command, shell=True, check=True)
            self.get_logger().info("Simulation unpaused successfully.")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Failed to unpause simulation. Error: {e}")



##########################################################
################## ROS DATA PROCESSING ###################
##########################################################


class RosData(Node):
    def __init__(self):    
        super().__init__('ros_data')

        # Reset class instance & run gazebo 

        reset = Reset()
        reset.reset

        # Subsribing to topics data

        self.state_subscription = self.create_subscription(
            Float32Array,
            'packed/state/data',
            self.process_state_data,
            1
        )

        self.reward_subscription = self.create_subscription(
            Float32,
            'reward/data',
            self.process_reward_data,
            1
        )

        # Publishers for the joints

        self.joint_publishers = []
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
        
        for joint_name in self.joint_names:
            publisher = self.create_publisher(Float32, f'/arm/{joint_name}/wrench', 1)
            self.joint_publishers.append(publisher)

        # Initialize the DDPG agent
        state_dim = 12  
        action_dim = 8
        self.agent = DDPGAgent(state_dim, action_dim)

    def process_state_data(self, msg: Float32Array):

        data = msg.data

        # Extract gripper and object positions
        gripper_1_pos = data[4:7]
        gripper_2_pos = data[15:18]
        object_pos = data[22:25]

        object_1_pos = [object_pos[0] - 0.125, object_pos[1], object_pos[2]]
        object_2_pos = [object_pos[0] + 0.125, object_pos[1], object_pos[2]]

        self.state = gripper_1_pos + gripper_2_pos + object_1_pos + object_2_pos

    def process_reward_data(self, msg: Float32):
        self.reward_value = msg.data

    
    def move_joints(self, action):

        for idx, publisher in enumerate(self.joint_publishers):
            msg = Float32()
            msg.data = float(action[idx])

            publisher.publish(msg)



def main(self):
    rclpy.init()
    ros_data = RosData()
    reset = Reset()

    # Training loop

    num_episodes = 100
    max_steps = 1000
    for episode in range(num_episodes):

        self.get_logger().info(f'Running poch: {episode}')
        # Reset environment and get initial state
        reset.reset
        state = ros_data.process_state_data

        for step in range(max_steps):
            # Select action from the agent's policy
            action = ros_data.agent.select_action(state)

            # Execute actions
            ros_data.move_joints(action)

            # observe next state and reward
            next_state = ros_data.process_state_data
            reward = ros_data.process_reward_data

            next_state, reward, done, _ = env.step(action)

            # Update agent
            ros_data.agent.update(state, action, reward, next_state, done)

            if done:
                break


    rclpy.spin(ros_data)

    ros_data.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()