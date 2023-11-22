from email import iterators
import rclpy
import math
import subprocess
import os

from rclpy.node import Node
from std_msgs.msg import Float64
from ros_gz_interfaces.srv import ControlWorld

class JointTorqueController(Node):
    def __init__(self):
        super().__init__('joint_torque_controller')

        self.control_world_client = self.create_client(ControlWorld, '/world/full_env/control')
        
        self.joint_publishers = []
        self.joint_names = [
                            'joint0_1', 
                            'joint1_1', 
                            'joint2_1',
                            'joint3_1',
                            'right_finger_1_joint',
                            'left_finger_1_joint',
                            'joint0_2', 
                            'joint1_2', 
                            'joint2_2',
                            'joint3_2',
                            'right_finger_2_joint',
                            'left_finger_2_joint',
                            ]
        
        for joint_name in self.joint_names:
            publisher = self.create_publisher(Float64, f'/arm/{joint_name}/wrench', 1)
            self.joint_publishers.append(publisher)

        self.move_timer = self.create_timer(1, self.move_joints)
        self.reset_timer = self.create_timer(10000, self.reset)

        

        self.angle = 0
        
        self.iterations_per_epoch = 30
        self.current_iteration = 0

    def move_joints(self):

        # keep the angle between 0 and 2π
        self.angle = (self.angle + 10) % (2 * math.pi) 
        
        # Test multipliers for each joint 
        joint_multipliers_test = [
                                  8.5, 9, 3, 2, 1, 1,
                                  6, 6, 6, 6, 6, 6
                                  ]

        for idx, publisher in enumerate(self.joint_publishers):
            msg = Float64()
            msg.data = joint_multipliers_test[idx] * math.sin(self.angle)
            publisher.publish(msg)
            self.get_logger().info(f'Joint {idx} torque: "{msg.data}"')

        self.current_iteration += 1

        self.get_logger().info(f'Iteration nº: {self.current_iteration}')

        if self.current_iteration >= self.iterations_per_epoch:
            self.current_iteration = 0
            self.reset()



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
        # Start Gazebo with the desired SDF file in the background
        home_directory = os.path.expanduser("~")
        sdf_file_path = os.path.join(home_directory, 'tfg', 'rwork', 'src', 'sdf_files', 'full_env_simpler.sdf')

        try:
            subprocess.Popen(['ign', 'gazebo', sdf_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            self.get_logger().error("Failed to start Gazebo process.")

    def unpause(self):
        # Use subprocess to execute the ros2 service call command
        command = "ros2 service call /world/full_env/control ros_gz_interfaces/srv/ControlWorld '{world_control: {pause: false}}'"
        try:
            subprocess.run(command, shell=True, check=True)
            self.get_logger().info("Simulation unpaused successfully.")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Failed to unpause simulation. Error: {e}")


def main(args=None):
    rclpy.init(args=args)
    joint_torque_controller = JointTorqueController()
    rclpy.spin(joint_torque_controller)
    joint_torque_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
