import rclpy
import subprocess
import os

from rclpy.node import Node
from std_msgs.msg import Float64
from ros_gz_interfaces.srv import ControlWorld

# THIS NODE NEEDS TO BE WORKING TOGETHER WITH A SERVICE BRIDGE. 
# CAN BE FOUND IN --> bridge_commands

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


def main(args=None):
    rclpy.init(args=args)
    reset = Reset()
    rclpy.spin(reset)
    reset.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
