import subprocess
import os
import time
import psutil

from rclpy.node import Node

class Reset(Node):
    def __init__(self):
        super().__init__('reset')

    def reset(self):
        self.get_logger().info("Resetting simulation...")
        self.kill_gazebo_process()
        time.sleep(3)
        self.run_gazebo()
        time.sleep(7)
        self.unpause()

    def kill_gazebo_process(self):
        # Find and kill the Gazebo process
        try:
            subprocess.run(['pkill', '-f', 'gazebo'], check=True)
        except subprocess.CalledProcessError:
            self.get_logger().warning("Failed to kill Gazebo process.")

    def run_gazebo(self):
        self.get_logger().info("Check if gazebo is dead...")
        print(not self.is_gazebo_running())

        if not self.is_gazebo_running():
            self.get_logger().info("starting gazebo simulator...")
            home_directory = os.path.expanduser("~")
            sdf_file_path = os.path.join(home_directory, 'tfg', 'Simulation', 'src', 'sdf_files', 'full_env_simpler.sdf')

            try:
                subprocess.Popen(['ign', 'gazebo', sdf_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                self.get_logger().error("Failed to start Gazebo process.")
        
        else:
            self.get_logger().error("Gazebo is still running...")
            time.sleep(1)
            self.get_logger().error("Trying again...")
            self.reset()

    def unpause(self):
        # Use subprocess to execute the ros2 service call command
        command = 'ros2 service call /world/full_env_simpler/control ros_gz_interfaces/srv/ControlWorld "{world_control: {pause: false}}"'
        try:
            subprocess.run(command, shell=True, check=True)
            self.get_logger().info("Simulation unpaused successfully.")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Failed to unpause simulation. Error: {e}")

    def is_gazebo_running(self):
        for process in psutil.process_iter(['pid', 'name']):
            if 'gazebo' in process.info['name']:
                return True
        return False
