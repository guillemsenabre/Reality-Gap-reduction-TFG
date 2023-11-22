import subprocess
import os

# Get the full path to the home directory
home_directory = os.path.expanduser("~")

# Full path to the SDF file
sdf_file_path = os.path.join(home_directory, 'tfg', 'rwork', 'src', 'sdf_files', 'full_env_simpler.sdf')

# Run Gazebo subprocess
try:
    result = subprocess.run(['ign', 'gazebo', sdf_file_path], check=True, stderr=subprocess.PIPE)
except subprocess.CalledProcessError as e:
    print("Error running Gazebo:")
    print(e.stderr.decode())
    raise  # Re-raise the exception to terminate the script

# Use subprocess to execute the ros2 service call command
command = 'ros2 service call /world/full_env_simpler/control ros_gz_interfaces/srv/ControlWorld "{world_control: {pause: false}}"'

try:
    subprocess.run(command, shell=True, check=True)
    print("Simulation unpaused successfully.")
except subprocess.CalledProcessError as e:
    print(f"Failed to unpause simulation. Error: {e}")