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

# If no error occurred, print the output (if needed)
print(result.stdout.decode())
