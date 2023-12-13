import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    user_input = input("Configure for learning or inference? (Type 'learning' or 'inference'): ")

    if user_input.lower() == 'learning':
        return LaunchDescription([
            Node(
                package='arm_pkg',
                executable='training',
                name='training'
            ),
            Node(
                package='arm_pkg',
                executable='reward_function',
                name='reward_function'
            ),
            Node(
                package='arm_pkg',
                executable='robots_state',
                name='robots_state'
            )
        ])
    elif user_input.lower() == 'inference':
        print("Nothing to run here yet...")
    else:
        print("Invalid input. Please type 'learning' or 'inference'.")

if __name__ == '__main__':
    generate_launch_description()
