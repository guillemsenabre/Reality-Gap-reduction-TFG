from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='arm_pkg',
            executable='main',
            name='main'
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

if __name__ == '__main__':
    generate_launch_description()
