import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

class robotStates(Node):
    def __init__(self):
        super().__init__('robot_states')

        self.get_logger().info('Starting Node')

        self.subscription = self.create_subscription(
            JointState,
            '/world/two_joint_arm_world/dynamic_pose/info',
            self.pose_callback,
            5)

    def pose_callback(self, msg):
        self.get_logger().info(f'Received pose data: {msg}')


def main(args=None):
    rclpy.init(args=args)
    robot_states = robotState()
    rclpy.spin(robot_states)
    robot_states.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()