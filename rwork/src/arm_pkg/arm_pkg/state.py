import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

class JointStateListener(Node):
    def __init__(self):
        super().__init__('joint_state_listener')

        self.get_logger().info('Starting Node')

        # create_subscription is from class Node (rclpy.node) -->
        # create_subscription(
        #   <message type>,
        #   <topic to subscribe>,
        #   <call to custom function>,
        #   queue size (how many messages can be stored in the queue if they
        # arrive faster than they're being processed),
        # )

        '''
        self.subscription = self.create_subscription(
            JointState,
            '/world/two_joint_arm_world/model/two_joint_arm/joint_state',
            self.check_data,
            10)
        '''

        self.subscription = self.create_subscription(
            JointState,
            '/world/two_joint_arm_world/dynamic_pose/info',
            self.check_data,
            5)

    def listener_callback(self, msg):
        self.get_logger().info('Received joint states:')
        for name, position in zip(msg.name, msg.position):
            self.get_logger().info(f'Joint: {name}, Position: {position}')
    
    def check_data(self, msg):
        if msg:
            self.listener_callback(msg)
        else:
            self.get_logger().info('No data to process')

def main(args=None):
    rclpy.init(args=args)
    joint_state_listener = JointStateListener()
    rclpy.spin(joint_state_listener)
    joint_state_listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
