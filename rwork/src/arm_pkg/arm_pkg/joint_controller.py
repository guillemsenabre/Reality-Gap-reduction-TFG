import rclpy
import math

from rclpy.node import Node
from std_msgs.msg import Float64

class JointTorqueController(Node):
    def __init__(self):
        super().__init__('joint_torque_controller')
        self.publisher_joint0 = self.create_publisher(Float64, '/two_joint_arm/joint0/pos_eff', 1)
        self.publisher_joint1 = self.create_publisher(Float64, '/two_joint_arm/joint1/pos_eff', 1)
        self.publisher_joint2 = self.create_publisher(Float64, '/two_joint_arm/joint2/pos_eff', 1)
        self.publisher_joint3 = self.create_publisher(Float64, '/two_joint_arm/joint3/pos_eff', 1)

        self.timer = self.create_timer(1, self.move_joints)

        self.angle = 0

    def move_joints(self):

        self.angle += 10
        
        msg_joint0, msg_joint1, msg_joint2, msg_joint3 = Float64()
        
        msg_joint0.data = 0.5 * math.sin(self.angle)# Apply a constant torque of x Nm
        msg_joint1.data = 10 * math.sin(self.angle)
        msg_joint2.data = 3 * math.sin(self.angle)
        msg_joint3.data = 1 * math.sin(self.angle)
        self.publisher_joint0.publish(msg_joint0)
        self.publisher_joint1.publish(msg_joint1)

        self.get_logger().info('Joint 0 torque: "%s"' % msg_joint0.data)
        self.get_logger().info('Joint 1 torque: "%s"' % msg_joint1.data)
        self.get_logger().info('Joint 2 torque: "%s"' % msg_joint2.data)
        self.get_logger().info('Joint 3 torque: "%s"' % msg_joint3.data)

def main(args=None):
    rclpy.init(args=args)
    joint_torque_controller = JointTorqueController()
    rclpy.spin(joint_torque_controller)
    joint_torque_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
