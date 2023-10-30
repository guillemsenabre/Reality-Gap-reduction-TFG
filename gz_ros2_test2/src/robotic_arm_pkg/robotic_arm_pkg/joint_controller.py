import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class JointTorqueController(Node):
    def __init__(self):
        super().__init__('joint_torque_controller')
        self.publisher_joint0 = self.create_publisher(Float64, '/two_joint_arm/joint0/pos_eff', 10)
        self.publisher_joint1 = self.create_publisher(Float64, '/two_joint_arm/joint1/pos_eff', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        msg_joint0 = Float64()
        msg_joint1 = Float64()
        msg_joint0.data = 0.5  # Apply a constant torque of x Nm
        msg_joint1.data = 10.5
        self.publisher_joint0.publish(msg_joint0)
        self.publisher_joint1.publish(msg_joint1)

        self.get_logger().info('Joint 0 torque: "%s"' % msg_joint0.data)
        self.get_logger().info('Joint 1 torque: "%s"' % msg_joint1.data)

def main(args=None):
    rclpy.init(args=args)
    joint_torque_controller = JointTorqueController()
    rclpy.spin(joint_torque_controller)
    joint_torque_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
