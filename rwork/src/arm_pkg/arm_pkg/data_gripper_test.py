import rclpy

from rclpy.node import Node
from std_msgs.msg import Float64

class JointTorqueController(Node):
    def __init__(self):
        super().__init__('joint_torque_controller')
        
        self.joint_publishers = []
        self.joint_names = [
                            'joint1_1', 
                            ]
        
        for joint_name in self.joint_names:
            publisher = self.create_publisher(Float64, f'/arm/{joint_name}/wrench', 1)
            self.joint_publishers.append(publisher)


        self.torque_direction = 1

        self.timer = self.create_timer(3, self.move_joint)

    def move_joint(self):

        for idx, publisher in enumerate(self.joint_publishers):
            msg = Float64()
            msg.data = 2.0 * self.torque_direction
            publisher.publish(msg)
            self.get_logger().info(f'{self.joint_names[idx]} torque: "{msg.data}"')

        self.torque_direction *= -1

def main(args=None):
    rclpy.init(args=args)
    joint_torque_controller = JointTorqueController()
    rclpy.spin(joint_torque_controller)
    joint_torque_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
