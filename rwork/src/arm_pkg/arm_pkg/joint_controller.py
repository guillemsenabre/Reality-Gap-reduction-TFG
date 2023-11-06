import rclpy
import math

from rclpy.node import Node
from std_msgs.msg import Float64

class JointTorqueController(Node):
    def __init__(self):
        super().__init__('joint_torque_controller')
        
        self.joint_publishers = []
        self.joint_names = [
                            'joint0_1', 
                            'joint1_1', 
                            'joint2_1',
                            'joint3_1',
                            'joint4_1',
                            'right_finger_joint_1',
                            'left_finger_joint_1',
                            'joint0_2', 
                            'joint1_2', 
                            'joint2_2',
                            'joint3_2',
                            'joint4_2',
                            'right_finger_joint_2',
                            'left_finger_joint_2',
                            ]
        
        for joint_name in self.joint_names:
            publisher = self.create_publisher(Float64, f'/two_joint_arm/{joint_name}/pos_eff', 1)
            self.joint_publishers.append(publisher)

        self.timer = self.create_timer(1, self.move_joints)

        self.angle = 0

    def move_joints(self):

        # keep the angle between 0 and 2π
        self.angle = (self.angle + 10) % (2 * math.pi) 
        
        # Test multipliers for each joint 
        joint_multipliers_test = [
                                  8.5, 9, 3, 2, 2, 1, 1,
                                  6, 6, 6, 6, 6, 6, 6
                                  ]

        for idx, publisher in enumerate(self.joint_publishers):
            msg = Float64()
            msg.data = joint_multipliers_test[idx] * math.sin(self.angle)
            publisher.publish(msg)
            self.get_logger().info(f'Joint {idx} torque: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    joint_torque_controller = JointTorqueController()
    rclpy.spin(joint_torque_controller)
    joint_torque_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
