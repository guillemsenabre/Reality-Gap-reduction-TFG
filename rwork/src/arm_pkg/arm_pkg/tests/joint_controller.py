from email import iterators
import rclpy
import math
import subprocess
import os

from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

class JointTorqueController(Node):
    def __init__(self):
        super().__init__('joint_torque_controller')
        
        self.joint_publishers = []
        self.joint_names = [
                            'joint0_1', 
                            'joint1_1', 
                            'joint2_1',
                            'joint3_1',
                            'right_finger_1_joint',
                            'left_finger_1_joint',
                            'joint0_2', 
                            'joint1_2', 
                            'joint2_2',
                            'joint3_2',
                            'right_finger_2_joint',
                            'left_finger_2_joint',
                            ]
        
        for joint_name in self.joint_names:
            publisher = self.create_publisher(Float64, f'/arm/{joint_name}/wrench', 1)
            self.joint_publishers.append(publisher)

        self.move_timer = self.create_timer(1, self.move_joints)

        self.angle = 0
        
        self.iterations_per_epoch = 20
        self.current_iteration = 0

    def move_joints(self):

        self.angle = (self.angle + 10) % (2 * math.pi) 
        
        joint_multipliers_test = [
                                  8.5, 9, 3, 2, 1, 1,
                                  6, 6, 6, 6, 6, 6
                                  ]

        for idx, publisher in enumerate(self.joint_publishers):
            msg = Float64()
            msg.data = joint_multipliers_test[idx] * math.sin(self.angle)
            publisher.publish(msg)
            self.get_logger().info(f'Joint {idx} torque: "{msg.data}"')

        self.current_iteration += 1

        self.get_logger().info(f'Iteration nº: {self.current_iteration}')

        if self.current_iteration >= self.iterations_per_epoch:
            self.current_iteration = 0
            self.reset()



    def reset(self):
        self.get_logger().info("Resetting simulation...")

        # Create a JointState message to set joint positions
        joint_state_msg = JointState()
        joint_state_msg.name = self.joint_names
        joint_state_msg.position = [0.0] * len(self.joint_names)

        self.reset_publisher = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )

        self.reset_publisher.publish(joint_state_msg)    
        

def main(args=None):
    rclpy.init(args=args)
    joint_torque_controller = JointTorqueController()
    rclpy.spin(joint_torque_controller)
    joint_torque_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
