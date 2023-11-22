import rclpy
import math

from rclpy.node import Node
from std_msgs.msg import Float64
from ros_gz_interfaces.msg import Float32Array

class JointTorqueController(Node):
    def __init__(self):
        super().__init__('joint_torque_controller')

        self.current_angles_subscription = self.create_subscription(
            Float32Array,
            'packed/state/data',
            self.joint_angles_1,
            1)

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
        
        self.iterations_per_epoch = 30
        self.current_iteration = 0

    def move_joints(self):

        # keep the angle between 0 and 2π
        self.angle = (self.angle + 10) % (2 * math.pi) 
        
        # Test multipliers for each joint 
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
        # Calculate the reset positions by subtracting the current positions from themselves
        reset_positions = [-self.get_current_joint_position(idx) for idx in range(len(self.joint_publishers))]

        # Publish the reset positions to reset the joint positions
        for idx, publisher in enumerate(self.joint_publishers):
            msg = Float64()
            msg.data = reset_positions[idx]
            publisher.publish(msg)
            self.get_logger().info(f'Resetting Joint {idx} to {msg.data}')

        self.get_logger().info('Joints reset to initial positions')


def main(args=None):
    rclpy.init(args=args)
    joint_torque_controller = JointTorqueController()
    rclpy.spin(joint_torque_controller)
    joint_torque_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
