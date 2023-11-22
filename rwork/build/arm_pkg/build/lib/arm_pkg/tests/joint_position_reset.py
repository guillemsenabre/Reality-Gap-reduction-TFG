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
            self.get_current_positions,
            1)
        
        self.joint_publishers = []
        self.pos_publishers = []

        self.joint_names = [
                            'joint0_1', 
                            'joint1_1', 
                            'joint2_1',
                            'joint3_1',
                            'joint0_2', 
                            'joint1_2', 
                            'joint2_2',
                            'joint3_2',
                            ]
        
        for joint_name in self.joint_names:
            torque_publisher = self.create_publisher(Float64, f'/arm/{joint_name}/wrench', 1)
            pos_publisher = self.create_publisher(Float64, f'/{joint_name}/reset', 1)
            self.joint_publishers.append(torque_publisher)
            self.pos_publishers.append(pos_publisher)

        self.move_timer = self.create_timer(1, self.move_joints)
        

        self.angle = 0
        self.iterations_per_epoch = 30
        self.current_iteration = 0

    def move_joints(self):

        # keep the angle between 0 and 2π
        self.angle = (self.angle + 10) % (2 * math.pi) 
        
        # Test multipliers for each joint 
        joint_multipliers_test = [
                                  8.5, 9, 3, 2,
                                  6, 6, 6, 6
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
        reset_positions = [-self.get_current_position[idx] for idx in range(len(self.pos_publishers))]

        # Publish the reset positions to reset the joint positions
        for idx, publisher in enumerate(self.pos_publishers):
            msg = Float64()
            msg.data = reset_positions[idx]
            publisher.publish(msg)
            self.get_logger().info(f'Resetting Joint {idx} to {msg.data}')

        self.get_logger().info('Joints reset to initial positions')


    def get_current_positions(self, msg: Float32Array):

        '''
                                    Data structure

        [
            'f',
            J01, J11, J21, J31, EX1, EY1, EZ1, EI1, EJ1, EK1,EW1,   --> ROBOT 1
            J02, J12, J22, J32, EX2, EY2, EZ2, EI2, EJ2, EK2,EW2,   --> ROBOT 2  
            OBJX, OBJY, OBJZ, OBJI, OBJJ, OBJK, OBJW                --> OBJECT
        ] 

        '''

        data = msg.data

        # Extract gripper and object positions
        joints1_pos = data[0:4]
        joints2_pos = data[10:14]
        current_joint_position = joints1_pos + joints2_pos

        return current_joint_position


def main(args=None):
    rclpy.init(args=args)
    joint_torque_controller = JointTorqueController()
    rclpy.spin(joint_torque_controller)
    joint_torque_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
