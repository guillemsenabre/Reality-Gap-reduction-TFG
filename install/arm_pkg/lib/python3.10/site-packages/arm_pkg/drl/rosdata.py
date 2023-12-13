import time
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64, Float32
from ros_gz_interfaces.msg import Float32Array

from .configuration import Configuration
from .ddpg import DDPGAgent



class RosData(Node):
    def __init__(self):    
        super().__init__('ros_data')

        self.get_logger().info(f'Starting training...')

        config = Configuration()
        
        self.state = np.array([])
        self.reward_list = []
        self.reward_value = 0.0
        self.margin_value = config.margin_value
        
        self.maximum_accumulative_reward = config.maximum_accumulative_reward

        state_dim = config.state_dim  
        action_dim = config.action_dim

        self.agent = DDPGAgent(state_dim, action_dim)


        # Subscribing to topics data
        self.state_subscription = self.create_subscription(
            Float32Array,
            'packed/state/data',
            self.process_state_data,
            10
        )

        self.reward_subscription = self.create_subscription(
            Float32,
            'reward/data',
            self.process_reward_data,
            10
        )


        self.joint_publishers = []
        self.joint_names = config.joint_names
        
        for joint_name in self.joint_names:
            publisher = self.create_publisher(Float64, f'/arm/{joint_name}/wrench', 1)
            self.joint_publishers.append(publisher)

    def process_state_data(self, msg: Float32Array):
        data = msg.data
        
        # Extract gripper and object positions

        gripper_1_pos = np.array(data[4:7])

        gripper_2_pos = np.array(data[15:18])
        object_pos = np.array(data[22:25])

        #self.get_logger().info(f"OBJECT POS: {object_pos}")

        object_1_pos = np.array([object_pos[0] - 0.125, object_pos[1], object_pos[2]])
        object_2_pos = np.array([object_pos[0] + 0.125, object_pos[1], object_pos[2]])

        self.state = np.concatenate([gripper_1_pos, gripper_2_pos, object_1_pos, object_2_pos])

    def process_reward_data(self, msg: Float64):
        self.reward_value = msg.data

    def terminal_condition(self):
        self.reward_list.append(self.reward_value)

        if self.maximum_accumulative_reward == len(self.reward_list):
            margin = abs(self.margin_value * self.reward_list[0])
            difference = abs(self.reward_list[0] - self.reward_list[-1]) 
            self.reward_list = []

            if difference <= margin:
                print(f'Reached local minimum!')
                print(f'Difference: {round(difference, 4)}')
                print(f'Margin: {round(margin, 4)}')
                return True
                        
        elif (self.state[11] or self.state[8]) < 1.2:
            print(f'Oops, object dropped')
            return True
        
        else:
            return False
        
        return False
            
    def move_joints(self, action):
        for idx, publisher in enumerate(self.joint_publishers):
            msg = Float64()
            msg.data = float(action[idx])
            publisher.publish(msg)
            #self.get_logger().info(f'Joint {idx} action: {action[idx]}, torque: {msg.data}')

        time.sleep(0.01)



if __name__ == '__main__':
    rclpy.init()
    ros_data = RosData()
    rclpy.spin(ros_data)
    ros_data.destroy_node()
    rclpy.shutdown()