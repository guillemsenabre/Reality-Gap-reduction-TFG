import rclpy
import numpy as np
from rclpy.node import Node
from ros_gz_interfaces.msg import Float32Array
from std_msgs.msg import Float32




'''
                            Data structure

[
    'f',
    J01, J11, J21, J31, EX1, EY1, EZ1, EI1, EJ1, EK1,EW1,   --> ROBOT 1
    J02, J12, J22, J32, EX2, EY2, EZ2, EI2, EJ2, EK2,EW2,   --> ROBOT 2  
    OBJX, OBJY, OBJZ, OBJI, OBJJ, OBJK, OBJW                --> OBJECT
] 

'''


class Reward(Node):
    def __init__(self):
        super().__init__('reward_function')
        self.get_logger().info('Starting reward function node ...')

        # Publisher

        self.reward_publisher = self.create_publisher(
            Float32,
            'reward/data',
            1)
        
        # Subscriber

        self.state_subscription = self.create_subscription(
            Float32Array,
            'packed/state/data',
            self.reward_function,
            1
        )

        self.get_logger().info('Waiting for data ...')


    def manhattan_distance(self, msg: Float32Array):
                
        data = msg.data

        # Extract gripper and object positions
        gripper_1_pos = data[4:7]
        gripper_2_pos = data[15:18]
        object_pos = data[22:25]

        object_1_pos = [object_pos[0] - 0.125, object_pos[1], object_pos[2]]
        object_2_pos = [object_pos[0] + 0.125, object_pos[1], object_pos[2]]

        rg1 = abs(gripper_1_pos[0] - object_1_pos[0]) + abs(gripper_1_pos[1] - object_1_pos[1]) + abs(gripper_1_pos[2] - object_1_pos[2])
        rg2 = abs(gripper_2_pos[0] - object_2_pos[0]) + abs(gripper_2_pos[1] - object_2_pos[1]) + abs(gripper_2_pos[2] - object_2_pos[2])

        distance_reward = -np.exp(rg1) - np.exp(rg2)

        
        return distance_reward

    
    def object_deviation(self, msg: Float32Array):
        data = msg.data
        object_deviation = data[25:29]  # OBJI, OBJJ, OBJK, OBJW
        print(f'Object deviation: {object_deviation}')
        deviation_penalty = -0.5 * (abs(object_deviation[0]) + abs(object_deviation[1]) + abs(object_deviation[2]))
        print(f'deviation penalty: {deviation_penalty}')
        return deviation_penalty
    
    #NOTE - Should award roll ~ pi/2 rad, pitch ~ yaw ~ 0.0 rad
    def end_effector_deviation(self, msg: Float32Array):
        pass
    
    def grabbing_object(self, msg: Float32Array):
        pass

    def limit_base_joint(self, msg: Float32Array):
        pass
    
    def reward_function(self, msg: Float32Array):

        distance_reward = self.manhattan_distance(msg)
        deviation_reward = self.object_deviation(msg)

        reward = distance_reward + deviation_reward

        self.get_logger().info(f'R: {reward}')
        self.reward_publisher.publish(Float32(data=reward))



def main(args=None):
    rclpy.init(args=args)
    reward_function = Reward()
    rclpy.spin(reward_function)
    reward_function.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()