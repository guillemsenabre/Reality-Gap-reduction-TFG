import rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import Float32Array

class Reward(Node):
    def __init__(self):
        super().__init__('reward_function')
        self.get_logger().info('Starting reward function node ...')


        self.state_subscription = self.create_subscription(
            Float32Array,
            'packed/state/data',
            self.states_unpack,
            1
        )

        self.get_logger().info('Waiting for data ...')

    def states_unpack(self, msg: Float32Array):

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
        gripper_1_pos = data[4:7]
        gripper_2_pos = data[15:18]
        object_pos = data[22:25]

        # Log gripper and object positions for debugging
        self.get_logger().info('Gripper 1 Position: {}'.format(gripper_1_pos))
        self.get_logger().info('Gripper 2 Position: {}'.format(gripper_2_pos))
        self.get_logger().info('Object Position: {}'.format(object_pos))



    def manhattan_distance(self, g1_pos, g2_pos, obj_pos):
        
        rg1 = abs(g1_pos[0] - obj_pos[0]) + abs(g1_pos[1] - obj_pos[1]) + abs(g1_pos[2] - obj_pos[2])

        rg2 = abs(g2_pos[0] - obj_pos[0]) + abs(g2_pos[1] - obj_pos[1]) + abs(g2_pos[2] - obj_pos[2])
    






    
    def object_deviation(self):
        pass




def main(args=None):
    rclpy.init(args=args)
    reward_function = Reward()
    rclpy.spin(reward_function)
    reward_function.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()