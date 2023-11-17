import rclpy
from rclpy.node import Node

class Reward(Node):

    def __init__(self):
        super().__init__('reward_function')


    def manhattan_distance(self, gripper1_pos, gripper2_pos, object_pos):
        return None
    
    def object_deviation(self):
        return None










def main(args=None):
    rclpy.init(args=args)
    reward_function = Reward()
    rclpy.spin(reward_function)
    reward_function.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()