import rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import Float32Array

class Reward(Node):
    def __init__(self):
        super().__init__('reward_function')
        self.get_logger().info('Starting reward function node')

        self.state_subscription = self.create_subscription(
            Float32Array,
            '/state/data',
            1
        )


    def manhattan_distance(self, g1_pos, g2_pos, obj_pos):
        
        rg1 = abs(g1_pos[0] - obj_pos[0]) + abs(g1_pos[1] - obj_pos[1]) + abs(g1_pos[2] - obj_pos[2])

        rg2 = abs(g2_pos[0] - obj_pos[0]) + abs(g2_pos[1] - obj_pos[1]) + abs(g2_pos[2] - obj_pos[2])
    
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