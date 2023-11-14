import rclpy

from rclpy.node import Node
from geometry_msgs.msg import PoseArray


        ######  STATE CLASS  ######

class objectState(Node):
    def __init__(self):
        super().__init__('object_state')

        self.get_logger().info('Starting Object State Node')

        # Subscriptions

        self.grippers_subscription = self.create_subscription(
            PoseArray,
            '/world/full_env/dynamic_pose/info',
            self.gripper_pose,
            1)

        ######  GRIPPER GENERAL COORDENATES FUNCTIONS ######
    
    def extract_coordinates(self, pose):
        
        return {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
            "orientation": {
                "x": pose.orientation.x,
                "y": pose.orientation.y,
                "z": pose.orientation.z,
                "w": pose.orientation.w
            }
        }

    def gripper_pose(self, msg: PoseArray):
        self.latest_end_effector_pose_1 = self.extract_coordinates(msg.poses[15])
        self.latest_end_effector_pose_2 = self.extract_coordinates(msg.poses[27])
        self.update_robot_state()


        ######  INITIALIZATION FUNCTIONS ######


def main(args=None):
    rclpy.init(args=args)
    robot_state = robotState()
    rclpy.spin(robot_state)
    robot_state.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
