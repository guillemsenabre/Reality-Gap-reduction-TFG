import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray

class robotState(Node):
    def __init__(self):
        super().__init__('robot_state')

        self.get_logger().info('Starting Node')

        self.subscription = self.create_subscription(
            PoseArray,
            '/world/full_env/dynamic_pose/info',
            #self.pose_callback,
            self.gripper_pose,
            1)
        

    def gripper_pose(self, msg: PoseArray):

        ef_rb1 = msg.poses[15]
        ef_rb2 = msg.poses[27]

        ef1_data = {
            "x": ef_rb1.position.x,
            "y": ef_rb1.position.y,
            "z": ef_rb1.position.z,
            "orientation": {
                "x": ef_rb1.orientation.x,
                "y": ef_rb1.orientation.y,
                "z": ef_rb1.orientation.z,
                "w": ef_rb1.orientation.w
            }
        }

        ef2_data = {
            "x": ef_rb2.position.x,
            "y": ef_rb2.position.y,
            "z": ef_rb2.position.z,
            "orientation": {
                "x": ef_rb2.orientation.x,
                "y": ef_rb2.orientation.y,
                "z": ef_rb2.orientation.z,
                "w": ef_rb2.orientation.w
            }
        }

        self.get_logger().info(
            f'Pose gripper 1: {ef1_data}'
            )
        self.get_logger().info(
            f'Pose gripper 2: {ef2_data}'
            )




    def all_pose_callback(self, msg: PoseArray):
        pose_data = {}
        for idx, pose in enumerate(msg.poses):
            pose_name = f"pose_joint_{idx+1}"
            
            pose_data[pose_name] = {
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
        
        self.get_logger().info(f"Received pose data: {pose_data}")


def main(args=None):
    rclpy.init(args=args)
    robot_state = robotState()
    rclpy.spin(robot_state)
    robot_state.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
