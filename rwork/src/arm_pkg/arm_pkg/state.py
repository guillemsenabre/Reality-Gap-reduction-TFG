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

        end_effector_pose = msg.poses[-1]

        pose_data = {
            "x": end_effector_pose.position.x,
            "y": end_effector_pose.position.y,
            "z": end_effector_pose.position.z,
            "orientation": {
                "x": end_effector_pose.orientation.x,
                "y": end_effector_pose.orientation.y,
                "z": end_effector_pose.orientation.z,
                "w": end_effector_pose.orientation.w
            }
        }




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
