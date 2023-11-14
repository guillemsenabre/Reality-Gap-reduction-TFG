import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray

class robotState(Node):
    def __init__(self):
        super().__init__('robot_state')

        self.get_logger().info('Starting Node')

        
        # Storage for latest data

        self.latest_joint_state_1 = None
        self.latest_joint_state_2 = None

        self.latest_end_effector_pose_1 = None
        self.latest_end_effector_pose_2 = None

        # Subscriptions to nodes

        self.grippers_subscription = self.create_subscription(
            PoseArray,
            '/world/full_env/dynamic_pose/info',
            self.gripper_pose,
            1)

        self.j1_subscription = self.create_subscription(
            JointState,
            '/world/full_env/model/arm_1/joint_state',
            self.joint_angles_1,
            1)
        
        self.j2_subscription = self.create_subscription(
            JointState,
            '/world/full_env/model/arm_2/joint_state',
            self.joint_angles_2,
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

                ######  JOINT ANGLES FUNCTIONS ######

    def joint_angles_1(self, msg):
        self.get_logger().info('Joints 1')
        for name, position in zip(msg.name, msg.position):
            self.get_logger().info(f'Joint: {name}, Angle: {position}')
    
    def joint_angles_2(self, msg):
        self.get_logger().info('Joints 2')
        for name, position in zip(msg.name, msg.position):
            self.get_logger().info(f'Joint: {name}, Angle: {position}')



                 ######  INITIALIZATION FUNCTIONS ######


def main(args=None):
    rclpy.init(args=args)
    robot_state = robotState()
    rclpy.spin(robot_state)
    robot_state.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
