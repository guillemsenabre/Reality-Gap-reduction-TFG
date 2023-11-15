import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray


        ######  STATE CLASS  ######

class robotState(Node):
    def __init__(self):
        super().__init__('robot_state')

        self.get_logger().info('Starting Robots State Node')

        # Storage for latest data

        self.latest_joint_state_1 = None
        self.latest_joint_state_2 = None

        self.latest_end_effector_pose_1 = None
        self.latest_end_effector_pose_2 = None

        self.latest_object_pose = None

        # Subscriptions

        self.pose_subscription = self.create_subscription(
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
        self.update_robot_state()

        ######  OBJECT POSE ######
    
    def object_pose(self, msg: PoseArray):
        self.latest_object_pose = self.extract_coordinates(msg.poses[5])

        ######  JOINT ANGLES PROCESSING ######

    def joint_angles_1(self, msg):
        self.latest_joint_state_1 = {name: position for name, position in zip(msg.name, msg.position)}
        self.update_robot_state()

    def joint_angles_2(self, msg):
        self.latest_joint_state_2 = {name: position for name, position in zip(msg.name, msg.position)}
        self.update_robot_state()


        ###### SYNCHRONIZATION FUNCTION ######

    def update_robot_state(self):
        if self.latest_joint_state_1 and self.latest_end_effector_pose_1:
            robot_state_1 = (self.latest_joint_state_1, self.latest_end_effector_pose_1)
            self.get_logger().info(f'Robot 1 State: {robot_state_1}')

        if self.latest_joint_state_2 and self.latest_end_effector_pose_2:
            robot_state_2 = (self.latest_joint_state_2, self.latest_end_effector_pose_2)
            self.get_logger().info(f'Robot 2 State: {robot_state_2}')

        if self.latest_object_pose is not None:
            object_pose = self.latest_object_pose
            self.get_logger().info(f'Object pose: {object_pose}')   




        ######  INITIALIZATION FUNCTION ######


def main(args=None):
    rclpy.init(args=args)
    robot_state = robotState()
    rclpy.spin(robot_state)
    robot_state.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
