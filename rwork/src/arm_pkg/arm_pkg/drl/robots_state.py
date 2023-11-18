import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray
from ros_gz_interfaces.msg import Float32Array


        ######  STATE CLASS  ######

class RobotState(Node):
    def __init__(self):
        super().__init__('robot_state')

        self.get_logger().info('Starting Robots State Node')

        # Storage for latest data

        self.latest_joint_state_1 = None
        self.latest_joint_state_2 = None

        self.latest_end_effector_pose_1 = None
        self.latest_end_effector_pose_2 = None

        self.latest_object_pose = None


        # Publisher for robot 1 data
        self.robot1_publisher = self.create_publisher(
            Float32Array,
            '/robot1_states',
            1)

        # Publisher for robot 2 data
        self.robot2_publisher = self.create_publisher(
            Float32Array,
            '/robot2_states',
            1)

        # Publisher for object data
        self.object_publisher = self.create_publisher(
            Float32Array,
            '/object_states',
            1)

        # Subscriptions

        self.pose_subscription = self.create_subscription(
            PoseArray,
            '/world/full_env/dynamic_pose/info',
            self.gripper_object_pose,
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

    def gripper_object_pose(self, msg: PoseArray):
        self.latest_end_effector_pose_1 = self.extract_coordinates(msg.poses[15])
        self.latest_end_effector_pose_2 = self.extract_coordinates(msg.poses[27])
        
        self.latest_object_pose = self.extract_coordinates(msg.poses[3])

        self.states()


        ######  JOINT ANGLES PROCESSING ######

    def joint_angles_1(self, msg):
        # Exclude fixed joints and finger joints
        relevant_joints = [joint for joint in msg.name if "joint" in joint and "finger" not in joint]
        self.latest_joint_state_1 = [position for position in msg.position]

    def joint_angles_2(self, msg):
        # Exclude fixed joints and finger joints
        relevant_joints = [joint for joint in msg.name if "joint" in joint and "finger" not in joint]
        self.latest_joint_state_2 = [position for position in msg.position]


    
    def states(self):
        # Robot 1 data
        robot1_data = {
            "joint_state_1": self.latest_joint_state_1,
            "gripper_pose_1": self.latest_end_effector_pose_1
        }

        # Robot 2 data
        robot2_data = {
            "joint_state_2": self.latest_joint_state_2,
            "gripper_pose_2": self.latest_end_effector_pose_2
        }

        # Object data
        object_data = {
            "object_pose": self.latest_object_pose
        }

        self.get_logger().info(f'State (Robot 1): {robot1_data}')
        self.get_logger().info(f'State (Robot 2): {robot2_data}')
        self.get_logger().info(f'State (Object): {object_data}')

        # Publish to topics
        self.robot1_publisher.publish(Float32Array(data=[float(value) for value in robot1_data.values()]))
        self.robot2_publisher.publish(Float32Array(data=[float(value) for value in robot2_data.values()]))
        self.object_publisher.publish(Float32Array(data=[float(value) for value in object_data.values()]))


        ######  INITIALIZATION FUNCTION ######


def main(args=None):
    rclpy.init(args=args)
    robot_state = RobotState()
    rclpy.spin(robot_state)
    robot_state.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
