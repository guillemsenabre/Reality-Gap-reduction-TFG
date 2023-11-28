import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray
from ros_gz_interfaces.msg import Float32Array


        ######  STATE CLASS  ######

class RobotState(Node):
    def __init__(self):
        super().__init__('robot_state')

        self.get_logger().info('Starting State Receiver Node')

        # Storage for latest data

        self.latest_joint_state_1 = None
        self.latest_joint_state_2 = None

        self.latest_end_effector_pose_1 = None
        self.latest_end_effector_pose_2 = None

        self.latest_object_pose = None

        # Publisher

        self.state_publisher = self.create_publisher(
            Float32Array,
            'packed/state/data',
            1)

        # Subscriptions

        self.get_logger().info('Waiting for data...')

        self.j1_subscription = self.create_subscription(
            JointState,
            '/world/full_env_simpler/model/arm_1/joint_state',
            self.joint_angles_1,
            1)
        
        self.j2_subscription = self.create_subscription(
            JointState,
            '/world/full_env_simpler/model/arm_2/joint_state',
            self.joint_angles_2,
            1)

        self.pose_subscription = self.create_subscription(
            PoseArray,
            '/world/full_env_simpler/dynamic_pose/info',
            self.gripper_object_pose,
            1)
        

        ######  GRIPPER GENERAL COORDENATES FUNCTIONS ######
    
    # Gripper coordinates functions
    def extract_coordinates(self, pose):
        return [
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        ]

    def gripper_object_pose(self, msg: PoseArray):
        self.get_logger().info("Getting pose data")
        self.latest_end_effector_pose_1 = self.extract_coordinates(msg.poses[15])
        self.latest_end_effector_pose_2 = self.extract_coordinates(msg.poses[27])
        self.latest_object_pose = self.extract_coordinates(msg.poses[3])

        self.get_logger().info(f'Gripper 1: {self.latest_end_effector_pose_1}')
        self.get_logger().info(f'Gripper 2: {self.latest_end_effector_pose_2}')
        self.get_logger().info(f'Object: {self.latest_object_pose}')

        self.states_pack()

    # Joint angles processing functions
    def joint_angles_1(self, msg):
        # Exclude fixed joints and finger joints
        relevant_joints = [joint for joint in msg.name if "joint" in joint and "finger" not in joint]
        self.latest_joint_state_1 = [msg.position[msg.name.index(joint)] for joint in relevant_joints]

        #self.get_logger().info(f'Joint State (Robot 1): {dict(zip(relevant_joints, self.latest_joint_state_1))}')

    def joint_angles_2(self, msg):
        # Exclude fixed joints and finger joints
        relevant_joints = [joint for joint in msg.name if "joint" in joint and "finger" not in joint]
        self.latest_joint_state_2 = [msg.position[msg.name.index(joint)] for joint in relevant_joints]

        #self.get_logger().info(f'Joint State (Robot 2): {dict(zip(relevant_joints, self.latest_joint_state_2))}')

    # States function
    def states_pack(self):
        # Flatten the data and publish to the topic
        # * for unpacking data
        flat_data = [
            *self.latest_joint_state_1, *self.latest_end_effector_pose_1,
            *self.latest_joint_state_2, *self.latest_end_effector_pose_2,
            *self.latest_object_pose
        ]

        self.get_logger().info(f'State: {flat_data}')
        float_array_msg = Float32Array(data=flat_data)
        self.state_publisher.publish(float_array_msg)


        ######  INITIALIZATION FUNCTION ######


def main(args=None):
    rclpy.init(args=args)
    robot_state = RobotState()
    rclpy.spin(robot_state)
    robot_state.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
