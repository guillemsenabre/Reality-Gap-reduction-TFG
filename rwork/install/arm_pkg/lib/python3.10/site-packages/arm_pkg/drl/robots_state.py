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


        ######  JOINT ANGLES PROCESSING ######

    def joint_angles_1(self, msg):
        # Exclude fixed joints and finger joints
        relevant_joints = [joint for joint in msg.name if "joint" in joint and "finger" not in joint]
        self.latest_joint_state_1 = {name: position for name, position in zip(relevant_joints, msg.position)}


    def joint_angles_2(self, msg):
        # Exclude fixed joints and finger joints
        relevant_joints = [joint for joint in msg.name if "joint" in joint and "finger" not in joint]
        self.latest_joint_state_2 = {name: position for name, position in zip(relevant_joints, msg.position)}
    
    def gripper_object_pose(self, msg: PoseArray):
        self.latest_end_effector_pose_1 = self.extract_coordinates(msg.poses[15])
        self.latest_end_effector_pose_2 = self.extract_coordinates(msg.poses[27])
        
        self.latest_object_pose = self.extract_coordinates(msg.poses[3])

        self.states()
        
    
    def states(self):
        # Separate data for each robot and object
        robot1_data = {
            "joint_state_1": self.latest_joint_state_1,
            "gripper_pose_1": self.latest_end_effector_pose_1
        }

        robot2_data = {
            "joint_state_2": self.latest_joint_state_2,
            "gripper_pose_2": self.latest_end_effector_pose_2
        }

        object_data = {
            "object_pose": self.latest_object_pose
        }

        # Flatten nested dictionaries
        flatten = lambda d: {k + '_' + k2 if k2 in d2 else k2: v2 for k, v in d.items() for k2, v2 in v.items()}
        flattened_robot1_data = flatten(robot1_data)
        flattened_robot2_data = flatten(robot2_data)
        flattened_object_data = flatten(object_data)

        # Log the data for debugging
        self.get_logger().info(f'State (Robot 1): {flattened_robot1_data}')
        self.get_logger().info(f'State (Robot 2): {flattened_robot2_data}')
        self.get_logger().info(f'State (Object): {flattened_object_data}')

        # Publish data for each robot and object
        self.robot1_publisher.publish(Float32Array(data=[float(value) for value in flattened_robot1_data.values()]))
        self.robot2_publisher.publish(Float32Array(data=[float(value) for value in flattened_robot2_data.values()]))
        self.object_publisher.publish(Float32Array(data=[float(value) for value in flattened_object_data.values()]))



        ######  INITIALIZATION FUNCTION ######


def main(args=None):
    rclpy.init(args=args)
    robot_state = RobotState()
    rclpy.spin(robot_state)
    robot_state.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
