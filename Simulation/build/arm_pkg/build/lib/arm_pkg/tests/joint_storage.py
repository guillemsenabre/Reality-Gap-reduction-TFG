import csv
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateProcessor(Node):

    def __init__(self):
        super().__init__('joint_state_listener')
        self.subscription = self.create_subscription(
            JointState, # message type
            '/world/two_joint_arm_world/model/two_joint_arm/joint_state', # topic
            self.listener_callback,
            10) # message queue
        
        # Initialize the csv file and keep it opened (better performance)
        self.csv_file_path = '~/gz_ros2_test2/src/csv_files/joint_states.csv'
        self.csv_file = self.initialize_csv_file()

    def initialize_csv_file(self):
        
        # expand the file path so it's correctly expanded.
        expanded_file_path = os.path.expanduser(self.csv_file_path)

        # check that the file doesn't already exists
        if not os.path.isfile(expanded_file_path):

            csv_file = open(expanded_file_path, 'w', newline='')

            # create a csv writer object to add rows data into the file
            csv_writer = csv.writer(csv_file)

            # Writing the header of the CSV file
            header = ['time_sec', 'time_nsec', 'joint_name', 'position', 'velocity', 'effort']
            csv_writer.writerow(header)
        
        else:
            csv_file = open(expanded_file_path, 'a', newline = '')
        
        return csv_file

    # subscribe to node and receive data
    def listener_callback(self, msg):
        self.get_logger().info('Received joint states:')
        csv_writer = csv.writer(self.csv_file)
        for name, position, velocity, effort in zip(msg.name, msg.position, msg.velocity, msg.effort):
            self.get_logger().info(f'Joint: {name}, Position: {position}, Velocity: {velocity}, Effort: {effort}')
            row = [msg.header.stamp.sec, msg.header.stamp.nanosec, name, position, velocity, effort]
            csv_writer.writerow(row)

def main(args=None):
    rclpy.init(args=args)
    joint_state_processor = JointStateProcessor()
    rclpy.spin(joint_state_processor)
    joint_state_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
