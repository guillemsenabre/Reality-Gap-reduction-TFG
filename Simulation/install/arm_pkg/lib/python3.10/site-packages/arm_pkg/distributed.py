import rclpy
import JointStateListener from joint_listener.py

from rclpy.node import Node
from sensor_msgs.msg import JointState



def main():
     rclpy.init()
     listener = JointStateListener()
     rclpy.spin(listener)
     listener.destroy_node()
     rclpy.shutdown()


if __name__=='__main__':
    main()