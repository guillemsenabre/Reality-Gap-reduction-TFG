import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray
from ros_gz_interfaces.msg import Float32Array

# Train the model in the simulation
# Save the model from the simulation
# Freeze a layer of neurons
# Inference with the real robots while continue learning with the freezed layer

