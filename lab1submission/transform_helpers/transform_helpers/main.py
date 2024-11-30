import rclpy
from rclpy.node import Node
# TODO: Import the message type that holds data describing robot joint angle states
# this tutorial may have hints: https://docs.ros.org/en/rolling/Tutorials/Intermediate/URDF/Using-URDF-with-Robot-State-Publisher.html#publish-the-state
from sensor_msgs.msg import JointState
# TODO: Import the class that publishes coordinate frame transform information
# this tutorial may have hints: https://docs.ros.org/en/rolling/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Py.html
from tf2_ros import TransformBroadcaster
# TODO: Import the message type that expresses a transform from one coordinate frame to another
# this same tutorial from earlier has hints: https://docs.ros.org/en/rolling/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Py.html
from geometry_msgs.msg import TransformStamped

import numpy as np
from numpy.typing import NDArray

from transform_helpers.utils import rotmat2q

# Modified DH Params for the Franka FR3 robot arm
# https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters
# meters
a_list = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
d_list = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]

# radians
alpha_list = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
theta_list = [0] * len(alpha_list)

DH_PARAMS = np.array([a_list, d_list, alpha_list, theta_list]).T

BASE_FRAME = "base"
FRAMES = ["fr3_link0", "fr3_link1", "fr3_link2", "fr3_link3", "fr3_link4", "fr3_link5", "fr3_link6", "fr3_link7", "fr3_link8"]

def get_transform_n_to_n_plus_one(n: int, theta: float) -> NDArray:
    """
    Calculate the transform from frame n to frame n-1 using modified DH parameters.

    Args:
        n (int): The current joint index (1-based, as per DH conventions).
        theta (float): The joint angle for the current joint.

    Returns:
        NDArray: A 4x4 homogeneous transformation matrix.
    """
    # TODO: implement this function
    # note that it may be helpful to refer to documentation on modified denavit hartenberg parameters:
    # https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters#Modified_DH_parameters
    # Extract the DH parameters for joint n
    
    a = DH_PARAMS[n, 0]       # Link length
    d = DH_PARAMS[n, 1]       # Link offset
    alpha = DH_PARAMS[n, 2]   # Twist angle
    theta += DH_PARAMS[n, 3]  # Joint angle (includes theta offset in modified DH parameters)

    a_prev = DH_PARAMS[n, 0]  # Link length
    d_prev = DH_PARAMS[n, 1]  # Link offset
    alpha_prev = DH_PARAMS[n, 2]  # Twist angle
    theta_prev = DH_PARAMS[n, 3]  # Joint angle (includes theta offset in modified DH parameters)
    # Compute the transformation matrix using the modified DH convention
    transform_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0, a_prev],
        [np.sin(theta) * np.cos(alpha_prev), np.cos(theta) * np.cos(alpha_prev), -np.sin(alpha_prev), -d * np.sin(alpha_prev)],
        [np.sin(theta) * np.sin(alpha_prev), np.cos(theta) * np.sin(alpha_prev), np.cos(alpha_prev), d * np.cos(alpha_prev)],
        [0, 0, 0, 1]
    ])

    return transform_matrix



class ForwardKinematicCalculator(Node):

    def __init__(self):
        super().__init__('fk_calculator')

        # TODO: create a subscriber to joint states, can you find which topic
        # this publishes on by using ros2 topic list while running the example?
        self.joint_sub = self.create_subscription(
            JointState,  # Message type
            '/joint_states',  # Topic name
            self.publish_transforms,  # Callback function
            10  # QoS depth
        )

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Set the prefix for the frame names
        self.prefix = "my_robot/"

    def publish_transforms(self, msg: JointState):
        """
        Publish transforms for all robot links using Modified DH parameters.
        """
        self.get_logger().debug(str(msg))

        # Loop through all frames from base to end-effector
        for i in range(len(FRAMES)):

            if i == 0:
                # The base frame has no parent
                parent_id = self.prefix + BASE_FRAME
                frame_id = self.prefix + FRAMES[i]
            else:
                parent_id = self.prefix + FRAMES[i-1]
                frame_id = self.prefix + FRAMES[i]
            # if i == 0:
            #     # The base frame has no parent
            #     parent_id = BASE_FRAME
            # else:
            #     # Parent link of fr3_link1 is fr3_link0, and so on
            #     parent_id = self.prefix + FRAMES[i - 1]
            # if i != len(FRAMES) - 1:
            #     # Parent link of fr3_link1 is fr3_link0, and so on
            #     parent_id = self.prefix + FRAMES[i]
            # else:
            #     # Parent of the last link is the previous link
            #     parent_id = self.prefix + FRAMES[i - 1]

            # Determine the joint angle (theta) for the current link
            if i < len(FRAMES) - 1 and i != 0:
                theta = msg.position[i]
            else:
                theta = 0  # Static link

            # Create a TransformStamped message
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = parent_id
            t.child_frame_id = frame_id

            if i == 0:
                transform = np.eye(4)
            else:
                # Compute the Modified DH transform matrix
                transform = get_transform_n_to_n_plus_one(i-1, theta)

            # Extract translation and rotation
            translation = transform[:3, 3]
            rotation_matrix = transform[:3, :3]
            quat = rotmat2q(rotation_matrix)  # Convert rotation matrix to quaternion

            # TODO: set the translation and rotation in the message we have created
            # you can check the documentation for the message type for ros2
            # to see what members it has

            # Set the translation and rotation in the TransformStamped message
            t.transform.translation.x = translation[0]
            t.transform.translation.y = translation[1]
            t.transform.translation.z = translation[2]

            t.transform.rotation.x = quat.x
            t.transform.rotation.y = quat.y
            t.transform.rotation.z = quat.z
            t.transform.rotation.w = quat.w

            # Publish the transform
            self.tf_broadcaster.sendTransform(t)

    



def main(args=None):
    # TODO: initialize our class and start it spinning
    # this example may be helpful: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html#write-the-subscriber-node
    rclpy.init(args=args)

    # Create an instance of the ForwardKinematicCalculator class
    fk_calculator = ForwardKinematicCalculator()

    # Spin the node to process callbacks
    try:
        rclpy.spin(fk_calculator)  # This keeps the node active, listening for messages
    except KeyboardInterrupt:
        fk_calculator.get_logger().info("Keyboard interrupt detected. Shutting down.")

    # Destroy the node explicitly (optional, ensures clean shutdown)
    fk_calculator.destroy_node()

    # Shutdown the ROS 2 Python client library
    rclpy.shutdown()


if __name__ == '__main__':
    main()