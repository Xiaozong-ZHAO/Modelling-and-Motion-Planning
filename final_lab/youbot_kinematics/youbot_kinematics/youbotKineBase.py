import rclpy
from rclpy.node import Node
import numpy as np
from numpy.typing import NDArray

from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Quaternion
from trajectory_msgs.msg import JointTrajectory

# TODO: Ensure this library is findable and the method is implemented
from transform_helpers.utils import rotmat2q

class YoubotKinematicBase(Node):
    def __init__(self, tf_suffix=''):
        super().__init__('youbot_kinematic_base')
        # Robot variables
        # Identify class used when broadcasting tf with a suffix
        self.tf_suffix = tf_suffix
	
        youbot_dh_parameters = {'a': [-0.033, 0.155, 0.135, +0.002, 0.0],
                                'alpha': [np.pi / 2, 0.0, 0.0, np.pi / 2, np.pi],
                                'd': [0.145, 0.0, 0.0, 0.0, -0.185],
                                'theta': [np.pi, np.pi / 2, 0.0, -np.pi / 2, np.pi]}
        
        self.dh_params = youbot_dh_parameters.copy()

        # Set current joint position
        self.current_joint_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Set joint limits
        self.joint_limit_min = np.array([-169 * np.pi / 180, -65 * np.pi / 180, -150 * np.pi / 180,
                                         -102.5 * np.pi / 180, -167.5 * np.pi / 180])
        self.joint_limit_max = np.array([169 * np.pi / 180, 90 * np.pi / 180, 146 * np.pi / 180,
                                         102.5 * np.pi / 180, 167.5 * np.pi / 180])
                                         

        # ROS related
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            5)
        self.joint_state_sub  # prevent unused variable warning
        self.traj_publisher = self.create_publisher(JointTrajectory, '/EffortJointInterface_trajectory_controller/command', 5)
        self.traj_publisher
        # Initialize the transform broadcaster
        self.pose_broadcaster = TransformBroadcaster(self)


    def joint_state_callback(self, msg):
        """ ROS callback function for joint states of the robot. Broadcasts the current pose of end effector.

        Args:
            msg (JointState): Joint state message containing current robot joint position.

        """
        self.current_joint_position = list(msg.position)
        current_pose = self.forward_kinematics(self.current_joint_position)
        self.broadcast_pose(current_pose)

    def broadcast_pose(self, pose):
        """Given a pose transformation matrix, broadcast the pose to the TF tree.

        Args:
            pose (np.ndarray): Transformation matrix of pose to broadcast.

        """
        transform = TransformStamped()

        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'base_link'
        transform.child_frame_id = 'arm_end_effector_' + self.tf_suffix

        transform.transform.translation.x = pose[0, 3]
        transform.transform.translation.y = pose[1, 3]
        transform.transform.translation.z = pose[2, 3]
        transform.transform.rotation = rotmat2q(pose[:3, :3])

        self.pose_broadcaster.sendTransform(transform)

    def forward_kinematics(self, joint_readings, up_to_joint=5):
        """This function solves forward kinematics by multiplying frame transformation up until a specified
        frame number. The frame transformation used in the computation are derived from dh parameters found in the
        init method and joint_readings.
        Args:
            joint_readings (list): the state of the robot joints. In a youbot those are revolute
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematics.
                Defaults to 5.
        """
        # validate input
        if up_to_joint < 1 or up_to_joint > len(joint_readings):
            raise ValueError("up_to_joint must be between or no more than the number of joints")
        if len(joint_readings) < up_to_joint:
            raise ValueError("joint number insufficient")
        
        # Get the DH parameter from the object
        dh_params = self.dh_params

        # We assume the initial base position is identity matrix
        T = np.eye(4)

        for i in range(up_to_joint):
            a = dh_params['a'][i]
            alpha = dh_params['alpha'][i]
            d = dh_params['d'][i]
            theta = dh_params['theta'][i] + joint_readings[i]

            A = self.standard_dh(a, alpha, d, theta)
            # Update the base position and orientation
            T = T.dot(A)

        # convert the transformation matrix to rorigues vector
        p = self.rotmat2rodrigues(T)
        return p

    def get_jacobian(self, joint):
        """Compute Jacobian given the robot joint values. Implementation found in child classes.

        Args:
            joint (list): the state of the robot joints. In a youbot those are revolute.
        Returns: Jacaobian matrix.

        """
        raise NotImplementedError

    @staticmethod
    def standard_dh(a, alpha, d, theta):
        """This function computes the homogeneous 4x4 transformation matrix T_i based on the four standard DH parameters
         associated with link i and joint i.
        Args:
            a ([int, float]): Link Length. The distance along x_i ( the common normal) between z_{i-1} and z_i
            alpha ([int, float]): Link twist. The angle between z_{i-1} and z_i around x_i.
            d ([int, float]): Link Offset. The distance along z_{i-1} between x_{i-1} and x_i.
            theta ([int, float]): Joint angle. The angle between x_{i-1} and x_i around z_{i-1}
        Returns:
            [np.ndarray]: the 4x4 transformation matrix T_i describing  a coordinate transformation from
            the concurrent coordinate system i to the previous coordinate system i-1
        """
        assert isinstance(a, (int, float)), "wrong input type for a"
        assert isinstance(alpha, (int, float)), "wrong input type for =alpha"
        assert isinstance(d, (int, float)), "wrong input type for d"
        assert isinstance(theta, (int, float)), "wrong input type for theta"
        # Initialize the transformation matrix
        A = np.zeros((4, 4))
        # TODO: implement a method to get the transform matrix using DH Parameters

        # The first row
        A[0, 0] = np.cos(theta)
        A[0, 1] = -np.sin(theta) * np.cos(alpha)
        A[0, 2] = np.sin(theta) * np.sin(alpha)
        A[0, 3] = a * np.cos(theta)

        # The second row
        A[1, 0] = np.sin(theta)
        A[1, 1] = np.cos(theta) * np.cos(alpha)
        A[1, 2] = -np.cos(theta) * np.sin(alpha)
        A[1, 3] = a * np.sin(theta)

        # The third row
        A[2, 0] = 0
        A[2, 1] = np.sin(alpha)
        A[2, 2] = np.cos(alpha)
        A[2, 3] = d

        # The fourth row
        A[3, 0] = 0
        A[3, 1] = 0
        A[3, 2] = 0
        A[3, 3] = 1

        # Ensure the output is of the right type and shape
        assert isinstance(A, np.ndarray), "Output wasn't of type ndarray"
        assert A.shape == (4, 4), "Output had wrong dimensions"
        return A

    def rotmat2rodrigues(self, T):
        """Convert rotation matrix to rodrigues vector. Done by first converting to quaternion then to rodrigues.

        Args:
            T (np.ndarray): Rotation matrix to convert to rodrigues representation.

        Returns:
            p (np.ndarray): An array for the first 5 elements specifying translation and the last three specifying
        rotation.
        """
        assert isinstance(T, np.ndarray)

        # TODO: Implement a method to convert from a rotation matrix to a rodrigues vector

        p = np.empty(6, float)

        # extract rotation and translation components
        rotation = T[:3, :3]
        translation = T[:3, 3]

        # Convert the rotation matrix to quaternion
        q = rotmat2q(rotation)

        # separate the quaternion to scalar and vector
        q_w = q.w
        q_vec = np.array([q.x, q.y, q.z])

        if np.isclose(q_w, 1.0):
            # if the scalar part is 1, the rotation is 0
            p[3:] = np.zeros(3)
        else:
            # convert the quaternion to rodirgues vector
            angle = 2 * np.arccos(q_w)
            axis = q_vec / np.linalg.norm(q_vec)
            p[3:] = angle * axis
        
        p[:3] = translation
        return p