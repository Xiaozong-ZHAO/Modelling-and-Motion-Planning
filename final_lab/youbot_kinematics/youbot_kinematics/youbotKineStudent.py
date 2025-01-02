import rclpy
import time
import threading

import numpy as np
from youbot_kinematics.youbotKineBase import YoubotKinematicBase
from youbot_kinematics.target_data import TARGET_JOINT_POSITIONS


class YoubotKinematicStudent(YoubotKinematicBase):
    def __init__(self):
        super(YoubotKinematicStudent, self).__init__(tf_suffix='student')

        # Set the offset for theta
        youbot_joint_offsets = [170.0 * np.pi / 180.0,
                                -65.0 * np.pi / 180.0,
                                146 * np.pi / 180,
                                -102.5 * np.pi / 180,
                                -167.5 * np.pi / 180]

        # Apply joint offsets to dh parameters
        self.dh_params['theta'] = [theta + offset for theta, offset in
                                   zip(self.dh_params['theta'], youbot_joint_offsets)]

        # Joint reading polarity signs
        self.youbot_joint_readings_polarity = [-1, 1, 1, 1, 1]

    def forward_kinematics(self, joints_readings, up_to_joint=5):
        """This function solve forward kinematics by multiplying frame transformation up until a specified
        frame number. The frame transformation used in the computation are derived from dh parameters and
        joint_readings.
        Args:
            joints_readings (list): the state of the robot joints. In a youbot those are revolute
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematics.
                Defaults to 5.
        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix describing the pose of frame_{up_to_joint}
                w.r.t the base of the robot.
        """
        assert isinstance(self.dh_params, dict)
        assert isinstance(joints_readings, list), "joint readings of type " + str(type(joints_readings))
        assert isinstance(up_to_joint, int)
        assert up_to_joint >= 0
        assert up_to_joint <= len(self.dh_params['a'])

        T = np.identity(4)

        # Apply offset and polarity to joint readings (found in URDF file)
        joints_readings = [sign * angle for sign, angle in zip(self.youbot_joint_readings_polarity, joints_readings)]

        for i in range(up_to_joint):
            A = self.standard_dh(self.dh_params['a'][i],
                                 self.dh_params['alpha'][i],
                                 self.dh_params['d'][i],
                                 self.dh_params['theta'][i] + joints_readings[i])
            T = T.dot(A)
            
        assert isinstance(T, np.ndarray), "Output wasn't of type ndarray"
        assert T.shape == (4, 4), "Output had wrong dimensions"
        return T

    def get_jacobian(self, joints_readings):
        """Given the joint values of the robot, compute the Jacobian matrix.

        Args:
            joint (list): the state of the robot joints. In a youbot those are revolute

        Returns:
            Jacobian (numpy.ndarray): NumPy matrix of size 6x5 which is the Jacobian matrix.
        """
        joint_num = 6
        # add polarity and offset to the theta DH component
        joints_readings = [sign * angle for sign, angle in zip(self.youbot_joint_readings_polarity, joints_readings)]
        # create a container to store the 5 transformation matrices T01 to T45
        T_all = np.zeros((4, 4, joint_num))
        # get the transformation matrix T01 to T45
        for i in range(joint_num):
            if i == 0:
                T_all[:, :, i] = np.identity(4)
                # convert z0 to [0 0 -1]
                T_all[:3, 2, i] = [0, 0, 1]
                continue
            T = self.standard_dh(self.dh_params['a'][i-1],
                                 self.dh_params['alpha'][i-1],
                                 self.dh_params['d'][i-1],
                                 self.dh_params['theta'][i-1] + joints_readings[i-1])
            # store the transformation to the variable T_all
            T_all[:, :, i] = T
        # calculate the transformation matrices frmo T02 to T04
        for i in range(1, joint_num):
            T_all[:, :, i] = T_all[:, :, i - 1].dot(T_all[:, :, i])

        # create a container to store the position of corresponding matrices in T_all
        p_all = np.zeros((3, joint_num))
        # get the position of the transformation matrices
        for i in range(joint_num):
            p_all[:, i] = T_all[:3, 3, i]

        # create a container to store the z axis of the transformation matrices
        z_all = np.zeros((3, joint_num))
        # get the z axis of the transformation matrices
        for i in range(joint_num):
            z_all[:, i] = T_all[:3, 2, i]
        # calculate the 6x5 Jacobian matrix
        jacobian = np.zeros((6, joint_num))
        for i in range(joint_num):
            if i == 0:
                jacobian[:3, i] = np.cross(-z_all[:, i], p_all[:, joint_num-1] - p_all[:, i])
                jacobian[3:, i] = -z_all[:, i]
            else:
                jacobian[:3, i] = np.cross(z_all[:, i], p_all[:, joint_num-1] - p_all[:, i])
                jacobian[3:, i] = z_all[:, i]
        jacobian = jacobian[:, :5]
        # assert isinstance(joint, list)
        # assert len(joint) == 5

        # TODO: create the jacobian matrix
        # Your code starts here ----------------------------
        # Your code ends here ------------------------------
        # assert jacobian.shape == (6, 5)
        return jacobian


    def check_singularity(self, jacobian):
        """Check for singularity condition given robot joints. Coursework 2 Question 4c.
        Reference Lecture 5 slide 30.

        Args:
            jacobian (numpy.ndarray): Jacobian matrix of size 6x5.

        Returns:
            singularity (bool): True if in singularity and False if not in singularity.

        """
        assert isinstance(jacobian, np.ndarray)
        assert jacobian.shape == (6, 5), "Jacobian matrix has to be a 6x5 matrix"

        # TODO: Implement this
        # Your code starts here ----------------------------
        # Calculate the rank of the Jacobian matrix
        rank = np.linalg.matrix_rank(jacobian)
        # Define singularity condition
        singularity = bool(rank < min(jacobian.shape))
        # Your code ends here ------------------------------
        assert isinstance(singularity, bool)
        return singularity


def main(args=None):
    rclpy.init(args=args)

    kinematic_student = YoubotKinematicStudent()

    for i in range(TARGET_JOINT_POSITIONS.shape[0]):
        target_joint_angles = TARGET_JOINT_POSITIONS[i]
        target_joint_angles = target_joint_angles.tolist()
        pose = kinematic_student.forward_kinematics(target_joint_angles)
        # we would probably compute the jacobian at our current joint angles, not the target
        # but this is just to check your work
        jacobian = kinematic_student.get_jacobian(target_joint_angles)
        singularity = kinematic_student.check_singularity(jacobian)
        print("target joint angles")
        print(target_joint_angles)
        print("pose")
        print(pose)
        print("jacobian")
        print(jacobian)
        print("singularity")
        print(singularity)

    rclpy.spin(kinematic_student)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    kinematic_student.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()