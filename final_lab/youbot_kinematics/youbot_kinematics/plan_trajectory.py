import rclpy
from rclpy.node import Node
from scipy.linalg import expm
from scipy.linalg import logm
from itertools import permutations
import time
import threading
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker

import numpy as np
from youbot_kinematics.youbotKineStudent import YoubotKinematicStudent
from youbot_kinematics.target_data import TARGET_JOINT_POSITIONS


class YoubotTrajectoryPlanning(Node):
    def __init__(self):
        # Initialize node
        super().__init__('youbot_trajectory_planner')

        # Save question number for check in main run method
        self.kdl_youbot = YoubotKinematicStudent()

        # Create trajectory publisher and a checkpoint publisher to visualize checkpoints
        self.traj_pub = self.create_publisher(JointTrajectory, '/EffortJointInterface_trajectory_controller/command',
                                        5)
        self.checkpoint_pub = self.create_publisher(Marker, "checkpoint_positions", 100)

    def run(self):
        """This function is the main run function of the class. When called, it runs question 6 by calling the q6()
        function to get the trajectory. Then, the message is filled out and published to the /command topic.
        """
        print("run q6a")
        self.get_logger().info("Waiting 5 seconds for everything to load up.")
        time.sleep(2.0)
        traj = self.q6()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ["arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"]
        self.traj_pub.publish(traj)

    def q6(self):
        """ This is the main q6 function. Here, other methods are called to create the shortest path required for this
        question. Below, a general step-by-step is given as to how to solve the problem.
        Returns:
            traj (JointTrajectory): A list of JointTrajectory points giving the robot joint positions to achieve in a
            given time period.
        """
        # TODO: implement this
        # Steps to solving Q6.
        # 1. Load in targets from the bagfile (checkpoint data and target joint positions).
        # 2. Compute the shortest path achievable visiting each checkpoint Cartesian position.
        # 3. Determine intermediate checkpoints to achieve a linear path between each checkpoint and have a full list of
        #    checkpoints the robot must achieve. You can publish them to see if they look correct. Look at slides 39 in lecture 7
        # 4. Convert all the checkpoints into joint values using an inverse kinematics solver.
        # 5. Create a JointTrajectory message.

        # Your code starts here ------------------------------
        target_cart_tf, target_joint_position = self.load_targets()
        sorted_order, min_dist = self.get_shortest_path(target_cart_tf)
        num_points = 5
        full_checkpoint_tfs = self.intermediate_tfs(sorted_order, target_cart_tf, num_points)
        init_joint_position = target_joint_position[:, 0]
        q_checkpoints = self.full_checkpoints_to_joints(full_checkpoint_tfs, init_joint_position)
        # Create the JointTrajectory message
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ["arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"]
        
        # Define the duration for each point
        duration_per_point = 0.1
        # Create the trajectory points
        for i in range(q_checkpoints.shape[1]):
            point = JointTrajectoryPoint()
            point.positions = q_checkpoints[:, i].tolist()
            point.time_from_start = rclpy.duration.Duration(seconds=i * duration_per_point).to_msg()
            traj.points.append(point)

        self.publish_traj_tfs(full_checkpoint_tfs)
        # Your code ends here ------------------------------

        assert isinstance(traj, JointTrajectory)
        return traj

    def load_targets(self):
        """This function loads the checkpoint data from the TARGET_JOINT_POSITIONS variable. In this variable you will find each
        row has target joint positions. You need to use forward kinematics to get the goal end-effector position.
        Returns:
            target_cart_tf (4x4x5 np.ndarray): The target 4x4 homogenous transformations of the checkpoints found in the
            bag file. There are a total of 5 transforms (4 checkpoints + 1 initial starting cartesian position).
            target_joint_positions (5x5 np.ndarray): The target joint values for the 4 checkpoints + 1 initial starting
            position.
        """
        num_target_positions = len(TARGET_JOINT_POSITIONS)
        self.get_logger().info(f"{num_target_positions} target positions")
        # Initialize arrays for checkpoint transformations and joint positions
        target_joint_positions = np.zeros((5, num_target_positions + 1))
        # Create a 4x4 transformation matrix, then stack 6 of these matrices together for each checkpoint
        target_cart_tf = np.zeros((4, 4, num_target_positions + 1))
        # Get the current starting position of the robot
        target_joint_positions[:, 0] = self.kdl_youbot.current_joint_position
        # Initialize the first checkpoint as the current end effector position
        target_cart_tf[:, :, 0] = self.kdl_youbot.forward_kinematics(target_joint_positions[0, :].tolist())

        # TODO: populate the transforms in the target_cart_tf object
        # populate the joint positions in the target_joint_positions object
        # Your code starts here ------------------------------
        for i in range(4):
            # Get the target joint positions
            target_joint_positions[i+1, :] = TARGET_JOINT_POSITIONS[i]
            # Get the target cartesian positions
            target_cart_tf[:, :, i+1] = self.kdl_youbot.forward_kinematics(target_joint_positions[i+1,:].tolist())
        # Your code ends here ------------------------------
        self.get_logger().info(f"{target_cart_tf.shape} target poses")
        assert isinstance(target_cart_tf, np.ndarray)
        assert target_cart_tf.shape == (4, 4, num_target_positions + 1)
        assert isinstance(target_joint_positions, np.ndarray)
        assert target_joint_positions.shape == (5, num_target_positions + 1)

        return target_cart_tf, target_joint_positions
        

    def get_shortest_path(self, checkpoints_tf):
        """This function takes the checkpoint transformations and computes the order of checkpoints that results
        in the shortest overall path.
        Args:
            checkpoints_tf (np.ndarray): The target checkpoints transformations as a 4x4x5 numpy ndarray.
        Returns:
            sorted_order (np.array): An array of size 5 indicating the order of checkpoint
            min_dist:  (float): The associated distance to the sorted order giving the total estimate for travel
            distance.
        """
        num_checkpoints = checkpoints_tf.shape[2] # 5 checkpoints for this instance.
        # TODO: implement this method. Make it flexible to accomodate different numbers of targets.
        # Your code starts here ------------------------------
        # extract the position component of the transformation matrix
        position = checkpoints_tf[:3, -1, :] # 3x5
        # Compute the distance between each pairs of checkpoints
        dist = np.zeros((num_checkpoints, num_checkpoints))
        for i in range(num_checkpoints):
            for j in range(num_checkpoints):
                # Compute the distance between any two checkpoints
                dist[i, j] = np.linalg.norm(position[:, i] - position[:, j])
        # Sort the shortest distance and permutation here
        min_dist = np.inf # this is for the first comparison
        # IT HAS TO BE THE FIRST CHECKPOINT!!!
        for perm in permutations(range(1, num_checkpoints)):
            # Compute the total distance of the permutation
            full_path = (0,) + perm
            # calculate the total distance
            total_dist = 0
            for k in range(num_checkpoints-1):
                total_dist += dist[full_path[k], full_path[k+1]]
            if total_dist < min_dist:
                min_dist = total_dist
                sorted_order = np.array(full_path)
            

        # Your code ends here ------------------------------
        assert isinstance(sorted_order, np.ndarray)
        assert sorted_order.shape == (num_checkpoints,)
        assert isinstance(min_dist, float)

        return sorted_order, min_dist

    def publish_traj_tfs(self, tfs):
        """This function gets a np.ndarray of transforms and publishes them in a color coded fashion to show how the
        Cartesian path of the robot end-effector.
        Args:
            tfs (np.ndarray): A array of 4x4xn homogenous transformations specifying the end-effector trajectory.
        """
        id = 0
        for i in range(0, tfs.shape[2]):
            marker = Marker()
            marker.id = id
            id += 1
            marker.header.frame_id = 'base_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0 + id * 0.05
            marker.color.b = 1.0 - id * 0.05
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = tfs[0, -1, i]
            marker.pose.position.y = tfs[1, -1, i]
            marker.pose.position.z = tfs[2, -1, i]
            self.checkpoint_pub.publish(marker)

    def intermediate_tfs(self, sorted_checkpoint_idx, target_checkpoint_tfs, num_points):
        """This function takes the target checkpoint transforms and the desired order based on the shortest path sorting, 
        and calls the decoupled_rot_and_trans() function.
        Args:
            sorted_checkpoint_idx (list): List describing order of checkpoints to follow.
            target_checkpoint_tfs (np.ndarray): the state of the robot joints. In a youbot those are revolute
            num_points (int): Number of intermediate points between checkpoints.
        Returns:
            full_checkpoint_tfs: 4x4x(4xnum_points + 5) homogeneous transformations matrices describing the full desired
            poses of the end-effector position.
        """
        # TODO: implement this
        # Your code starts here ------------------------------
        # Calculate the number of checkpoints
        num_checkpoints = len(sorted_checkpoint_idx)
        full_checkpoint_tfs = []
        for i in range(num_checkpoints - 1):
            # Get the indicies for the current and next checkpoint
            checkpoint_a_idx = sorted_checkpoint_idx[i]
            checkpoint_b_idx = sorted_checkpoint_idx[i + 1]
            # Extract the transformations for the 2 checkpoints
            checkpoint_a_tf = target_checkpoint_tfs[:, :, checkpoint_a_idx]
            checkpoint_b_tf = target_checkpoint_tfs[:, :, checkpoint_b_idx]
            # Interpolate between the 2 checkpoints
            interpolated_tfs = self.decoupled_rot_and_trans(checkpoint_a_tf, checkpoint_b_tf, num_points)
            full_checkpoint_tfs.append(interpolated_tfs) # Each element is a 4x4xnum_points matrix
        # Concatenate all interpolated poses into a single matrix
        full_checkpoint_tfs = np.concatenate(full_checkpoint_tfs, axis=2)
        # for i in range(full_checkpoint_tfs.shape[2]):
        #     print(f"Checkpoint {i}: {full_checkpoint_tfs[:3, -1, i]}")
        # Add the Last checkpoint
        final_checkpoint_tf = target_checkpoint_tfs[:, :, sorted_checkpoint_idx[-1]]
        full_checkpoint_tfs = np.concatenate((full_checkpoint_tfs, final_checkpoint_tf[:, :, np.newaxis]), axis=2)
        # Your code ends here ------------------------------
        return full_checkpoint_tfs

    def decoupled_rot_and_trans(self, checkpoint_a_tf, checkpoint_b_tf, num_points):
        """This function takes two checkpoint transforms and computes the intermediate transformations
        that follow a straight line path by decoupling rotation and translation.
        Args:
            checkpoint_a_tf (np.ndarray): 4x4 transformation describing pose of checkpoint a.
            checkpoint_b_tf (np.ndarray): 4x4 transformation describing pose of checkpoint b.
            num_points (int): Number of intermediate points between checkpoint a and checkpoint b.
        Returns:
            tfs: 4x4x(num_points) homogeneous transformations matrices describing the full desired
            poses of the end-effector position from checkpoint a to checkpoint b following a linear path.
        """
        # TODO: implement this
        # Your code starts here ------------------------------
        # Extract the translation and rotation part respectively
        translation_a = checkpoint_a_tf[:3, -1]
        translation_b = checkpoint_b_tf[:3, -1]
        rotation_a = checkpoint_a_tf[:3, :3]
        rotation_b = checkpoint_b_tf[:3, :3]

        # Compute the relative rotation
        relative_rotation = np.dot(np.linalg.inv(rotation_a), rotation_b)
        log_relative_rotation = logm(relative_rotation)

        # Initialize storages for the intermediate transformations
        tfs = np.zeros((4, 4, num_points))
        for i in range(num_points):
            # Compute the intermediate translation
            t = i / (num_points)
            # Interpolate the translation component
            interpolated_translation = (1 - t) * translation_a + t * translation_b
            # Interpolate the rotation component
            interpolated_rotation = np.dot(rotation_a, expm(t * log_relative_rotation))

            # Construct the homogeneous transformation matrix
            tfs[:3, :3, i] = interpolated_rotation
            tfs[:3, -1, i] = interpolated_translation
            tfs[-1, -1, i] = 1
        # Your code ends here ------------------------------
        return tfs

    def full_checkpoints_to_joints(self, full_checkpoint_tfs, init_joint_position):
        """This function takes the full set of checkpoint transformations, including intermediate checkpoints, 
        and computes the associated joint positions by calling the ik_position_only() function.
        Args:
            full_checkpoint_tfs (np.ndarray, 4x4xn): 4x4xn transformations describing all the desired poses of the end-effector
            to follow the desired path.
            init_joint_position (np.ndarray):A 5x1 array for the initial joint position of the robot.
        Returns:
            q_checkpoints (np.ndarray, 5xn): For each pose, the solution of the position IK to get the joint position
            for that pose.
        """
        # TODO: Implement this
        # Your code starts here ------------------------------
        # Firstly, get the starting point as the initial guess for first checkpoint
        q0 = init_joint_position
        checkpoint_num = full_checkpoint_tfs.shape[2]
        # Handle the case where no checkpoint provided
        if checkpoint_num == 0:
            self.get_logger().warning("No checkpoints provided")
            return np.zeros((5, 0))
        

        q_checkpoints = np.zeros((5, checkpoint_num))
        for i in range(checkpoint_num):
            try:
                # Solve the inverse kinematics for the current pose
                q, error = self.ik_position_only(full_checkpoint_tfs[:, :, i], q0)
                # Log the error values
                self.get_logger().info(f"Checkpoint {i} error: {error}")
                # Update the joint configurations
                q_checkpoints[:, i] = q

                # Update the initial guess for the next checkpoint
                q0 = q
            except Exception as e:
                self.get_logger().error(f"Failed to solve IK for checkpoint {i}: {e}")
                break
        # Your code ends here ------------------------------
        return q_checkpoints

    def ik_position_only(self, pose, q0, lam=0.25, num=500, threshold=1e-3, max_lam = 10):
        """This function implements position only inverse kinematics.
        Args:
            pose (np.ndarray, 4x4): 4x4 transformations describing the pose of the end-effector position.
            q0 (np.ndarray, 5x1):A 5x1 array for the initial starting point of the algorithm.
            lam (float): The damping factor for the damped least squares method.
            num (int): The maximum number of iterations to run the algorithm.
            threshold (float): The threshold for the error to determine convergence.
            max_lam (float): The maximum damping factor to prevent infinite adjustment.
        Returns:
            q (np.ndarray, 5x1): The IK solution for the given pose.
            error (float): The Cartesian error of the solution.
        """

        # TODO: Implement this
        # Some useful notes:
        # We are only interested in position control - take only the position part of the pose as well as elements of the
        # Jacobian that will affect the position of the error.

        # Your code starts here ------------------------------
        # Start with the initial guess
        q = q0.copy()
        # Extract the target position from the pose
        p_desired = pose[:3, -1]

        for i in range(num):
            # Compute the current end-effector position
            T_current = self.kdl_youbot.forward_kinematics(q.tolist())
            p_current = T_current[:3, -1]

            # Compute the error
            error = p_desired - p_current

            # Check convergence
            if np.linalg.norm(error) < threshold:
                return q, np.linalg.norm(error)
            # Calculate the jacobian
            J = self.kdl_youbot.get_jacobian(q)

            retry_count = 0
            max_retries = 5
            J_translation = J[:3, :]
            while retry_count < max_retries:
                # Check the singularity
                singularity = self.kdl_youbot.check_singularity(J)
                if not singularity:
                    # Reduce damping if not singular
                    lam = max(lam / 1.5, 0.25)
                    break
                # If singular, increase damping
                self.get_logger().info("Singularity reached, increasing damping")
                lam = min(lam * 2, max_lam)
                retry_count += 1
            
            if retry_count == max_retries:
                self.get_logger().info("Singularity reached 5 times, proceeding with current damping")

            # Compute the change in joint angle with damped least squares
            dq = np.linalg.pinv(J_translation.T.dot(J_translation) + lam * np.eye(5)).dot(J_translation.T).dot(error)
            # Update the joint angles
            q += dq
        # Your code ends here ------------------------------
        return q, np.linalg.norm(error)


def main(args=None):
    rclpy.init(args=args)

    youbot_planner = YoubotTrajectoryPlanning()

    youbot_planner.run()

    rclpy.spin(youbot_planner)


    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    youbot_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()