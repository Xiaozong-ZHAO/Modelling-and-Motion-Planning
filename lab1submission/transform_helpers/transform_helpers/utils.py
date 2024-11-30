from geometry_msgs.msg import Quaternion

import numpy as np
from numpy.typing import NDArray

def rotmat2q(T: NDArray) -> Quaternion:
    """
    Function that transforms a 3x3 rotation matrix to a ROS quaternion representation.

    Args:
        T (NDArray): A 3x3 rotation matrix.

    Returns:
        Quaternion: A ROS Quaternion message.
    """
    # Check if the input matrix is 3x3
    if T.shape != (3, 3):
        raise ValueError("Input rotation matrix must be 3x3.")

    # Allocate the Quaternion message
    # print the rotation matrix
    q = Quaternion()

    # Extract the elements of the rotation matrix
    m00, m01, m02 = T[0, 0], T[0, 1], T[0, 2]
    m10, m11, m12 = T[1, 0], T[1, 1], T[1, 2]
    m20, m21, m22 = T[2, 0], T[2, 1], T[2, 2]

    # Calculate the trace of the matrix
    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q.w = 0.25 / s
        q.x = (m21 - m12) * s
        q.y = (m02 - m20) * s
        q.z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        q.w = (m21 - m12) / s
        q.x = 0.25 * s
        q.y = (m01 + m10) / s
        q.z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        q.w = (m02 - m20) / s
        q.x = (m01 + m10) / s
        q.y = 0.25 * s
        q.z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        q.w = (m10 - m01) / s
        q.x = (m02 + m20) / s
        q.y = (m12 + m21) / s
        q.z = 0.25 * s
    return q