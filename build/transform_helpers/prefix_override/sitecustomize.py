import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/relogic/ros2_ws/src/COMP0246_Labs/install/transform_helpers'
