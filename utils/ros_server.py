#!/usr/bin/env python3
import json
import pickle
import time
import socket
import threading
from typing import List, Tuple, Any

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist

from autolife_robot_sdk.utils import get_wifi_mac_address, get_mac_from_ip

def _get_device_id() -> str:
    """Determine DEVICE_ID based on device type (NX or other)."""
    device_id = "00000000"
    hostname = socket.gethostname().upper()
    if "NX" in hostname:
        device_id = get_wifi_mac_address("enP8p1s0")
    else:
        device_id = get_mac_from_ip("192.168.10.2")

    return device_id

DEVICE_ID = _get_device_id()
print(f"DEVICE_ID: {DEVICE_ID}")

class RosNode(Node):
    def __init__(self, stop_event=None):
        super().__init__('face_node_' + DEVICE_ID)

        self.robot_action_pub = self.create_publisher(
            String,
            f'robot_action_{DEVICE_ID}',
            10
        )

        self.audio_text_pub = self.create_publisher(
            String,
            f'audio_text_{DEVICE_ID}',
            10
        )


def main(args=None):
    """Main function to initialize and run the ROS 2 node."""
    rclpy.init(args=args)
    node = RosNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        node.get_logger().error(f"Unhandled exception: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()