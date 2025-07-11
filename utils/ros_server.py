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
    
    # 测试所有音频模式
    audio_modes = [
        ("ECHO", "ECHO模式"),
        ("SLEEP", "睡眠模式"),
        ("AI_REALTIME", "AI实时对话模式"),
    ]

    def __init__(self, stop_event=None):
        super().__init__('face_node_' + DEVICE_ID)

        self.status = "enable"
        self.time_sleep = 10

        self.robot_action_pub = self.create_publisher(
            String,
            f'robot_action_{DEVICE_ID}',
            10
        )

        self.audio_topic_pub = self.create_publisher(
            String,
            f'audio_topic_{DEVICE_ID}',
            10
        )

        self.face_detection_sub = self.create_subscription(
            String,
            f'face_detection_{DEVICE_ID}',
            self.face_detection_callback,
            10
        )

    def face_detection_callback(self, msg):
        try:
            data = json.loads(msg.data)
            status = data.get("status", "")
            time_sleep = data.get("time_sleep", 10)

            if status:
                self.status = status
            if time_sleep:
                self.time_sleep = int(time_sleep)

            print(msg.data)

        except json.JSONDecodeError:
            print(f"Failed to decode JSON from message data: {msg.data}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

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