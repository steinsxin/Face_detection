import traceback
import pickle
import time
import json
from typing import List, Tuple, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from autolife_robot_sdk.utils import get_wifi_mac_address


DEVICE_ID = get_wifi_mac_address("enP8p1s0").replace(":", "")
print(DEVICE_ID)

class RobotTargetEEPoseDemo(Node):
    """
    A ROS2 node to demonstrate robot target end-effector pose control.
    """

    def __init__(self):
        """
        Initialize the RobotTargetEEPoseDemo node.
        """
        super().__init__('robot_target_ee_pose_demo')
        self.robot_target_ee_pose_pub = self.create_publisher(String, f'/robot_target_ee_pose{DEVICE_ID}', 10)
        self.publisher = self.create_publisher(String, "control_topic_" + DEVICE_ID, 10)
        self.control_reset_pub = self.create_publisher(String, "control_reset_" + DEVICE_ID, 10)

        # Initialize joint angles to zero
        self.target_left_arm_joints_deg = [0] * 11
        self.target_right_arm_joints_deg = [0] * 11

        self.robot_state = 0

        self.get_logger().info("Robot Target End-Effector Pose Demo Initialized")
        self.publish_joint_angles()

    def publish_joint_angles(self):
        """
        Publish the current joint angles.
        """
        message = String()
        message.data = json.dumps({
            "l": self.target_left_arm_joints_deg,
            "r": self.target_right_arm_joints_deg,
            "s": self.robot_state
        })
        self.robot_target_ee_pose_pub.publish(message)

    def set_joint_angles(self, joint_index: int, left_angle: float, right_angle: float):
        """
        Set the target angles for a specific joint.

        :param joint_index: The index of the joint (0-10).
        :param left_angle: The target angle for the left arm joint.
        :param right_angle: The target angle for the right arm joint.
        """
        if 0 <= joint_index < 11:
            self.target_left_arm_joints_deg[joint_index] = left_angle
            self.target_right_arm_joints_deg[joint_index] = right_angle
            self.publish_joint_angles()
            self.get_logger().info(f"Joint {joint_index + 1} set to left: {left_angle}°, right: {right_angle}°")
        else:
            self.get_logger().error(f"Invalid joint index: {joint_index}")

    def reset_joint_angles(self, duration: float = 2.5, steps: int = 50):
        """
        平滑地将所有关节角度归零。
        
        :param duration: 插值持续时间（秒）
        :param steps: 插值步数
        """
        joint_targets = []
        for joint_id in range(11):
            current_left = self.target_left_arm_joints_deg[joint_id]
            current_right = self.target_right_arm_joints_deg[joint_id]
            joint_targets.append((joint_id, 0.0, 0.0))  # 目标都是 0

        self.smooth_set_multiple_joints(joint_targets, duration=duration, steps=steps)
        self.get_logger().info("All joints smoothly reset to home position")

    def smooth_set_multiple_joints(
        self,
        joint_targets: List[Tuple[int, float, float]],  # (joint_id, left_target, right_target)
        duration: float = 3.0,
        steps: int = 50
    ):
        """
        平滑地同步控制多个关节的左右角度。

        :param joint_targets: [(joint_id, left_target, right_target), ...]
        :param duration: 持续时间（秒）
        :param steps: 插值步数
        """
        # 获取每个关节的初始角度
        start_angles = []
        for joint_id, _, _ in joint_targets:
            start_left = self.target_left_arm_joints_deg[joint_id]
            start_right = self.target_right_arm_joints_deg[joint_id]
            start_angles.append((start_left, start_right))

        interval = duration / steps

        for i in range(steps + 1):
            for idx, (joint_id, target_left, target_right) in enumerate(joint_targets):
                start_left, start_right = start_angles[idx]
                t = i / steps
                # 可替换为更平滑插值函数（如 smoothstep）
                left_angle = start_left + (target_left - start_left) * t
                right_angle = start_right + (target_right - start_right) * t
                self.set_joint_angles(joint_id, left_angle, right_angle)
            time.sleep(interval)


    def smooth_set_multiple_wg_joints(
        self,
        joint_targets: List[Tuple[int, float]],  # (joint_id, target_angle)
        duration: float = 3.0,
        steps: int = 50
    ):
        """
        平滑地同步控制多个腰腿关节角度（仅左边，右边固定为0）。

        :param joint_targets: [(joint_id, target_angle), ...]
        :param duration: 总持续时间（秒）
        :param steps: 插值步数
        """
        # 获取每个关节的起始角度
        start_angles = []
        for joint_id, _ in joint_targets:
            start_angle = self.target_left_arm_joints_deg[joint_id]
            start_angles.append(start_angle)

        interval = duration / steps

        for i in range(steps + 1):
            t = i / steps
            for idx, (joint_id, target_angle) in enumerate(joint_targets):
                start_angle = start_angles[idx]
                # 线性插值
                current_angle = start_angle + (target_angle - start_angle) * t
                self.set_joint_angles(joint_id, current_angle, 0)  # 右边为0
            time.sleep(interval)

    def bow_salute(self):
        self.get_logger().info("Running Robot Target End-Effector Pose Demo")
        time.sleep(1)

        self.smooth_set_multiple_joints([
            (4, 50, 0),
            (6, -70, 0),
            (7, 110, 0),
            (8, 40, 0)
        ], duration=3.0)

        self.smooth_set_multiple_wg_joints([
            (2, 30)
        ], duration=3.0)

        time.sleep(1)

        self.reset_joint_angles()
        time.sleep(2)

        self.get_logger().info("Finished Demo")

    def wave(self, number):
        """
        Run the robot target wave pose demo.
        """
        self.get_logger().info("Running Robot Target Wave Pose Demo")
        time.sleep(1)

        # 手臂抬前 + 手臂收回（整组动作）
        self.smooth_set_multiple_joints([
            (4, 50, 0),
            (7, 120, 0)
        ], duration=3.0)
        
        time.sleep(2)
        # 动态挥手动作：左右摆动 number 次
        steps = 50
        duration = 2.0
        interval = duration / steps

        for _ in range(number):
            # 第一步：9号从 -40 → 40，6号从 40 → -40
            for i in range(steps + 1):
                angle_9 = -20 + (40 * i / steps)   # -40 到 40
                angle_6 = 20 - (40 * i / steps)    # 40 到 -40
                self.set_joint_angles(9, angle_9, 0)
                self.set_joint_angles(6, angle_6, 0)
                time.sleep(interval)

            # 第二步：9号从 40 → -40，6号从 -40 → 40
            for i in range(steps + 1):
                angle_9 = 20 - (40 * i / steps)    # 40 到 -40
                angle_6 = -20 + (40 * i / steps)   # -40 到 40
                self.set_joint_angles(9, angle_9, 0)
                self.set_joint_angles(6, angle_6, 0)
                time.sleep(interval)

        # 手腕复位（平滑）
        self.smooth_set_multiple_joints([
            (9, 0, 0),
            (6, 0, 0)
        ], duration=2.0)

        time.sleep(1)

        # 重置关节角度（整体复位）
        self.reset_joint_angles()
        self.get_logger().info("Finished Demo")

    def right_hand_movement(self):
        self.get_logger().info("Running Robot Target Wave Pose Demo")
        time.sleep(1)

        self.smooth_set_multiple_joints([
            (4, 0, -50),
            (7, 0, -40),
            (9, 0, -40)
        ], duration=3.0)
        
        self.smooth_set_multiple_joints([
            (6, 0, 70),
            (7, 0, -90),
            (8, 0, -40),
            (9, 0, 0)
        ], duration=3.0)

        self.smooth_set_multiple_joints([
            (5, 0, -50),
            (6, 0, -60),
            (7, 0, -40),
            (8, 0, 0),
            (9, 0, 0)
        ], duration=3.0)
        
        time.sleep(1)

        self.reset_joint_angles()
        time.sleep(2)

        self.get_logger().info("Finished Demo")

    def guide(self):
        """
        Run the robot target Guide pose demo.
        """
        self.get_logger().info("Running Robot Target Guide Pose Demo")
        time.sleep(1)

        self.smooth_set_multiple_joints([
            (4, 30, 30),
            (7, 50, 0),
            (9, 20, -60),
            (10, -20, 0)
        ], duration=3.0)

        self.smooth_set_multiple_wg_joints([
            (2, 15)
        ], duration=2.0)

        self.smooth_set_multiple_joints([
            (9, 20, 0)
        ], duration=1.0)

        self.reset_joint_angles()
        time.sleep(2)
        self.get_logger().info("Finished Demo")

    def please_say_hello(self):
        self.get_logger().info("Running Robot Target please_say_hello Pose Demo")
        time.sleep(1)

        self.smooth_set_multiple_joints([
            (4, 0, -50),
            (6, 0, 70),
            (7, 0, -110),
            (8, 0, -40)
        ], duration=3.0)

        self.smooth_set_multiple_wg_joints([
            (0, 30),
            (1, 40),
            (2, 10)
        ], duration=3.0)
        time.sleep(2)

        self.reset_joint_angles()
        time.sleep(2)
        self.get_logger().info("Finished Demo")

    def control_reset(self) -> None:
        """Publish a control reset message."""
        msg = String()
        json_msg = {"type": "control_reset", "status": "async"}
        msg.data = json.dumps(json_msg)
        self.control_reset_pub.publish(msg)
    
    def replay(self, filepath: str) -> None:
        """Replay messages from a recorded pickle file.

        Args:
            filepath: Path to the pickle file containing recorded messages.
        """
        msg_list = self.load_pickle(filepath)
        if msg_list:
            self.replay_messages(msg_list)

    def load_pickle(self, filepath: str) -> List[Tuple[float, str]]:
        """Load message data from a pickle file.

        Args:
            filepath: Path to the pickle file.

        Returns:
            List of tuples containing (timestamp, message_data).
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.get_logger().info(f"Successfully loaded recording: {filepath}")
                return data
        except Exception as e:
            self.get_logger().error(f"Failed to load recording: {e}")
            return []

    def replay_messages(self, msg_list: List[Tuple[float, str]]) -> None:
        """Replay messages with original timing.

        Args:
            msg_list: List of tuples containing (timestamp, message_data).
        """
        prev_time = 0.0
        start_time = time.time()

        self.get_logger().info("Arm Test Start")
        for index, item in enumerate(msg_list):
            current_time, msg_data = item

            # Calculate wait time
            wait_time = current_time if index == 0 else current_time - prev_time

            wait_time_start = time.time()
            while time.time() - wait_time_start < wait_time:
                time.sleep(0.001)

            # Publish message
            msg = String()
            try:
                command = json.loads(msg_data)
                command['ts'] = time.time() * 1000
                msg.data = json.dumps(command)
                self.publisher.publish(msg)
            except json.JSONDecodeError as e:
                self.get_logger().error(f"Failed to decode JSON: {e}")
                continue

            prev_time = current_time

        self.get_logger().info("Replay completed")


def main() -> None:
    """
    Main function to run the RobotTargetEEPoseDemo node.
    """
    rclpy.init()
    robot_target_ee_pose_demo = RobotTargetEEPoseDemo()

    try:
        # robot_target_ee_pose_demo.bow_salute()
        robot_target_ee_pose_demo.please_say_hello()
        # robot_target_ee_pose_demo.reset_joint_angles()

        # robot_target_ee_pose_demo.wave(2)
        # robot_target_ee_pose_demo.guide()
        # time.sleep(1)
        # robot_target_ee_pose_demo.replay("../pkl/command_hello.pkl")
    except KeyboardInterrupt:
        robot_target_ee_pose_demo.get_logger().info("Quit by interrupt")
    except Exception as e:
        robot_target_ee_pose_demo.get_logger().error(f"Unexpected exception during playback: {e}")
        traceback.print_exc()
    finally:
        robot_target_ee_pose_demo.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()