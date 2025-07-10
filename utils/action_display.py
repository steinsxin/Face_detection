import time
import json
import traceback
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

from publisher_node import PublisherNode
from receiver_node import ReceiverNode

class MotorMap:
    LEFT_MAP = {
        "Joint_Left_Shoulder_Inner": 4,
        "Joint_Left_Shoulder_Outer": 5,
        "Joint_Left_UpperArm": 6,
        "Joint_Left_Elbow": 7,
        "Joint_Left_Forearm": 8,
        "Joint_Left_Wrist_Upper": 9,
        "Joint_Left_Wrist_Lower": 10,
    }

    RIGHT_MAP = {
        "Joint_Right_Shoulder_Inner": 4,
        "Joint_Right_Shoulder_Outer": 5,
        "Joint_Right_UpperArm": 6,
        "Joint_Right_Elbow": 7,
        "Joint_Right_Forearm": 8,
        "Joint_Right_Wrist_Upper": 9,
        "Joint_Right_Wrist_Lower": 10,
    }

    WAIST_LEG_MAP = {
        "Joint_Ankle": 0,
        "Joint_Knee": 1,
        "Joint_Waist_Pitch": 2,
        "Joint_Waist_Yaw": 3,
    }

    @classmethod
    def resolve_joint(cls, joint_name: str):
        """
        返回 (target_array_name, index) 元组
        target_array_name: "left", "right", or "waist-leg"
        index: 对应的 target_joints_deg 索引
        """
        if joint_name in cls.WAIST_LEG_MAP:
            return "shared", cls.WAIST_LEG_MAP[joint_name]
        elif joint_name in cls.LEFT_MAP:
            return "left", cls.LEFT_MAP[joint_name]
        elif joint_name in cls.RIGHT_MAP:
            return "right", cls.RIGHT_MAP[joint_name]
        else:
            raise ValueError(f"[MotorMap] 未知关节名: {joint_name}")

class ActionDisplay:
    def __init__(self, pub_node=None, sub_node=None):
        self.target_left_arm_joints_deg = [0] * 11
        self.target_right_arm_joints_deg = [0] * 11
        self.robot_state = 0
        self.pub_node = pub_node
        self.sub_node = sub_node

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
        self.pub_node.robot_target_ee_pose_pub.publish(message)

    def set_joint_angle_by_name(self, joint_name: str, angle: float):
        """
        根据关节名设置角度。自动判断设置到左臂、右臂或共享（前4个）
        """
        try:
            target, index = MotorMap.resolve_joint(joint_name)
        except ValueError as e:
            print(e)
            return

        if target == "left":
            self.target_left_arm_joints_deg[index] = angle
        elif target == "right":
            self.target_right_arm_joints_deg[index] = angle
        elif target == "shared":
            self.target_left_arm_joints_deg[index] = angle
            self.target_right_arm_joints_deg[index] = angle

    def set_waist_leg_angles(self, waist_leg_angles: list[float]) -> None:
        """
        设置腰腿部分关节角度，对应前 4 个共享电机

        :param waist_leg_angles: [Joint_Ankle, Joint_Knee, Joint_Waist_Pitch, Joint_Waist_Yaw]
        """
        if len(waist_leg_angles) != 4:
            raise ValueError("waist_leg_angles 长度必须为 4")

        for idx, angle in enumerate(waist_leg_angles):
            self.target_left_arm_joints_deg[idx] = angle
            self.target_right_arm_joints_deg[idx] = angle

    def set_left_arm_angles(self, left_arm_angles: list[float]) -> None:
        """
        设置左臂的关节角度

        :param left_arm_angles: 按照 LEFT_MAP 顺序排列，共 7 个角度值
        """
        if len(left_arm_angles) != 7:
            raise ValueError("left_arm_angles 长度必须为 7")

        for joint_name, angle in zip(MotorMap.LEFT_MAP, left_arm_angles):
            joint_idx = MotorMap.LEFT_MAP[joint_name]
            self.target_left_arm_joints_deg[joint_idx] = angle

    def set_right_arm_angles(self, right_arm_angles: list[float]) -> None:
        """
        设置右臂的关节角度

        :param right_arm_angles: 按照 RIGHT_MAP 顺序排列，共 7 个角度值
        """
        if len(right_arm_angles) != 7:
            raise ValueError("right_arm_angles 长度必须为 7")

        for joint_name, angle in zip(MotorMap.RIGHT_MAP, right_arm_angles):
            joint_idx = MotorMap.RIGHT_MAP[joint_name]
            self.target_right_arm_joints_deg[joint_idx] = angle

    def set_all_joint_angle(self,
        waist_leg_angles: list[float],
        left_arm_angles: list[float],
        right_arm_angles: list[float]
    ) -> None:
        """
        设置所有关节的目标角度。

        :param waist_leg_angles: [Joint_Ankle, Joint_Knee, Joint_Waist_Pitch, Joint_Waist_Yaw]
        :param left_arm_angles: 依次为 LEFT_MAP 顺序，共 7 项
        :param right_arm_angles: 依次为 RIGHT_MAP 顺序，共 7 项
        """
        if len(waist_leg_angles) != 4:
            raise ValueError("waist_leg_angles 长度必须为 4")
        if len(left_arm_angles) != 7:
            raise ValueError("left_arm_angles 长度必须为 7")
        if len(right_arm_angles) != 7:
            raise ValueError("right_arm_angles 长度必须为 7")

        # 更新共享的前4个关节（腰腿部分）
        for idx, angle in enumerate(waist_leg_angles):
            self.target_left_arm_joints_deg[idx] = angle
            self.target_right_arm_joints_deg[idx] = angle

        # LEFT_ARM 映射角度（索引从4开始）
        for joint_name, angle in zip(MotorMap.LEFT_MAP, left_arm_angles):
            joint_idx = MotorMap.LEFT_MAP[joint_name]
            self.target_left_arm_joints_deg[joint_idx] = angle

        # RIGHT_ARM 映射角度（索引从4开始）
        for joint_name, angle in zip(MotorMap.RIGHT_MAP, right_arm_angles):
            joint_idx = MotorMap.RIGHT_MAP[joint_name]
            self.target_right_arm_joints_deg[joint_idx] = angle
    
    # 平滑插值函数 3t^2 - 2t^3
    def smoothstep(self, t: float) -> float:
        return t * t * (3 - 2 * t)

    def smooth_set_joints(self, joint_targets: dict[str, float], duration=3.0, steps=100):
        """
        joint_targets: {joint_name: target_angle, ...}
        只对指定关节平滑过渡，其他关节角度保持不变。
        """
        interval = duration / steps
        # 记录起始角度
        start_angles = {}
        for joint_name in joint_targets:
            target_type, idx = MotorMap.resolve_joint(joint_name)
            if target_type == "left":
                start_angles[joint_name] = self.sub_node.current_left_arm_joints_deg[idx]
            elif target_type == "right":
                start_angles[joint_name] = self.sub_node.current_right_arm_joints_deg[idx]
            elif target_type == "shared":
                start_angles[joint_name] = self.sub_node.current_left_arm_joints_deg[idx]
        
        for i in range(steps + 1):
            t = i / steps
            interp_t = self.smoothstep(t)
            for joint_name, target_angle in joint_targets.items():
                start_angle = start_angles[joint_name]
                angle = start_angle + (target_angle - start_angle) * interp_t
                self.set_joint_angle_by_name(joint_name, angle)
            self.publish_joint_angles()
            time.sleep(interval)

    def smooth_set_all_joints(self, duration=3.0, steps=100):
        """
        平滑地将当前所有关节角度过渡到 target_*_joints_deg 中的值。
        包括腰腿关节（前4个）+ 左臂 + 右臂，共18个关节。

        :param duration: 插值总时长（秒）
        :param steps: 插值步数
        """
        interval = duration / steps

        # 缓存起始角度（此时的目标角度）
        start_left_angles = self.sub_node.current_left_arm_joints_deg[:]
        start_right_angles = self.sub_node.current_right_arm_joints_deg[:]

        # 设定最终目标角度（防止插值过程中被改变）
        end_left_angles = self.target_left_arm_joints_deg[:]
        end_right_angles = self.target_right_arm_joints_deg[:]

        for i in range(steps + 1):
            t = i / steps
            interp_t = self.smoothstep(t)
            for joint_id in range(len(self.target_left_arm_joints_deg)):
                start_left = start_left_angles[joint_id]
                start_right = start_right_angles[joint_id]
                target_left = end_left_angles[joint_id]
                target_right = end_right_angles[joint_id]

                interpolated_left = start_left + (target_left - start_left) * interp_t
                interpolated_right = start_right + (target_right - start_right) * interp_t

                self.target_left_arm_joints_deg[joint_id] = interpolated_left
                self.target_right_arm_joints_deg[joint_id] = interpolated_right

            self.publish_joint_angles()
            time.sleep(interval)
        
    def reset_joint_angles(self, duration: float = 3.0, steps: int = 100):
        self.set_all_joint_angle(
            [0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        )
        self.smooth_set_all_joints(duration=duration, steps=steps)


    def bow_salute(self): # 鞠躬
        print("Running Robot Target End-Effector Pose Demo")
        time.sleep(1)

        self.set_left_arm_angles([50, 0, -70, 110, 40, 0, 0])
        self.smooth_set_all_joints(duration=5.0)

        self.smooth_set_joints({"Joint_Waist_Pitch": 30}, duration=3.0)
        time.sleep(1)

        self.reset_joint_angles()
        time.sleep(2)

        print("Finished Demo")

    def guide(self): # 引导
        """
        Run the robot target Guide pose demo.
        """
        print("Running Robot Target Guide Pose Demo")
        time.sleep(1)

        self.set_left_arm_angles([30, 0, 0, 50, 0, 20, -20])
        self.set_right_arm_angles([30, 0, 0, 0, 0, -60, 0])
        self.smooth_set_all_joints(duration=3.0)

        self.smooth_set_joints({"Joint_Waist_Pitch": 15}, duration=2.0)
        self.smooth_set_joints({"Joint_Right_Wrist_Upper": 0}, duration=1.0)

        self.reset_joint_angles()
        time.sleep(2)
        print("Finished Demo")

    def please_say_hello(self): # 下蹲敬礼
        print("Running Robot Target please_say_hello Pose Demo")
        time.sleep(1)

        self.set_right_arm_angles([-50, 0, 70, -110, -40, 0, 0])
        self.smooth_set_all_joints(duration=3.0)
        
        self.smooth_set_joints({
                "Joint_Ankle": 30,
                "Joint_Knee": 40,
                "Joint_Waist_Pitch": 10
            },
            duration=3.0
        )
        time.sleep(1)

        self.reset_joint_angles()
        time.sleep(2)
        print("Finished Demo")

    def wave(self): # 挥手
        print("Running Robot wave Demo")
        time.sleep(1)

        self.set_left_arm_angles([50, 0, 0, 120, 0, 0, 0])
        self.smooth_set_all_joints(duration=2.0)
        time.sleep(2)

        for _ in range(2):
            self.smooth_set_joints({"Joint_Left_UpperArm": 10, "Joint_Left_Wrist_Upper": -20}, duration=2.0)
            self.smooth_set_joints({"Joint_Left_UpperArm": -10, "Joint_Left_Wrist_Upper": 20}, duration=2.0)

        self.reset_joint_angles()
        time.sleep(2)
        print("Finished Demo")

###########################################################################

def run_actions(pub_node, sub_node, exit_event):
    
    action_display = ActionDisplay(pub_node=pub_node, sub_node=sub_node)
    time.sleep(1)

    # bow_salute 鞠躬
    # guide 迎宾
    # please_say_hello 下蹲敬礼
    # wave 挥手
    action_dict = {
        "Action": "guide"  
    }

    msg = String()
    msg.data = json.dumps(action_dict)
    pub_node.robot_action_pub.publish(msg)

    while not exit_event.is_set():
        if sub_node.action_status:
            action_func = getattr(action_display, sub_node.action_name, None)
            if callable(action_func):
                print(f"[run_actions] Executing action: {sub_node.action_name}")
                action_func()
            else:
                print(f"[run_actions] No such method: {sub_node.action_name}")
            sub_node.action_status = False
        else:
            time.sleep(0.05)

def main() -> None:
    rclpy.init()
    exit_event = threading.Event()
    try:
        pub_node = PublisherNode()
        sub_node = ReceiverNode()

        action_thread = threading.Thread(target=run_actions, args=(pub_node, sub_node, exit_event))
        action_thread.start()

        executor = MultiThreadedExecutor()
        executor.add_node(pub_node)
        executor.add_node(sub_node)
        executor.spin()
    except Exception as e:
        print(f"Unexpected exception during execution: {e}")
        traceback.print_exc()
    finally:
        exit_event.set()
        action_thread.join()

        pub_node.destroy_node()
        sub_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Quit by interrupt")
    except Exception as e:
        print(f"Unexpected exception during execution: {e}")
        traceback.print_exc()