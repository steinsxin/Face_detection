import cv2
import time
import json
import threading
import traceback
from typing import Optional, Tuple

from utils import RealSenseCamera, FaceDetector
import numpy as np
import mediapipe as mp

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from utils.ros_server import RosNode

LOST_FACE_MAX_COUNT = 5 

class SharedState:
    def __init__(self):
        self._color_image: Optional[np.ndarray] = None
        self._depth_image: Optional[np.ndarray] = None
        self.image_ready = False
        self.lock = threading.Lock()

        self._face_start_time: Optional[float] = None
        self._face_confirmed_time: Optional[float] = None
        self._last_face_duration: float = 0.0

        self.face_confirmed: bool = False
        self.face_distance: float = 0.0
        self.check_time = 5.0

        self.audio_done: bool = True
        self.action_done: bool = True

        self._face_duration: float = 0.0  # 实时持续时间
        self._last_update_time: float = time.time()  # 上一次更新的时间戳


    def update_images(self, color: np.ndarray, depth: np.ndarray):
        with self.lock:
            self._color_image = color
            self._depth_image = depth
            self.image_ready = True

    def get_latest_frame_pair(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self.lock:
            if self.image_ready and self._color_image is not None and self._depth_image is not None:
                self.image_ready = False
                return self._color_image.copy(), self._depth_image.copy()
            return None, None

    def update_face_info(self, detected: bool, distance: float):
        now = time.time()
        dt = now - self._last_update_time
        self._last_update_time = now

        with self.lock:
            # 人脸检测中，更新持续时间
            if detected and 0.3 <= distance <= 2.0:
                if self._face_start_time is None:
                    self._face_start_time = now
                    self._face_duration = 0.0
                else:
                    if self._face_duration < self.check_time:
                        self._face_duration += dt  # 实时增长(限制在检测时间内)
                
                if not self.face_confirmed and self._face_duration >= self.check_time:
                    self.face_confirmed = True
                    self._face_confirmed_time = now
                    self.face_distance = distance

            else:
                # 人脸丢失时，持续时间逐渐减少
                self._face_duration -= dt
                if self._face_duration < 0.0:
                    self._face_duration = 0.0

                if self.face_confirmed and self._face_start_time is not None:
                    self._face_start_time = None
                    self._face_confirmed_time = None
                    self.face_confirmed = False
                    self.face_distance = 0.0

    def get_face_duration(self) -> float:
        with self.lock:
            return self._face_duration

    def get_face_info(self) -> Tuple[bool, float]:
        with self.lock:
            return self.face_confirmed, self.face_distance

    def get_last_face_duration(self) -> float:
        with self.lock:
            return self._last_face_duration

    def mark_audio_done(self):
        with self.lock:
            self.audio_done = True

    def mark_action_done(self):
        with self.lock:
            self.action_done = True

    def all_tasks_done(self) -> bool:
        with self.lock:
            return self.audio_done and self.action_done

    def reset_all_face_state(self):
        with self.lock:
            self._face_start_time = None
            self._face_confirmed_time = None
            self._face_duration = 0.0
            self._last_face_duration = 0.0
            self.face_confirmed = False
            self.face_distance = 0.0
            self.action_triggered = False
            self.audio_done = False
            self.action_done = False


def capture_camera_loop(camera: RealSenseCamera, state: SharedState, stop_event: threading.Event) -> None:
    """
    摄像头线程：持续采集 color 和 depth 图像。
    """
    try:
        while not stop_event.is_set():
            images = camera.get_realsense_images()
            if images and images.get("color") is not None and images.get("depth") is not None:
                state.update_images(images["color"], images["depth"])
            else:
                print("[Camera] Incomplete image data.")
            time.sleep(0.05)
    except Exception:
        print("[Camera] Exception occurred:")
        traceback.print_exc()
    finally:
        camera.stop()
        print("[Camera] Stopped.")

def face_detection_loop(detector, state: SharedState, stop_event: threading.Event) -> None:
    try:
        lost_counter = 0

        while not stop_event.is_set():
            if state.all_tasks_done():
                print("[FaceDetector] All tasks done. Resetting state.")
                state.reset_all_face_state()

            color_image, depth_image = state.get_latest_frame_pair()

            if color_image is not None and depth_image is not None:
                has_face, boxes = detector.detect_faces(color_image)

                if has_face:
                    distance = detector.distance_solve(color_image, boxes)
                    if distance is not None:
                        state.update_face_info(True, distance / 1000)
                        lost_counter = 0 
                    else:
                        lost_counter += 1
                        if lost_counter >= LOST_FACE_MAX_COUNT:
                            state.update_face_info(False, 0.0)
                else:
                    lost_counter += 1
                    if lost_counter >= LOST_FACE_MAX_COUNT:
                        state.update_face_info(False, 0.0)

                face_confirmed, face_distance = state.get_face_info()

                if face_confirmed:
                    print(f"[FaceDetector] Confirmed face at {face_distance:.2f} meters")

                duration = state.get_face_duration()
                print(f"[FaceDetector] Current face duration: {duration:.2f} seconds")

            time.sleep(0.05)
    except Exception:
        print("[FaceDetector] Exception occurred:")
        traceback.print_exc()
    finally:
        print("[FaceDetector] Stopped.")


def action_loop(node: RosNode, state: SharedState, stop_event: threading.Event) -> None:
    try:
        while not stop_event.is_set():
            has_face, distance = state.get_face_info()
            if has_face:
                audio_dict = {
                    "target": "gpt",
                    "text": "你好，我是智动1号机器人"
                }
                msg = String()
                msg.data = json.dumps(audio_dict)
                node.audio_topic_pub.publish(msg)
                print("[Audio] Published audio message:", msg.data)

                time.sleep(node.time_sleep)
                state.mark_audio_done()

            time.sleep(0.05)
    except Exception:
        print("[Action] Exception occurred:")
        traceback.print_exc()
    finally:
        print("[Action] Stopped.")

def audio_loop(node: RosNode, state: SharedState, stop_event: threading.Event) -> None:
    try:
        while not stop_event.is_set():
            has_face, distance = state.get_face_info()
            if has_face:
                action_dict = {
                    "action": "bow_salute",
                    "status": "run",
                }
                msg = String()
                msg.data = json.dumps(action_dict)
                node.robot_action_pub.publish(msg)
                print("[Action] Published action message:", msg.data)

                time.sleep(node.time_sleep)
                state.mark_action_done()

            time.sleep(0.05)
    except Exception:
        print("[Audio] Exception occurred:")
        traceback.print_exc()
    finally:
        print("[Audio] Stopped.")

def main():
    print("[Main] Starting system...")
    rclpy.init()
    ros_node = RosNode()

    camera = RealSenseCamera()
    camera.start()

    detector = FaceDetector(
        model_file="models/deploy.prototxt",
        weights_file="models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    )

    shared_state = SharedState()
    stop_event = threading.Event()
    camera_thread = threading.Thread(
        target=capture_camera_loop, args=(camera, shared_state, stop_event),
        name="CameraThread", daemon=True
    )
    detector_thread = threading.Thread(
        target=face_detection_loop, args=(detector, shared_state, stop_event),
        name="FaceDetectThread", daemon=True
    )
    action_thread = threading.Thread(
        target=action_loop, args=(ros_node, shared_state, stop_event),
        name="ActionDisplayThread", daemon=True
    )
    audio_thread = threading.Thread(
        target=audio_loop, args=(ros_node, shared_state, stop_event),
        name="AudioThread", daemon=True
    )

    camera_thread.start()
    detector_thread.start()
    action_thread.start()
    audio_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Main] Quit by user interrupt.")
    except Exception as e:
        print(f"[Main] Unexpected error: {e}")
        traceback.p

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Quit by interrupt")
    except Exception as e:
        print(f"Unexpected exception during execution: {e}")
        traceback.print_exc()