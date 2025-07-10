import cv2
import time
import threading
import traceback
from typing import Optional, Tuple

from utils import RealSenseCamera, FaceDetector
import numpy as np
import mediapipe as mp

class SharedState:
    """
    用于线程间共享图像数据的状态容器。
    """
    def __init__(self):
        self._color_image: Optional[np.ndarray] = None
        self._depth_image: Optional[np.ndarray] = None
        self.image_ready = False
        self.lock = threading.Lock()

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


def face_detection_loop(detector, state, stop_event: threading.Event) -> None:
    """
    人脸检测线程：使用 color + depth 图像进行处理。
    """

    try:
        while not stop_event.is_set():
            color_image, depth_image = state.get_latest_frame_pair()

            if color_image is not None and depth_image is not None:
                has_face, boxes = detector.detect_faces(color_image)
                print("[FaceDetector] Detected face." if has_face else "[FaceDetector] No face detected.")
                if has_face:
                    distance = detector.distance_solve(color_image, boxes)
                    if distance is not None:
                        print(f"[FaceDetector] Estimated face distance: {distance / 1000:.2f} meters")
                    else:
                        print("[FaceDetector] Failed to estimate face distance.")
                else:
                    print("[FaceDetector] No face detected.")

            else:
                print("[FaceDetector] No new color or depth image.")

            time.sleep(2)
    except Exception:
        print("[FaceDetector] Exception occurred:")
        traceback.print_exc()
    finally:
        print("[FaceDetector] Stopped.")


def main():
    print("[Main] Starting system...")

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

    camera_thread.start()
    detector_thread.start()

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