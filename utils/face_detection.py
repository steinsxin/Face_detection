import os
import cv2
import numpy as np
import traceback
import mediapipe as mp

class FaceModel:
    """
    存储用于 solvePnP 的人脸 3D 模型点 和 mediapipe 关键点索引映射
    """
    mp_face_mesh = mp.solutions.face_mesh

    # Mediapipe FaceMesh 中 468 点模型的索引映射
    KEYPOINT_INDEX = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 33,
        "right_eye_outer": 263,
        "left_mouth_corner": 61,
        "right_mouth_corner": 291,
    }

    # 3D 模型点坐标（单位：mm）
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),             # 鼻尖
        (0.0, -63.6, -12.5),         # 下巴
        (-42.0, 32.0, -26.0),        # 左眼角
        (42.0, 32.0, -26.0),         # 右眼角
        (-28.0, -28.9, -24.1),       # 左嘴角
        (28.0, -28.9, -24.1),        # 右嘴角
    ], dtype=np.float64)

class FaceDetector:
    """
    A class to detect faces in images using a pre-trained DNN model.
    """
    def __init__(self, model_file: str, weights_file: str):
        """
        Initialize the FaceDetector class.

        :param model_file: Path to the model configuration file.
        :param weights_file: Path to the model weights file.
        """
        self.image_width = 640
        self.image_height = 480
        self.confidence_threshold = 0.5

        self.net = cv2.dnn.readNetFromCaffe(model_file, weights_file)
        self.face_mesh = FaceModel.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1
        )

        # Camera intrinsic parameters (assuming pinhole camera with no distortion)
        self.focal_length = self.image_width  # Assume fx = fy
        self.center = (self.image_width / 2, self.image_height / 2)
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.center[0]],
            [0, self.focal_length, self.center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    def get_image_points(self, landmarks, image_width: int, image_height: int) -> np.ndarray:
        """
        Extract 2D image points from Mediapipe landmarks.

        :param landmarks: Mediapipe landmarks object.
        :param image_width: Width of the ROI.
        :param image_height: Height of the ROI.
        :return: Numpy array of 2D image points.
        """
        points = []
        for name in [
            "nose_tip",
            "chin",
            "left_eye_outer",
            "right_eye_outer",
            "left_mouth_corner",
            "right_mouth_corner"
        ]:
            idx = FaceModel.KEYPOINT_INDEX[name]
            lm = landmarks.landmark[idx]
            x = int(lm.x * image_width)
            y = int(lm.y * image_height)
            points.append((x, y))

        return np.array(points, dtype=np.float64)

    def distance_solve(self, color_image: np.ndarray, boxes: list) -> float or None:
        """
        Estimate face distance using solvePnP.

        :param color_image: BGR image from the camera.
        :param boxes: List of face bounding boxes.
        :return: Estimated distance in mm, or None if failed.
        """
        if not boxes:
            return None

        x1, y1, x2, y2 = map(int, boxes[0])
        face_roi = color_image[y1:y2, x1:x2]

        # Convert ROI to RGB for Mediapipe
        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_face)

        if not results.multi_face_landmarks:
            print("[FaceDetector] Mediapipe did not detect face landmarks.")
            return None

        landmarks = results.multi_face_landmarks[0]
        image_points = self.get_image_points(landmarks, x2 - x1, y2 - y1)

        # Adjust coordinates from ROI to full image
        image_points[:, 0] += x1
        image_points[:, 1] += y1

        try:
            success, rvec, tvec = cv2.solvePnP(
                FaceModel.MODEL_POINTS,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                distance = np.linalg.norm(tvec)
                return distance

        except cv2.error as e:
            print(f"[FaceDetector] solvePnP failed: {e}")

        return None

    def detect_faces(self, image_data: np.ndarray) -> list:
        """
        Detect faces and return a list of bounding box corner points.

        :param image_data: Input image
        :param confidence_threshold: Minimum confidence to consider a face
        :return: List of bounding boxes, each box is (startX, startY, endX, endY)
        """
        if image_data is None:
            print("输入的图像数据无效！")
            return False, []

        (h, w) = image_data.shape[:2]
        blob = cv2.dnn.blobFromImage(image_data, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype("int")
                boxes.append((startX, startY, endX, endY))

        return (len(boxes) > 0), boxes

def main() -> None:
    """
    Main function to demonstrate face detection using the FaceDetector class.
    """
    model_file = "models/deploy.prototxt"
    weights_file = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    face_detector = FaceDetector(model_file, weights_file)

    # Load image data as a NumPy array
    image_path = "images/01.jpg"
    output_path = "detected_faces.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("无法加载图像，请检查路径是否正确！")
        return

    has_face, boxes = face_detector.detect_faces(image)
    if not has_face:
        print("图像中未检测到人脸")
    else:
        print(f"检测到 {len(boxes)} 张人脸")
        for i, (startX, startY, endX, endY) in enumerate(boxes):
            print(f"人脸{i + 1}角点: ({startX}, {startY}), ({endX}, {endY})")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        distance = face_detector.distance_solve(image, boxes)
        print(f"[FaceDetector] Estimated face distance: {distance / 1000:.2f} meters")
        cv2.imwrite(output_path, image)
        print(f"结果图像已保存至: {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Quit by interrupt")
    except Exception as e:
        print(f"Unexpected exception during execution: {e}")
        traceback.print_exc()