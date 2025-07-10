try:
    from .realsense_camera import RealSenseCamera
except ImportError:
    print("Failed to import RealSenseCamera. Make sure you have the required dependencies installed.")
    RealSenseCamera = None

try:
    from .face_detection import FaceDetector
except ImportError:
    print("Failed to import FaceDetector. Make sure you have the required dependencies installed.")
    FaceDetector = None

try:
    from .ros_server import RosNode
except ImportError:
    print("Failed to import PublisherNode. Make sure you have the required dependencies installed.")
    RosNode = None