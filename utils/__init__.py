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
    from .audio_play import AudioPlayer
except ImportError:
    print("Failed to import AudioPlayer. Make sure you have the required dependencies installed.")
    AudioPlayer = None

try:
    from .publisher_node import PublisherNode
except ImportError:
    print("Failed to import PublisherNode. Make sure you have the required dependencies installed.")
    PublisherNode = None

try:
    from .receiver_node import ReceiverNode
except ImportError:
    print("Failed to import ReceiverNode. Make sure you have the required dependencies installed.")
    ReceiverNode = None