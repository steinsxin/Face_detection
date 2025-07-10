import pyrealsense2 as rs
import numpy as np
import time


class RealSenseCamera:
    """Class to handle RealSense camera data capture and provide color and depth image data access."""

    def __init__(self, depth_stream=True, color_stream=True):
        """
        Initialize the RealSenseCamera class.

        Args:
            depth_stream (bool): Enable depth stream. Defaults to True.
            color_stream (bool): Enable color stream. Defaults to True.
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if depth_stream:
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if color_stream:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = None

    def start(self):
        """Start the RealSense camera pipeline."""
        self.profile = self.pipeline.start(self.config)
        print("RealSense camera pipeline started.")

    def stop(self):
        """Stop the RealSense camera pipeline."""
        if self.profile:
            self.pipeline.stop()
            print("RealSense camera pipeline stopped.")

    def get_realsense_images(self):
        """
        Capture and return the latest color and depth images from the RealSense camera.

        Returns:
            dict: A dictionary with "color" and "depth" keys containing numpy arrays or None.
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        images = {}

        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            images["color"] = color_image
        else:
            images["color"] = None
            print("Failed to get color frame.")

        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            images["depth"] = depth_image
        else:
            images["depth"] = None
            print("Failed to get depth frame.")

        return images

def main():
    """Main function to demonstrate RealSense camera usage."""
    camera = RealSenseCamera()
    camera.start()
    try:
        while True:
            images = camera.get_images()
            color_image = images["color"]
            depth_image = images["depth"]

            if color_image is not None and depth_image is not None:
                print(f"Color image shape: {color_image.shape}, Depth image shape: {depth_image.shape}")
            else:
                print("Color or depth image not available.")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()