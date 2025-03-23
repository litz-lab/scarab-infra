#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import cv2
import numpy as np

class ImageRepublisher(Node):
    def __init__(self):
        super().__init__('image_republisher')

        self.subscription = self.create_subscription(
            CompressedImage,
            '/sensing/camera/traffic_light/image_raw/compressed',
            self.image_callback,
            10
        )

        self.publisher_ = self.create_publisher(
            Image,
            '/localization/pose_estimator/yabloc/image_processing/undistorted/image_raw',
            10
        )

    def image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        ros_image_msg = Image()
        ros_image_msg.header = msg.header
        ros_image_msg.height, ros_image_msg.width, channels = cv_image.shape
        ros_image_msg.encoding = 'bgr8'
        ros_image_msg.is_bigendian = False
        ros_image_msg.step = ros_image_msg.width * channels
        ros_image_msg.data = cv_image.tobytes()

        self.publisher_.publish(ros_image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImageRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
