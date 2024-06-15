import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from sensor_msgs.msg import Image
from std_msgs.msg import String

import os
import cv2
from cv_bridge import CvBridge

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        
        self.publisher = self.create_publisher(
            Image, 
            'image', 
            QoSProfile(depth=10))
        
        self.name_publisher = self.create_publisher(
            String, 
            'video_name', 
            QoSProfile(depth=10))
        
        self.timer = self.create_timer(1, self.time_callback)
        
        self.video_dir = 'src/ssd_detection/video/' # 디렉토리 경로
        self.video_files = [f"{chr(i)}.mp4" for i in range(ord('A'), ord('H') + 1)] # A~H.mp4 파일 목록
        self.current_video_index = 0
        
        self.load_video()
        
        self.bridge = CvBridge()
        self.n_frame = 0
        
    def load_video(self):
        while self.current_video_index < len(self.video_files):
            self.video_path = os.path.join(self.video_dir, self.video_files[self.current_video_index])
            if os.path.isfile(self.video_path):
                self.cap = cv2.VideoCapture(self.video_path)
                if self.cap.isOpened():
                    self.get_logger().info(f'Loaded video: {self.video_path}')
                    return
            self.current_video_index += 1
        
        self.current_video_index = 0  # 모든 영상이 끝나면 처음부터 다시 시작
        self.load_video()  # 첫 영상 로드 시도
    
    def time_callback(self):
        ret, frame = self.cap.read()
        if ret:
            fra = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.publisher.publish(fra)
            
            video_name_msg = String()
            video_name_msg.data = self.video_files[self.current_video_index][:-4] # 'A.mp4' -> 'A'
            self.name_publisher.publish(video_name_msg)
            
            self.n_frame += 1
            self.get_logger().info(f'Publishing images: [{self.n_frame}]')
        else:
            self.get_logger().info('End of video. Loading next video...')
            self.current_video_index += 1
            self.load_video()

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
