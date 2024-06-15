import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from sensor_msgs.msg import Image
from std_msgs.msg import String

import os
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms.functional import to_pil_image

from datetime import datetime

from PyQt5 import QtWidgets, QtCore, QtGui
import sys

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        self.subscription = self.create_subscription(
            Image,
            'image',
            self._image_callback,
            QoSProfile(depth=10))

        self.name_subscription = self.create_subscription(
            String,
            'video_name',
            self._name_callback,
            QoSProfile(depth=10))

        self.detection_subscription = self.create_subscription(
            String,
            'detection_info',
            self.listener_callback,
            QoSProfile(depth=10))

        self.subscription
        self.bridge = CvBridge()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ptr_weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self.ckpt_path = "src/ssd_detection/SSD_CheckPoint.pth"
        self.ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.net = self.ckpt['model'].to(self.device)
        self.net.score_thresh=0.6
        self.net.eval()

        self.preproc = self.ptr_weights.transforms()
        self.classes = self.ckpt['category_list']

        self.gui_publisher = self.create_publisher(String, 'detection_info', QoSProfile(depth=10))

        self.zone_id = "Unknown"

        # GUI 초기화
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle('Detection Info Display')

        screen = QtWidgets.QApplication.desktop().screenGeometry()
        width = int(screen.width() * 0.5)
        height = int(screen.height() * 0.75)
        self.window.setGeometry((screen.width() - width) // 2, (screen.height() - height) // 2, width, height)

        self.layout = QtWidgets.QHBoxLayout()

        self.left_layout = QtWidgets.QVBoxLayout()
        self.image_label = QtWidgets.QLabel()
        self.left_layout.addWidget(self.image_label)

        self.right_layout = QtWidgets.QGridLayout()
        self.zone_data = {zone: {'dotBlockA_counts': [], 'dotBlockC_counts': [], 'detection_time': '',
                                 'warning_triggered': False, 'current_block': 'A', 'persistent_warning': False}
                          for zone in 'ABCDEFGH'}

        self.displays = {}
        for idx, zone in enumerate('ABCDEFGH'):
            group_box = QtWidgets.QGroupBox(f'Zone {zone}')
            zone_layout = QtWidgets.QVBoxLayout()

            dotBlockA_display = QtWidgets.QLCDNumber()
            zone_layout.addWidget(dotBlockA_display)

            dotBlockC_display = QtWidgets.QLCDNumber()
            zone_layout.addWidget(dotBlockC_display)

            detection_time_display = QtWidgets.QLabel()
            zone_layout.addWidget(detection_time_display)

            warning_layout = QtWidgets.QHBoxLayout()
            warning_light = QtWidgets.QLabel()
            warning_light.setStyleSheet('background-color: green')
            warning_layout.addWidget(warning_light)
            warning_label = QtWidgets.QLabel('Need to Change')
            warning_layout.addWidget(warning_label)
            zone_layout.addLayout(warning_layout)

            block_indicator_layout = QtWidgets.QHBoxLayout()
            block_indicator = QtWidgets.QLabel()
            block_indicator.setStyleSheet('background-color: green')
            block_indicator_layout.addWidget(block_indicator)
            block_indicator_label = QtWidgets.QLabel('RealTime Detect')
            block_indicator_layout.addWidget(block_indicator_label)
            zone_layout.addLayout(block_indicator_layout)

            group_box.setLayout(zone_layout)
            self.right_layout.addWidget(group_box, idx // 4, idx % 4)

            self.displays[zone] = {'dotBlockA_display': dotBlockA_display, 'dotBlockC_display': dotBlockC_display,
                                   'detection_time_display': detection_time_display, 'warning_light': warning_light,
                                   'block_indicator': block_indicator}

        self.layout.addLayout(self.left_layout)
        self.layout.addLayout(self.right_layout)

        self.window.setLayout(self.layout)
        self.window.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_ros)
        self.timer.start(100)

        self.gui_timer = QtCore.QTimer()
        self.gui_timer.timeout.connect(self.update_gui)
        self.gui_timer.start(100)

    def _name_callback(self, msg):
        self.zone_id = msg.data

    def _image_callback(self, msg_data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg_data, "bgr8")
            image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except CvBridgeError as e:
            self.get_logger().error(f"Error Occured! Fail to convert the image message data to OpenCV: {e}")
            return

        image = to_pil_image(image)
        image = self.preproc(image)
        images = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.net(images)[0]

        boxes, scores, labels = prediction["boxes"], prediction["scores"], prediction["labels"]

        dotBlockA_count = len([label for label in labels if self.classes[label] == 'dotBlockA'])
        dotBlockC_count = len([label for label in labels if self.classes[label] == 'dotBlockC'])
        detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        info_msg = String()
        info_msg.data = f"{dotBlockA_count},{dotBlockC_count},{self.zone_id},{detection_time}"
        self.gui_publisher.publish(info_msg)

        for ind, (bndbox, score, label) in enumerate(zip(boxes, scores, labels)):
            label_text = f'{self.classes[label]}: {score:.1f}'
            if self.classes[label] == 'dotBlockA':
                box_color = (0, 255, 0)
            elif self.classes[label] == 'dotBlockC':
                box_color = (0, 0, 255)
            else:
                box_color = (255, 255, 0)

            cv_image = cv2.rectangle(cv_image,
                                     (int(bndbox[0]), int(bndbox[1])),
                                     (int(bndbox[2]), int(bndbox[3])),
                                     box_color,
                                     4)
            cv_image = cv2.putText(cv_image,
                                   label_text,
                                   (int(bndbox[0]) + 20, int(bndbox[1]) + 40),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   1,
                                   (255, 0, 255), 2)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_image.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(cv_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qImg))

    def listener_callback(self, msg):
        data = msg.data.split(',')
        dotBlockA_count = int(data[0])
        dotBlockC_count = int(data[1])
        zone_id = data[2]
        detection_time = data[3]

        if zone_id in self.zone_data:
            self.zone_data[zone_id]['dotBlockA_counts'].append(dotBlockA_count)
            self.zone_data[zone_id]['dotBlockC_counts'].append(dotBlockC_count)
            self.zone_data[zone_id]['detection_time'] = detection_time

            if len(self.zone_data[zone_id]['dotBlockA_counts']) > 10:
                self.zone_data[zone_id]['dotBlockA_counts'].pop(0)
            if len(self.zone_data[zone_id]['dotBlockC_counts']) > 10:
                self.zone_data[zone_id]['dotBlockC_counts'].pop(0)

            avg_dotBlockC = np.mean(self.zone_data[zone_id]['dotBlockC_counts'][-5:]) if len(self.zone_data[zone_id]['dotBlockC_counts']) >= 5 else 0

            self.zone_data[zone_id]['current_block'] = 'C' if avg_dotBlockC >= 1 else 'A'

            if avg_dotBlockC >= 1:
                self.zone_data[zone_id]['warning_triggered'] = True
                self.zone_data[zone_id]['persistent_warning'] = True
            else:
                self.zone_data[zone_id]['warning_triggered'] = False

    def update_gui(self):
        for zone_id, data in self.zone_data.items():
            avg_dotBlockA = np.mean(data['dotBlockA_counts']) if data['dotBlockA_counts'] else 0
            avg_dotBlockC = np.mean(data['dotBlockC_counts']) if data['dotBlockC_counts'] else 0

            self.displays[zone_id]['dotBlockA_display'].display(avg_dotBlockA)
            self.displays[zone_id]['dotBlockC_display'].display(avg_dotBlockC)
            self.displays[zone_id]['detection_time_display'].setText(f'Detection Time: {data["detection_time"]}')

            if data['persistent_warning']:
                self.displays[zone_id]['warning_light'].setStyleSheet('background-color: red')
            else:
                self.displays[zone_id]['warning_light'].setStyleSheet('background-color: green')

            if data['current_block'] == 'C':
                self.displays[zone_id]['block_indicator'].setStyleSheet('background-color: red')
            else:
                self.displays[zone_id]['block_indicator'].setStyleSheet('background-color: green')

    def update_ros(self):
        rclpy.spin_once(self, timeout_sec=0)

    def run(self):
        sys.exit(self.app.exec_())

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
