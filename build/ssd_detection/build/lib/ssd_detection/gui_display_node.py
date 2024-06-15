import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_msgs.msg import String
from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import numpy as np


class GUIDisplayNode(Node):
    def __init__(self):
        super().__init__('gui_display_node')

        qos_profile = QoSProfile(depth=10)

        self.subscription = self.create_subscription(
            String,
            'detection_info',
            self.listener_callback,
            qos_profile)

        self.zone_data = {
            zone: {
                'dotBlockA_counts': [],
                'dotBlockC_counts': [],
                'detection_time': '',
                'warning_triggered': False,
                'current_block': 'A',
                'persistent_warning': False
            } for zone in 'ABCDEFGH'
        }

        self.app = QtWidgets.QApplication(sys.argv)

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle('Detection Info Display')
        self.layout = QtWidgets.QGridLayout()

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
            self.layout.addWidget(group_box, idx // 4, idx % 4)

            self.displays[zone] = {
                'dotBlockA_display': dotBlockA_display,
                'dotBlockC_display': dotBlockC_display,
                'detection_time_display': detection_time_display,
                'warning_light': warning_light,
                'block_indicator': block_indicator
            }

        self.window.setLayout(self.layout)
        self.window.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_ros)
        self.timer.start(100)

        self.gui_timer = QtCore.QTimer()
        self.gui_timer.timeout.connect(self.update_gui)
        self.gui_timer.start(100)

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

            self.zone_data[zone_id]['current_block'] = 'C' if dotBlockC_count > 0 else 'A'

            if dotBlockC_count > 0:
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
    node = GUIDisplayNode()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
