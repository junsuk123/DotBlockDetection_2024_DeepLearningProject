# ssd_detection/launch/ssd_detection_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ssd_detection',
            executable='detection_example',
            name='detection_node',
            output='screen',
        ),
        Node(
            package='ssd_detection',
            executable='video_publisher',
            name='video_publisher_node',
            output='screen',
        ),
       
    ])
