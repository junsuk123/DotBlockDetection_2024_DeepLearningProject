# ssd_detection/setup.py

from setuptools import setup
from glob import glob
import os

package_name = 'ssd_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.py')),  # 이 줄 추가
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kimjunsuk',
    maintainer_email='kimjunsuk@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_example = ssd_detection.ssd_detection:main',
            'video_publisher = ssd_detection.video_publisher:main',
        ],
    },
)
