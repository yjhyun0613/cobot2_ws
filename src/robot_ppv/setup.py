from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'robot_ppv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), glob('resource/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eycho',
    maintainer_email='eycho96@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'robot_control_t = robot_ppv.robot_control:main',
            'robot_spin = robot_ppv.robot_spin:main',
            'robot_control_pose = robot_ppv.robot_control_pose:main',
            'robot_control_test = robot_ppv.robot_control_test:main',
            'robot_control_test2 = robot_ppv.robot_control_test2:main',
            'robot_control_test3 = robot_ppv.robot_control_test3:main',
            'robot_control_test2_1 = robot_ppv.robot_control_test2_1:main',
            'robot_control_test4 = robot_ppv.robot_control_test4:main',
            'robot_control_test5 = robot_ppv.robot_control_test5:main',
            'robot_control_test6 = robot_ppv.robot_control_test6:main',
            'robot_driver_check = robot_ppv.robot_driver_check:main',
            'robot_driver_check_test = robot_ppv.robot_driver_check_test:main',
            'robot_driver_check_test2 = robot_ppv.robot_driver_check_test2:main',
            'robot_driver_check_test3 = robot_ppv.robot_driver_check_test3:main',
            'robot_driver_check_test3_1 = robot_ppv.robot_driver_check_test3_1:main',
            'robot_driver_check_test3_2 = robot_ppv.robot_driver_check_test3_2:main',
            'robot_driver_check_test3_3 = robot_ppv.robot_driver_check_test3_3:main',
            'robot_driver_check_test3_4 = robot_ppv.robot_driver_check_test3_4:main',
            'robot_driver_check_test3_5 = robot_ppv.robot_driver_check_test3_5:main',
        ],
    },
)
