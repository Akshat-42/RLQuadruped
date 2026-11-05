from setuptools import find_packages, setup

package_name = 'quadruped_rl'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akshat',
    maintainer_email='akshatpaniker@gmail.com',
    description='Reinforcement Learning (PPO) training and ROS 2 inference for 8-DOF quadruped robot',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'inference_node = quadruped_rl.inference_node:main',
            'train = quadruped_rl.train:main',
            'eval_policy = quadruped_rl.eval_policy:main',
        ],
    },
)
