import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'lm'


def _model_data_files():
    out = []
    model_root = os.path.join('models')
    for root, _, files in os.walk(model_root):
        if not files:
            continue
        rel = os.path.relpath(root, model_root)
        install_dir = os.path.join('share', package_name, 'models')
        if rel != '.':
            install_dir = os.path.join(install_dir, rel)
        out.append((install_dir, [os.path.join(root, f) for f in files]))
    return out


def _keyframe_data_files():
    files = glob(os.path.join('keyframes', '*.npz'))
    if not files:
        return []
    return [(os.path.join('share', package_name, 'keyframes'), files)]


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*_launch.py')),
    ] + _model_data_files() + _keyframe_data_files(),
    install_requires=[
        'setuptools',
        'numpy',
        'mujoco',
    ],
    zip_safe=True,
    maintainer='sitongchen',
    maintainer_email='kevinchensitong110@gmail.com',
    description='ROS node for VLM interaction',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vlm_server = lm.vlm_service:main',
            'vlm_client = lm.vml:main',
            'dummy_camera = lm.dummy_camera:main',
            'vml = lm.vml:main',
            'keyframe_box_retarget = lm.keyframe_box_retarget:main',
            'keyframe_retargeter = lm.keyframe_retargeter_node:main',
            'mujoco_visualizer = lm.mujoco_visualizer:main',
        ],
    },
)
