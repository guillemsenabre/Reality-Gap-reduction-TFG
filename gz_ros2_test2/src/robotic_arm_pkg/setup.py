from setuptools import find_packages, setup

package_name = 'robotic_arm_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kyu8',
    maintainer_email='guillemsenabre@gmail.com',
    description='TODO: Package description',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
		'listener = robotic_arm_pkg.joint_listener:main',
		'processor = robotic_arm_pkg.joint_processor:main',
		'storage = robotic_arm_pkg.joint_storage:main',
		'controller = robotic_arm_pkg.joint_controller:main',
        ],
    },
)
