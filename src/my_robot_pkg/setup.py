from setuptools import setup, find_packages

package_name = 'my_robot_pkg'

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
    maintainer='walnu7z',
    maintainer_email='walter@ciencias.unam.mx',
    description='TODO: Package description',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ik_action_server = my_robot_pkg.ik_action_server:main',
            'ik_action_client = my_robot_pkg.ik_action_client:main',
        ],
    },
)
