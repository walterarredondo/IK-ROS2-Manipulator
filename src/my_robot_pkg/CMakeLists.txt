cmake_minimum_required(VERSION 3.8)
project(my_robot_pkg)

# Find dependencies
find_package(ament_cmake REQUIRED)

# Install the launch directory
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
install(
  DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Mark the package for ament
ament_package()
