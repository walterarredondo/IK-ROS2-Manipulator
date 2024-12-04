import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from my_robot_interfaces.action import InverseKinematics  # Custom action interface
import numpy as np
import time

class InverseKinematicsActionServer(Node):
    def __init__(self):
        super().__init__('ik_action_server')
        self._action_server = ActionServer(
            self,
            InverseKinematics,
            'inverse_kinematics',
            self.execute_callback
        )
        self.get_logger().info('Inverse Kinematics Action Server Started')

    def dummy_ik_solver(self, target_pose):
        """
        Dummy IK solver that simulates joint calculations
        In a real scenario, this would use actual IK algorithms
        """
        # Simulate joint angle computation
        joint1 = np.random.uniform(-np.pi, np.pi)
        joint2 = np.random.uniform(-np.pi, np.pi)
        return [joint1, joint2]

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        
        # Dummy IK computation
        target_pose = goal_handle.request.target_pose
        joint_angles = self.dummy_ik_solver(target_pose)
        
        # Simulate progress feedback
        feedback_msg = InverseKinematics.Feedback()
        for progress in range(0, 101, 10):
            feedback_msg.progress = progress
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(0.5)  # Simulate computation time
        
        # Set result
        result = InverseKinematics.Result()
        result.joint_positions = joint_angles
        goal_handle.succeed()
        
        return result

def main(args=None):
    rclpy.init(args=args)
    ik_server = InverseKinematicsActionServer()
    rclpy.spin(ik_server)
    rclpy.shutdown()

if __name__ == '__main__':
    main()