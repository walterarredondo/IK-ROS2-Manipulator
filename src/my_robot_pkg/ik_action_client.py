import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from my_robot_interfaces.action import InverseKinematics
from geometry_msgs.msg import Point

class InverseKinematicsActionClient(Node):
    def __init__(self):
        super().__init__('ik_action_client')
        self._action_client = ActionClient(
            self, 
            InverseKinematics, 
            'inverse_kinematics'
        )

    def send_goal(self, target_pose):
        goal_msg = InverseKinematics.Goal()
        goal_msg.target_pose = target_pose

        self._action_client.wait_for_server()
        
        send_goal_future = self._action_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback
        )
        
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback):
        self.get_logger().info(
            f'Received feedback: {feedback.feedback.progress}%'
        )

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(
            f'Result received: Joint Positions = {result.joint_positions}'
        )

def main(args=None):
    rclpy.init(args=args)
    
    action_client = InverseKinematicsActionClient()
    
    # Create a sample target pose
    target_pose = Point()
    target_pose.x = 0.5
    target_pose.y = 0.3
    target_pose.z = 0.4

    action_client.send_goal(target_pose)
    
    rclpy.spin(action_client)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()