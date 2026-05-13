#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from hw4_exploration.srv import AdjustBase, AdjustBaseResponse
import time

class AdjustBaseNode:
    def __init__(self):
        rospy.init_node("adjust_base_node")
        
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.service = rospy.Service("adjust_base_for_grasping", AdjustBase, self.handle_adjust)
        
        rospy.loginfo("Adjust Base Node Ready.")

    def handle_adjust(self, req):
        # Extremely simple open-loop movement for micro-adjustments
        # In a real scenario, this would use visual odometry feedback
        twist = Twist()
        rate = rospy.Rate(10)
        
        # Translation
        if req.delta_x != 0 or req.delta_y != 0:
            twist.linear.x = 0.1 if req.delta_x > 0 else -0.1 if req.delta_x < 0 else 0
            twist.linear.y = 0.1 if req.delta_y > 0 else -0.1 if req.delta_y < 0 else 0
            
            duration = max(abs(req.delta_x), abs(req.delta_y)) / 0.1
            start_time = time.time()
            while time.time() - start_time < duration:
                self.cmd_pub.publish(twist)
                rate.sleep()
                
            # Stop translation
            twist.linear.x = 0
            twist.linear.y = 0
            self.cmd_pub.publish(twist)
            
        # Rotation
        if req.delta_theta != 0:
            twist.angular.z = 0.2 if req.delta_theta > 0 else -0.2
            duration = abs(req.delta_theta) / 0.2
            start_time = time.time()
            while time.time() - start_time < duration:
                self.cmd_pub.publish(twist)
                rate.sleep()
                
            # Stop rotation
            twist.angular.z = 0
            self.cmd_pub.publish(twist)

        return AdjustBaseResponse(True, "Adjustment complete")

if __name__ == "__main__":
    try:
        node = AdjustBaseNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
