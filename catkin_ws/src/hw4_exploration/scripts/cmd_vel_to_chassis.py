#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist
try:
    from chassis_control.msg import SetVelocity, SetTranslation
except ImportError:
    # Fallback dummy classes in case they are not compiled on user host
    class SetVelocity:
        def __init__(self):
            self.velocity = 0.0
            self.direction = 0.0
            self.angular = 0.0
    class SetTranslation:
        def __init__(self):
            self.velocity_x = 0.0
            self.velocity_y = 0.0

class CmdVelToChassis:
    def __init__(self):
        rospy.init_node('cmd_vel_to_chassis')
        
        self.vel_pub = rospy.Publisher('/chassis_control/set_velocity', SetVelocity, queue_size=1)
        self.sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_cb)
        
        rospy.loginfo("[cmd_vel_to_chassis] Node started. Bridging /cmd_vel to /chassis_control/set_velocity")

    def cmd_vel_cb(self, msg):
        # Translate Twist to SetVelocity
        # linear.x -> forward (vy)
        # linear.y -> left (negative vx)
        # angular.z -> yaw rate (angular)
        
        # Speed components in mm/s
        vy = msg.linear.x * 1000.0
        vx = -msg.linear.y * 1000.0
        angular = msg.angular.z
        
        # Calculate velocity magnitude (mm/s) and direction (degrees, 0 to 360)
        velocity = math.hypot(vx, vy)
        
        if velocity > 0.01:
            direction_rad = math.atan2(vy, vx)
            direction_deg = math.degrees(direction_rad)
            if direction_deg < 0:
                direction_deg += 360.0
        else:
            direction_deg = 90.0 # Default forward direction when stopped
            
        # Create SetVelocity message
        vel_msg = SetVelocity()
        vel_msg.velocity = velocity
        vel_msg.direction = direction_deg
        vel_msg.angular = angular
        
        self.vel_pub.publish(vel_msg)

if __name__ == '__main__':
    try:
        node = CmdVelToChassis()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
