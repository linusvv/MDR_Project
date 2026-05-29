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

# How long (seconds) without a /cmd_vel message before sending a stop command
CMD_VEL_TIMEOUT = 10.0

class CmdVelToChassis:
    def __init__(self):
        rospy.init_node('cmd_vel_to_chassis')

        self.last_cmd_time = rospy.Time(0)
        self.is_stopped = True  # Track if we already sent a stop to avoid spamming
        self.last_direction_deg = 90.0  # Track last non-zero movement direction
        self.current_vel_msg = SetVelocity()
        self.current_vel_msg.velocity = 0.0
        self.current_vel_msg.direction = 90.0
        self.current_vel_msg.angular = 0.0
        
        self.vel_pub = rospy.Publisher('/chassis_control/set_velocity', SetVelocity, queue_size=1)
        self.sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_cb)

        # Publish timer: fires at 60 Hz to send continuous commands to the motor controller
        self.publish_timer = rospy.Timer(rospy.Duration(1.0 / 60.0), self.publish_cb)
        
        rospy.loginfo("[cmd_vel_to_chassis] Node started. Bridging /cmd_vel to /chassis_control/set_velocity at 60Hz (timeout=%.1fs)", CMD_VEL_TIMEOUT)

    def cmd_vel_cb(self, msg):
        # Speed components with dynamic limit clamping
        max_vel_m = rospy.get_param("/robot/max_vel", 0.06)
        max_omega = rospy.get_param("/robot/max_vel_theta", 0.3)
        
        linear_x = msg.linear.x
        linear_y = msg.linear.y
        mag = math.hypot(linear_x, linear_y)
        if mag > max_vel_m:
            scale = max_vel_m / mag
            linear_x *= scale
            linear_y *= scale
            
        vy = linear_x * 1000.0
        vx = -linear_y * 1000.0
        angular = max(-max_omega, min(max_omega, msg.angular.z))
        
        # Calculate velocity magnitude (mm/s) and direction (degrees, 0 to 360)
        velocity = math.hypot(vx, vy)
        
        if velocity > 0.01:
            direction_rad = math.atan2(vy, vx)
            direction_deg = math.degrees(direction_rad)
            if direction_deg < 0:
                direction_deg += 360.0
            self.last_direction_deg = direction_deg
        else:
            direction_deg = self.last_direction_deg
            
        # Update the current velocity message
        self.current_vel_msg.velocity = velocity
        self.current_vel_msg.direction = direction_deg
        self.current_vel_msg.angular = angular

        # Reset watchdog timer
        self.last_cmd_time = rospy.Time.now()
        self.is_stopped = False

    def publish_cb(self, event):
        """Continuously publish at 60Hz. Send zero-velocity if /cmd_vel has gone silent."""
        elapsed = (rospy.Time.now() - self.last_cmd_time).to_sec()
        
        if elapsed > CMD_VEL_TIMEOUT:
            if not self.is_stopped:
                self.current_vel_msg.velocity = 0.0
                self.current_vel_msg.direction = self.last_direction_deg
                self.current_vel_msg.angular = 0.0
                self.is_stopped = True
                rospy.logdebug("[cmd_vel_to_chassis] Watchdog: no /cmd_vel for %.2fs — sending stop.", elapsed)
        
        # Always publish the current command at 60Hz
        self.vel_pub.publish(self.current_vel_msg)

if __name__ == '__main__':
    try:
        node = CmdVelToChassis()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
