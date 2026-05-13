#!/usr/bin/env python3
import rospy
import math
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

class RTABLocalPlanner:
    def __init__(self):
        rospy.init_node('rtab_local_planner')
        
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        self.odom_sub = rospy.Subscriber('/rtabmap/odom', Odometry, self.odom_cb)
        self.path_sub = rospy.Subscriber('/graph_planner/path/global_path', Path, self.path_cb)
        self.map_sub = rospy.Subscriber('/rtabmap/grid_map', OccupancyGrid, self.map_cb)

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.odom_ready = False
        self.global_path = None
        self.map_data = None

        self.lookahead_distance = 0.6
        self.goal_tolerance = 0.2
        rospy.loginfo("RTAB Local Planner initialized. Waiting for topics...")

    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        _, _, self.robot_yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        self.odom_ready = True

    def path_cb(self, msg):
        self.global_path = msg
        rospy.loginfo("Global path received with %d points" % len(msg.poses))

    def map_cb(self, msg):
        self.map_data = msg

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if not self.odom_ready:
                rospy.logwarn_throttle(2.0, "Waiting for Odometry (/rtabmap/odom)...")
                rate.sleep()
                continue
                
            if not self.global_path or not self.global_path.poses:
                rospy.logwarn_throttle(2.0, "Waiting for Global Path (/graph_planner/path/global_path)...")
                rate.sleep()
                continue
            
            # Find the closest point
            min_dist = float('inf')
            closest_idx = 0
            for i, p in enumerate(self.global_path.poses):
                d = math.hypot(p.pose.position.x - self.robot_x, p.pose.position.y - self.robot_y)
                if d < min_dist:
                    min_dist = d
                    closest_idx = i
            
            # Find lookahead point
            target_idx = closest_idx
            for i in range(closest_idx, len(self.global_path.poses)):
                d = math.hypot(self.global_path.poses[i].pose.position.x - self.robot_x,
                               self.global_path.poses[i].pose.position.y - self.robot_y)
                if d > self.lookahead_distance:
                    target_idx = i
                    break
            else:
                target_idx = len(self.global_path.poses) - 1

            target_p = self.global_path.poses[target_idx].pose.position
            dist_to_final = math.hypot(self.global_path.poses[-1].pose.position.x - self.robot_x,
                                       self.global_path.poses[-1].pose.position.y - self.robot_y)

            cmd = Twist()
            if dist_to_final < self.goal_tolerance:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                rospy.loginfo_throttle(2.0, "Goal reached successfully!")
            else:
                target_yaw = math.atan2(target_p.y - self.robot_y, target_p.x - self.robot_x)
                err_yaw = math.atan2(math.sin(target_yaw - self.robot_yaw), math.cos(target_yaw - self.robot_yaw))

            # Basic proportional control pure pursuit
            if abs(err_yaw) > 0.3:
                cmd.linear.x = 0.0 # Stop completely and turn
                cmd.angular.z = 0.8 if err_yaw > 0 else -0.8
            else:
                cmd.linear.x = 0.3
                cmd.angular.z = 2.0 * err_yaw
                
                # Check very basic collision with RTAB map 
                # Look directly ahead 0.3m. If it's an obstacle, stop/turn.
                obstacle_detected = False
                if self.map_data:
                    ahead_x = self.robot_x + math.cos(self.robot_yaw) * 0.4
                    ahead_y = self.robot_y + math.sin(self.robot_yaw) * 0.4
                    res = self.map_data.info.resolution
                    ox = self.map_data.info.origin.position.x
                    oy = self.map_data.info.origin.position.y
                    grid_x = int((ahead_x - ox) / res)
                    grid_y = int((ahead_y - oy) / res)
                    w = self.map_data.info.width
                    h = self.map_data.info.height
                    
                    if 0 <= grid_x < w and 0 <= grid_y < h:
                        val = self.map_data.data[grid_y * w + grid_x]
                        if val > 50:
                            cmd.linear.x = 0.0
                            cmd.angular.z = 1.0 # Turn in place to avoid
                            obstacle_detected = True

                if obstacle_detected:
                    rospy.logwarn_throttle(1.0, "Obstacle in front! Turning to avoid...")
                else:
                    rospy.loginfo_throttle(1.0, f"Following path. dist_to_final: {dist_to_final:.2f}, cmd_v: {cmd.linear.x:.2f}, cmd_w: {cmd.angular.z:.2f}")

            self.cmd_pub.publish(cmd)
            rate.sleep()

if __name__ == '__main__':
    try:
        p = RTABLocalPlanner()
        p.run()
    except rospy.ROSInterruptException:
        pass
