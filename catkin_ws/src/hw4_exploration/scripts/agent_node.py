#!/usr/bin/env python3
import rospy
import math
import tf
import tf2_ros
import random
from std_msgs.msg import String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from apriltag_ros.msg import AprilTagDetectionArray
from hw4_exploration.srv import AnalyzeSign, DetectShopfront

class AgentNode:
    def __init__(self):
        rospy.init_node("agent_node")
        
        self.state = "IDLE"
        self.target_shop = ""
        
        self.path_pub = rospy.Publisher("/graph_planner/path/global_path", Path, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.prompt_sub = rospy.Subscriber("/user_prompt", String, self.prompt_cb)
        self.tag_sub = rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.tag_cb)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        rospy.wait_for_service("/analyze_sign")
        rospy.wait_for_service("/detect_shopfront")
        self.analyze_srv = rospy.ServiceProxy("/analyze_sign", AnalyzeSign)
        self.shop_srv = rospy.ServiceProxy("/detect_shopfront", DetectShopfront)
        
        self.last_tag_time = rospy.Time(0)
        self.tag_cooldown = rospy.Duration(10.0) # ignore tags for 10 seconds after reading one
        
        self.last_shop_check_time = rospy.Time.now()
        self.shop_check_interval = rospy.Duration(8.0) # Check for shop every 8 seconds of exploring
        
        self.turn_end_time = rospy.Time(0)
        self.turn_dx = 0.0
        self.turn_dy = 0.0
        self.turn_dyaw = 0.0
        
        # Stuck detection and recovery variables
        self.stuck_check_time = rospy.Time.now()
        self.stuck_start_pos = None
        self.recovery_stage = 0
        self.recovery_end_time = rospy.Time(0)
        self.recovery_spin_speed = 0.0
        
        self.rate = rospy.Rate(10)
        
        # Publish initial state to ROS parameters
        self.set_state("IDLE")
        rospy.loginfo("Agent Node initialized. Waiting for user input on /user_prompt...")
        
    def set_state(self, new_state):
        self.state = new_state
        self.stuck_start_pos = None
        rospy.set_param("/exploration_state", new_state)
        rospy.loginfo(f"Agent state transitioned to: {new_state}")
        
    def prompt_cb(self, msg):
        cmd = msg.data.strip()
        if cmd == "STOP":
            rospy.loginfo("Received STOP command. Stopping exploration.")
            self.set_state("IDLE")
            self.stop_robot()
        else:
            self.target_shop = cmd
            rospy.loginfo(f"Received target: {self.target_shop}. Starting EXPLORE.")
            self.last_shop_check_time = rospy.Time.now()
            self.set_state("EXPLORE")
            
    def tag_cb(self, msg):
        if self.state == "EXPLORE":
            if len(msg.detections) > 0:
                if (rospy.Time.now() - self.last_tag_time) > self.tag_cooldown:
                    rospy.loginfo("Detected AprilTag! Stopping to read sign.")
                    self.set_state("READ_SIGN")
                    
    def stop_robot(self):
        # Publish empty path to stop control_space_planner
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()
        self.path_pub.publish(path)

    def publish_local_path(self, dx, dy, dyaw):
        try:
            trans = self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0), rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return
            
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()
        
        p0 = PoseStamped()
        p0.pose.position.x = trans.transform.translation.x
        p0.pose.position.y = trans.transform.translation.y
        p0.pose.orientation = trans.transform.rotation
        path.poses.append(p0)
        
        q = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
        euler = tf.transformations.euler_from_quaternion(q)
        yaw = euler[2]
        
        target_yaw = yaw + dyaw
        p1 = PoseStamped()
        p1.pose.position.x = p0.pose.position.x + dx * math.cos(yaw) - dy * math.sin(yaw)
        p1.pose.position.y = p0.pose.position.y + dx * math.sin(yaw) + dy * math.cos(yaw)
        q1 = tf.transformations.quaternion_from_euler(0, 0, target_yaw)
        p1.pose.orientation = Quaternion(*q1)
        path.poses.append(p1)
        
        self.path_pub.publish(path)

    def run(self):
        while not rospy.is_shutdown():
            # Support real-time pause control
            if rospy.get_param("/exploration_paused", False):
                self.stop_robot()
                self.rate.sleep()
                continue
                
            curr_x, curr_y = None, None
            try:
                trans = self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0), rospy.Duration(0.1))
                curr_x = trans.transform.translation.x
                curr_y = trans.transform.translation.y
            except Exception:
                pass
                
            if self.state == "IDLE":
                pass
                
            elif self.state == "EXPLORE":
                if (rospy.Time.now() - self.last_shop_check_time) > self.shop_check_interval:
                    self.stop_robot()
                    rospy.loginfo("Pausing exploration to check for shopfront...")
                    self.set_state("CHECK_SHOP")
                else:
                    self.publish_local_path(1.5, 0.0, 0.0) # publish point 1.5m straight ahead
                    
                    # Stuck detection
                    if curr_x is not None and curr_y is not None:
                        if self.stuck_start_pos is None:
                            self.stuck_start_pos = (curr_x, curr_y)
                            self.stuck_check_time = rospy.Time.now()
                        else:
                            dist_moved = math.sqrt((curr_x - self.stuck_start_pos[0])**2 + (curr_y - self.stuck_start_pos[1])**2)
                            if dist_moved > 0.15:
                                self.stuck_start_pos = (curr_x, curr_y)
                                self.stuck_check_time = rospy.Time.now()
                            elif (rospy.Time.now() - self.stuck_check_time) > rospy.Duration(3.0):
                                rospy.logwarn("Robot stuck / dead end detected! Initiating recovery sequence.")
                                self.stop_robot()
                                self.set_state("RECOVERY")
                                self.recovery_stage = 1 # Back up
                                self.recovery_end_time = rospy.Time.now() + rospy.Duration(2.0)
                                self.recovery_spin_speed = 0.6 if random.random() < 0.5 else -0.6
                                self.stuck_start_pos = None
                                
            elif self.state == "RECOVERY":
                if rospy.Time.now() > self.recovery_end_time:
                    if self.recovery_stage == 1:
                        rospy.loginfo("Recovery Stage 1 (Backup) complete. Starting Stage 2 (Spin).")
                        self.recovery_stage = 2
                        self.recovery_end_time = rospy.Time.now() + rospy.Duration(2.6)
                    else:
                        rospy.loginfo("Recovery complete. Resuming exploration.")
                        self.last_shop_check_time = rospy.Time.now()
                        self.set_state("EXPLORE")
                else:
                    twist = Twist()
                    if self.recovery_stage == 1:
                        twist.linear.x = -0.15 # back up
                        twist.angular.z = 0.0
                    else:
                        twist.linear.x = 0.0
                        twist.angular.z = self.recovery_spin_speed # spin left or right 90 degrees
                    self.cmd_pub.publish(twist)
                    
            elif self.state == "READ_SIGN":
                self.stop_robot()
                rospy.sleep(1.5) # stabilize camera blur
                
                rospy.loginfo("Calling /analyze_sign API...")
                res = self.analyze_srv(self.target_shop)
                direction = res.direction
                rospy.loginfo(f"Sign says go: {direction}")
                
                self.last_tag_time = rospy.Time.now()
                
                if direction == "LEFT":
                    self.turn_dx = 1.0
                    self.turn_dy = 1.0
                    self.turn_dyaw = math.pi / 2
                    self.turn_end_time = rospy.Time.now() + rospy.Duration(6.0)
                elif direction == "RIGHT":
                    self.turn_dx = 1.0
                    self.turn_dy = -1.0
                    self.turn_dyaw = -math.pi / 2
                    self.turn_end_time = rospy.Time.now() + rospy.Duration(6.0)
                else: # STRAIGHT or UNKNOWN
                    self.turn_dx = 2.0
                    self.turn_dy = 0.0
                    self.turn_dyaw = 0.0
                    self.turn_end_time = rospy.Time.now() + rospy.Duration(3.0)
                    
                self.set_state("TURN")
                
            elif self.state == "TURN":
                if rospy.Time.now() > self.turn_end_time:
                    self.set_state("EXPLORE")
                    self.last_shop_check_time = rospy.Time.now()
                else:
                    self.publish_local_path(self.turn_dx, self.turn_dy, self.turn_dyaw)
                    
            elif self.state == "CHECK_SHOP":
                rospy.sleep(1.0)
                rospy.loginfo("Calling /detect_shopfront API...")
                res = self.shop_srv(self.target_shop)
                
                if self.target_shop.upper() in ["", "EXPLORE", "MAPPING"]:
                    rospy.loginfo("Completed local shop scan. Resuming systematic exploration.")
                    self.last_shop_check_time = rospy.Time.now()
                    self.set_state("EXPLORE")
                elif res.is_found:
                    rospy.loginfo("SUCCESS! Arrived at target shop.")
                    self.set_state("ARRIVED")
                else:
                    rospy.loginfo("Shop not found yet. Resuming exploration.")
                    self.last_shop_check_time = rospy.Time.now()
                    self.set_state("EXPLORE")
                    
            elif self.state == "ARRIVED":
                self.stop_robot()
                
            self.rate.sleep()

if __name__ == '__main__':
    node = AgentNode()
    node.run()
