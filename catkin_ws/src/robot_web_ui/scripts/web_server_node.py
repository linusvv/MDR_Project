#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import threading
import yaml
import os
import re
import math
from flask import Flask, render_template, Response, request, jsonify
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Twist, PointStamped, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from rosgraph_msgs.msg import Log
from cv_bridge import CvBridge
import tf
import tf.transformations

import json
from std_msgs.msg import String
from gpt_llm_client.srv import LLMQuery, LLMQueryRequest
from std_srvs.srv import Empty as EmptySrv
from hw4_exploration.srv import DetectShopfront, DetectShopfrontRequest

try:
    from apriltag_ros.msg import AprilTagDetectionArray
except ImportError:
    AprilTagDetectionArray = None

app = Flask(__name__, static_folder='../static', template_folder='../templates')

class RobotWebServer:
    def __init__(self):
        rospy.init_node('robot_web_ui_server', anonymous=True)
        self.bridge = CvBridge()

        # Image & Map buffers
        self.color_image = None
        self.depth_image = None
        self.camera_info = None
        self.tag_detections = []
        self.grid_map = None
        self.selected_motion_pts = []
        self.selected_motion_frame = "map"
        
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.map_coverage_m2 = 0.0
        self.stores = []
        self.detected_tags = {}
        self.shop_categories = {}
        self.local_costmap = None
        self.motion_candidates = []
        self.planner_logs = []
        self.R = np.eye(2)
        self.T = np.array([0.0, -3.125])
        self.T_map_odom = np.eye(4)
        self.searching_tag = False
        self.search_thread = None
        self.navigating_to_tag = False
        self.planner_logs = []
        
        # Set default parameter: Local AI mode active on startup
        rospy.set_param("/use_local_ai", True)
        
        self.lock = threading.Lock()
        self.tf_listener = tf.TransformListener()

        # Load AprilTag bundles configuration & Store coordinates
        self.yaml_path = "/home/linusv/project_5/HW4/tags.yaml"
        self.bundles = self.load_bundles(self.yaml_path)
        self.load_stores_from_txt()
        self.load_tag_true_poses()

        # ROS Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.user_prompt_pub = rospy.Publisher('/user_prompt', String, queue_size=5)
        self.path_pub = rospy.Publisher('/graph_planner/path/global_path', Path, queue_size=5)

        # ROS Subscribers
        rospy.Subscriber('/camera/color/image_raw', Image, self.color_cb)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_cb)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.cam_info_cb)
        rospy.Subscriber('/rtabmap/grid_map', OccupancyGrid, self.map_cb)
        rospy.Subscriber('/map/local_map/obstacle', OccupancyGrid, self.local_map_cb)
        rospy.Subscriber('/semantic_observations', String, self.semantic_cb)
        rospy.Subscriber('/points/selected_motion', PointCloud2, self.motion_cb)
        rospy.Subscriber('/points/motion_primitives', PointCloud2, self.motion_candidates_cb)
        rospy.Subscriber('/rosout', Log, self.rosout_cb)
        
        if AprilTagDetectionArray is not None:
            rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.tags_cb)

        rospy.loginfo("Web Server Node initialized.")

    def stop_robot(self):
        """Immediately stops all autonomous navigation and the robot movement."""
        with self.lock:
            self.searching_tag = False
            self.navigating_to_tag = False
        
        # 1. Stop the global/graph planner
        rospy.set_param("/exploration_state", "IDLE")
        
        # 2. Clear current path to stop local C++ planner
        empty_path = Path()
        empty_path.header.frame_id = "map"
        empty_path.header.stamp = rospy.Time.now()
        self.path_pub.publish(empty_path)
        
        # 3. Publish zero velocity
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        
        rospy.logwarn("[EMERGENCY STOP] Robot and planners stopped.")

    def color_cb(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Draw AprilTags if we have camera info
            with self.lock:
                if self.camera_info is not None and len(self.tag_detections) > 0:
                    cv_image = self.draw_tags(cv_image)
                    
                is_paused = rospy.get_param("/exploration_paused", False)
                explore_state = rospy.get_param("/exploration_state", "IDLE")
                if not is_paused and explore_state == "EXPLORE" and hasattr(self, 'selected_motion_pts') and self.camera_info is not None:
                    cv_image = self.draw_path(cv_image)
                    
            self.color_image = cv_image
        except Exception as e:
            rospy.logerr(f"Color Image Error: {e}")

    def depth_cb(self, msg):
        try:
            # Depth images are typically 16UC1 or 32FC1. We normalize to 8-bit for visualization
            cv_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            cv_image = np.nan_to_num(cv_image, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize to 0-255
            max_val = np.max(cv_image)
            if max_val > 0:
                cv_image = (cv_image / max_val * 255.0).astype(np.uint8)
            else:
                cv_image = cv_image.astype(np.uint8)
                
            # Apply colormap for better visualization
            cv_image_color = cv2.applyColorMap(cv_image, cv2.COLORMAP_JET)
            self.depth_image = cv_image_color
        except Exception as e:
            rospy.logerr(f"Depth Image Error: {e}")

    def cam_info_cb(self, msg):
        with self.lock:
            if self.camera_info is None:
                self.camera_info = msg

    def tags_cb(self, msg):
        with self.lock:
            self.tag_detections = msg.detections
            
            for detection in msg.detections:
                if not detection.id:
                    continue
                
                # Determine the TF frame name for this detection
                # If it's a bundle, use the bundle name. Otherwise use tag_ID.
                ids_tuple = tuple(sorted(detection.id))
                if ids_tuple in self.bundles:
                    tag_frame = self.bundles[ids_tuple]['name']
                else:
                    tag_frame = f"tag_{detection.id[0]}"
                
                tag_id = detection.id[0]
                
                # 1. First try to look up the tag's TF frame directly in odom
                if self.tf_listener.canTransform('odom', tag_frame, rospy.Time(0)):
                    try:
                        (trans, rot) = self.tf_listener.lookupTransform('odom', tag_frame, rospy.Time(0))
                        self.detected_tags[tag_id] = (trans[0], trans[1])
                        continue
                    except Exception:
                        pass
                
                # 2. If the tag frame is not directly broadcast, compute it using the camera's TF in odom
                frame_id = detection.pose.header.frame_id
                if self.tf_listener.canTransform('odom', frame_id, rospy.Time(0)):
                    try:
                        (trans, rot) = self.tf_listener.lookupTransform('odom', frame_id, rospy.Time(0))
                        
                        p = detection.pose.pose.pose.position
                        q = detection.pose.pose.pose.orientation
                        
                        T_odom_cam = tf.transformations.quaternion_matrix(rot)
                        T_odom_cam[:3, 3] = trans
                        
                        T_cam_tag = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
                        T_cam_tag[:3, 3] = [p.x, p.y, p.z]
                        
                        T_odom_tag = np.dot(T_odom_cam, T_cam_tag)
                        tx = T_odom_tag[0, 3]
                        ty = T_odom_tag[1, 3]
                        
                        self.detected_tags[tag_id] = (tx, ty)
                    except Exception:
                        pass

    def motion_cb(self, msg):
        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append((p[0], p[1], p[2]))
        with self.lock:
            self.selected_motion_pts = pts
            self.selected_motion_frame = msg.header.frame_id

    def motion_candidates_cb(self, msg):
        candidates = []
        # Motion primitives are published as a point cloud where each point's intensity (Z) is the same for a primitive
        last_primitive_id = None
        current_primitive = []
        
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pid = p[2]
            if last_primitive_id is not None and pid != last_primitive_id:
                candidates.append(current_primitive)
                current_primitive = []
            
            current_primitive.append((p[0], p[1]))
            last_primitive_id = pid
            
        if current_primitive:
            candidates.append(current_primitive)
            
        with self.lock:
            self.motion_candidates = candidates

    def rosout_cb(self, msg):
        """Filters logs to only keep those from the motion planner and related safety services."""
        # Relevant nodes for navigation safety
        relevant_nodes = [
            "/control_space_planner_node", 
            "/heightmap_costmap_node",
            "/heightmap_node",
            "/agent_node"
        ]
        if msg.name in relevant_nodes:
            with self.lock:
                # Format: [NodeName] Message
                log_entry = f"[{msg.name.split('/')[-1]}] {msg.msg}"
                self.planner_logs.append(log_entry)
                # Keep only last 50 logs to avoid memory bloat
                if len(self.planner_logs) > 50:
                    self.planner_logs.pop(0)

    def draw_path(self, cv_image):
        if not self.selected_motion_pts or self.camera_info is None:
            return cv_image
            
        K = np.array(self.camera_info.K).reshape((3, 3))
        cam_frame = self.camera_info.header.frame_id
        source_frame = self.selected_motion_frame
        
        if not self.tf_listener.canTransform(cam_frame, source_frame, rospy.Time(0)):
            return cv_image
            
        pts_2d = []
        for (x, y, z) in self.selected_motion_pts:
            ps = PointStamped()
            ps.header.frame_id = source_frame
            ps.header.stamp = rospy.Time(0)
            ps.point.x = x
            ps.point.y = y
            ps.point.z = 0.2  # Approximate height of the path above ground
            
            try:
                ps_cam = self.tf_listener.transformPoint(cam_frame, ps)
                if ps_cam.point.z > 0.1: # Point is in front of camera
                    u = int(K[0,0] * ps_cam.point.x / ps_cam.point.z + K[0,2])
                    v = int(K[1,1] * ps_cam.point.y / ps_cam.point.z + K[1,2])
                    
                    if 0 <= u < cv_image.shape[1] and 0 <= v < cv_image.shape[0]:
                        pts_2d.append((u, v))
            except tf.Exception:
                pass
                
        return cv_image

    def get_current_robot_pose(self):
        if self.tf_listener.canTransform('map', 'base_footprint', rospy.Time(0)):
            try:
                (trans, rot) = self.tf_listener.lookupTransform('map', 'base_footprint', rospy.Time(0))
                rx, ry = trans[0], trans[1]
                euler = tf.transformations.euler_from_quaternion(rot)
                yaw = euler[2]
                return rx, ry, yaw
            except Exception:
                pass
        return getattr(self, 'robot_x', 0.0), getattr(self, 'robot_y', 0.0), getattr(self, 'robot_yaw', 0.0)

    def find_tag_thread(self):
        rospy.loginfo("[Find Tag] Starting tag search process...")
        rate = rospy.Rate(10)
        scan_duration = 15.8
        tag_found = False

        while not rospy.is_shutdown() and self.searching_tag:
            # ================= PHASE 1: SCAN TURN =================
            rospy.loginfo("[Find Tag] Starting scan-turn phase...")
            # Disable planner during scanning
            rospy.set_param("/exploration_state", "IDLE")
            path = Path()
            path.header.frame_id = "map"
            path.header.stamp = rospy.Time.now()
            self.path_pub.publish(path) # Clear path
            self.cmd_vel_pub.publish(Twist()) # Stop movement
            rospy.sleep(0.2)
            
            start_time = rospy.Time.now()
            cmd = Twist()
            cmd.angular.z = 0.4 # Slowly turn in place
            
            while not rospy.is_shutdown() and self.searching_tag and (rospy.Time.now() - start_time).to_sec() < scan_duration:
                # Check for tag detection
                with self.lock:
                    detections_count = len(self.tag_detections) if hasattr(self, 'tag_detections') else 0
                if detections_count > 0:
                    rospy.loginfo("[Find Tag] Tag detected during scan phase!")
                    tag_found = True
                    break
                self.cmd_vel_pub.publish(cmd)
                rate.sleep()
                
            self.cmd_vel_pub.publish(Twist()) # Stop turn
            
            if tag_found or not self.searching_tag:
                break
                
            # ================= PHASE 2: WANDER TRAVEL =================
            rospy.loginfo("[Find Tag] No tag found during scan. Commencing safe wandering...")
            # Enable C++ planner
            rospy.set_param("/exploration_state", "EXPLORE")
            rospy.set_param("/exploration_paused", False)
            
            # Select random global heading direction to explore
            import random
            explore_angle = random.uniform(-math.pi, math.pi)
            rospy.loginfo(f"[Find Tag] Selected wandering heading: {explore_angle:.2f} rad.")
            
            # Track position to check for progress / stuck condition
            rx, ry, _ = self.get_current_robot_pose()
            last_progress_x = rx
            last_progress_y = ry
            last_progress_time = rospy.Time.now()
            
            # Wander for a maximum duration (e.g. 15.0 seconds) before doing another scan
            wander_start_time = rospy.Time.now()
            wander_duration = 15.0
            
            while not rospy.is_shutdown() and self.searching_tag and (rospy.Time.now() - wander_start_time).to_sec() < wander_duration:
                # Update robot pose
                rx, ry, ryaw = self.get_current_robot_pose()
                self.robot_x = rx
                self.robot_y = ry
                self.robot_yaw = ryaw
                
                # Check for tag detection
                with self.lock:
                    detections_count = len(self.tag_detections) if hasattr(self, 'tag_detections') else 0
                if detections_count > 0:
                    rospy.loginfo("[Find Tag] Tag detected while traveling!")
                    tag_found = True
                    break
                    
                # Project goal 2.5 meters ahead along explore_angle
                goal_x = rx + 2.5 * math.cos(explore_angle)
                goal_y = ry + 2.5 * math.sin(explore_angle)
                
                self.publish_path_to_goal(goal_x, goal_y)
                
                # Check progress every 1.0 second
                now = rospy.Time.now()
                if (now - last_progress_time).to_sec() >= 1.0:
                    dist_moved = math.hypot(rx - last_progress_x, ry - last_progress_y)
                    if dist_moved >= 0.08:
                        last_progress_x = rx
                        last_progress_y = ry
                        last_progress_time = now
                    elif (now - last_progress_time).to_sec() > 3.0:
                        # Stuck against a wall/corner, pick a new direction
                        explore_angle = random.uniform(-math.pi, math.pi)
                        rospy.logwarn(f"[Find Tag] Blocked/No progress. Changing heading to {explore_angle:.2f} rad.")
                        last_progress_x = rx
                        last_progress_y = ry
                        last_progress_time = rospy.Time.now()
                        
                rate.sleep()
                
            if tag_found or not self.searching_tag:
                break
                
        # Cleanup
        self.searching_tag = False
        self.stop_search()
        if tag_found:
            rospy.loginfo("[Find Tag] Search successfully completed.")
        else:
            rospy.loginfo("[Find Tag] Search finished/stopped.")

    def start_navigation_to_tag(self, tag_name):
        # Cancel any active search
        self.searching_tag = False
        
        # Stop any existing navigation thread
        self.navigating_to_tag = False
        if hasattr(self, 'nav_thread') and self.nav_thread.is_alive():
            self.nav_thread.join(timeout=1.0)
            
        self.navigating_to_tag = True
        self.nav_thread = threading.Thread(target=self.nav_to_tag_thread, args=(tag_name,))
        self.nav_thread.daemon = True
        self.nav_thread.start()
        return True

    def nav_to_tag_thread(self, tag_name):
        rospy.loginfo(f"[Tag Nav] Starting navigation to landmark {tag_name}...")
        
        # Enable C++ planner
        rospy.set_param("/exploration_state", "EXPLORE")
        rospy.set_param("/exploration_paused", False)
        
        # Get tag pose
        pose_info = self.tag_true_poses[tag_name]
        tag_pt = np.array([pose_info[0], pose_info[1]])
        psi_deg = pose_info[2]
        psi_rad = math.radians(psi_deg)
        
        # Transform true pose to SLAM map coordinates
        with self.lock:
            R = self.R
            T = self.T
        
        tag_map = np.dot(R, tag_pt) + T
        tag_x, tag_y = tag_map[0], tag_map[1]
        
        # Transform heading (psi) to map coordinates
        # Map heading = True heading + rotation angle from R
        rot_angle = math.atan2(R[1, 0], R[0, 0])
        psi_rad_map = psi_rad + rot_angle
        
        # Offset to stop in front of tag (0.8 meters)
        offset = 0.8
        gx = tag_x + offset * math.cos(psi_rad_map)
        gy = tag_y + offset * math.sin(psi_rad_map)
        gyaw = math.atan2(-math.sin(psi_rad_map), -math.cos(psi_rad_map)) # Point at the tag (face opposite to normal)
        
        rospy.loginfo(f"[Tag Nav] Target Pose in Map: ({gx:.2f}, {gy:.2f}, heading: {math.degrees(gyaw):.1f}°)")
        
        rate = rospy.Rate(10)
        
        # Track position for progress check
        rx, ry, _ = self.get_current_robot_pose()
        last_progress_x = rx
        last_progress_y = ry
        last_progress_time = rospy.Time.now()
        start_time = rospy.Time.now()
        
        use_waypoint = False
        recovery_waypoint = None
        
        while not rospy.is_shutdown() and self.navigating_to_tag:
            rx, ry, ryaw = self.get_current_robot_pose()
            self.robot_x = rx
            self.robot_y = ry
            self.robot_yaw = ryaw
            
            # Distance to the target position
            dist = math.hypot(rx - gx, ry - gy)
            
            # Check if arrived at position
            if dist < 0.9:
                # Arrived at position! Wait a bit for alignment
                rospy.loginfo("[Tag Nav] Arrived at target position. Waiting for alignment...")
                rospy.sleep(3.0)
                rospy.loginfo("[Tag Nav] Navigation and alignment completed successfully.")
                break
                
            # If we are close to the waypoint, clear it to resume direct path to goal
            if use_waypoint and recovery_waypoint is not None:
                dist_to_wp = math.hypot(rx - recovery_waypoint[0], ry - recovery_waypoint[1])
                if dist_to_wp < 0.6:
                    rospy.loginfo("[Tag Nav] Arrived at recovery waypoint. Clearing waypoint.")
                    use_waypoint = False
                    recovery_waypoint = None
                    
            # Publish path to goal
            path = Path()
            path.header.frame_id = "map"
            path.header.stamp = rospy.Time.now()
            
            p0 = PoseStamped()
            p0.header.frame_id = "map"
            p0.pose.position.x = rx
            p0.pose.position.y = ry
            
            # Orientation facing next point
            if use_waypoint and recovery_waypoint is not None:
                target_x, target_y = recovery_waypoint
            else:
                target_x, target_y = gx, gy
            angle_to_next = math.atan2(target_y - ry, target_x - rx)
            q0 = tf.transformations.quaternion_from_euler(0, 0, angle_to_next)
            p0.pose.orientation = Quaternion(*q0)
            path.poses.append(p0)
            
            if use_waypoint and recovery_waypoint is not None:
                p_wp = PoseStamped()
                p_wp.header.frame_id = "map"
                p_wp.pose.position.x = recovery_waypoint[0]
                p_wp.pose.position.y = recovery_waypoint[1]
                q_wp = tf.transformations.quaternion_from_euler(0, 0, angle_to_next)
                p_wp.pose.orientation = Quaternion(*q_wp)
                path.poses.append(p_wp)
                
            p1 = PoseStamped()
            p1.header.frame_id = "map"
            p1.pose.position.x = gx
            p1.pose.position.y = gy
            q1 = tf.transformations.quaternion_from_euler(0, 0, gyaw)
            p1.pose.orientation = Quaternion(*q1)
            path.poses.append(p1)
            
            self.path_pub.publish(path)
            
            # Progress check (stuck detection)
            now = rospy.Time.now()
            if (now - last_progress_time).to_sec() >= 1.0:
                dist_moved = math.hypot(rx - last_progress_x, ry - last_progress_y)
                if dist_moved >= 0.08:
                    last_progress_x = rx
                    last_progress_y = ry
                    last_progress_time = now
                elif (now - last_progress_time).to_sec() > 3.0:
                    rospy.logwarn("[Tag Nav] Stuck/No progress made for 3 seconds. Initiating recovery sequence...")
                    
                    # 1. Disable C++ planner by setting state to RECOVERY
                    rospy.set_param("/exploration_state", "RECOVERY")
                    
                    # 2. Go backwards for 0.5 seconds
                    backup_end = rospy.Time.now() + rospy.Duration(0.5)
                    while rospy.Time.now() < backup_end and not rospy.is_shutdown() and self.navigating_to_tag:
                        twist = Twist()
                        twist.linear.x = -0.15
                        twist.angular.z = 0.0
                        self.cmd_vel_pub.publish(twist)
                        rospy.sleep(0.05)
                        
                    # Stop backup
                    self.cmd_vel_pub.publish(Twist())
                    
                    # 3. Spin in place (turn)
                    import random
                    spin_dir = 0.6 if random.random() < 0.5 else -0.6
                    spin_end = rospy.Time.now() + rospy.Duration(1.5)
                    while rospy.Time.now() < spin_end and not rospy.is_shutdown() and self.navigating_to_tag:
                        twist = Twist()
                        twist.linear.x = 0.0
                        twist.angular.z = spin_dir
                        self.cmd_vel_pub.publish(twist)
                        rospy.sleep(0.05)
                        
                    # Stop spin
                    self.cmd_vel_pub.publish(Twist())
                    
                    # Get new pose after spin
                    rx, ry, ryaw = self.get_current_robot_pose()
                    
                    # 4. Try a different way: set a waypoint 1.5m ahead of the new yaw
                    waypoint_x = rx + 1.5 * math.cos(ryaw)
                    waypoint_y = ry + 1.5 * math.sin(ryaw)
                    recovery_waypoint = (waypoint_x, waypoint_y)
                    use_waypoint = True
                    rospy.loginfo(f"[Tag Nav] Set recovery waypoint at ({waypoint_x:.2f}, {waypoint_y:.2f})")
                    
                    # 5. Re-enable C++ planner
                    rospy.set_param("/exploration_state", "EXPLORE")
                    
                    # Reset progress tracker
                    last_progress_x = rx
                    last_progress_y = ry
                    last_progress_time = rospy.Time.now()
                    
            if (now - start_time).to_sec() > 120.0:
                rospy.logwarn("[Tag Nav] Navigation timeout (120 seconds).")
                break
                
            rate.sleep()
            
        self.navigating_to_tag = False
        self.stop_search()

    def choose_random_free_goal(self):
        rx, ry, _ = self.get_current_robot_pose()
        
        with self.lock:
            grid_map = self.grid_map if hasattr(self, 'grid_map') else None
            
        if grid_map is None:
            # Fallback if map not ready
            import random
            angle = random.uniform(0, 2 * math.pi)
            gx = rx + 2.0 * math.cos(angle)
            gy = ry + 2.0 * math.sin(angle)
            return gx, gy
            
        width = grid_map.info.width
        height = grid_map.info.height
        resolution = grid_map.info.resolution
        origin = grid_map.info.origin
        
        if width == 0 or height == 0:
            import random
            angle = random.uniform(0, 2 * math.pi)
            gx = rx + 2.0 * math.cos(angle)
            gy = ry + 2.0 * math.sin(angle)
            return gx, gy
            
        import random
        map_data = np.array(grid_map.data, dtype=np.int8).reshape((height, width))
        
        for _ in range(100):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(1.5, 3.5)
            
            gx = rx + distance * math.cos(angle)
            gy = ry + distance * math.sin(angle)
            
            col = int((gx - origin.position.x) / resolution)
            row = int((gy - origin.position.y) / resolution)
            
            if 0 <= col < width and 0 <= row < height:
                if map_data[row, col] == 0:
                    inflation_ok = True
                    check_r = 2
                    for r in range(-check_r, check_r + 1):
                        for c in range(-check_r, check_r + 1):
                            nr = row + r
                            nc = col + c
                            if 0 <= nc < width and 0 <= nr < height:
                                if map_data[nr, nc] != 0:
                                    inflation_ok = False
                                    break
                        if not inflation_ok:
                            break
                            
                    if inflation_ok:
                        return gx, gy
                        
        # Final fallback
        angle = random.uniform(0, 2 * math.pi)
        gx = rx + 2.0 * math.cos(angle)
        gy = ry + 2.0 * math.sin(angle)
        return gx, gy

    def publish_path_to_goal(self, gx, gy):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()
        
        rx, ry, _ = self.get_current_robot_pose()
        
        p0 = PoseStamped()
        p0.header.frame_id = "map"
        p0.pose.position.x = rx
        p0.pose.position.y = ry
        # Set orientation facing the goal to guide local planner rotation
        angle_to_goal = math.atan2(gy - ry, gx - rx)
        q = tf.transformations.quaternion_from_euler(0, 0, angle_to_goal)
        p0.pose.orientation.x = q[0]
        p0.pose.orientation.y = q[1]
        p0.pose.orientation.z = q[2]
        p0.pose.orientation.w = q[3]
        path.poses.append(p0)
        
        p1 = PoseStamped()
        p1.header.frame_id = "map"
        p1.pose.position.x = gx
        p1.pose.position.y = gy
        p1.pose.orientation.x = q[0]
        p1.pose.orientation.y = q[1]
        p1.pose.orientation.z = q[2]
        p1.pose.orientation.w = q[3]
        path.poses.append(p1)
        
        self.path_pub.publish(path)

    def stop_search(self):
        rospy.set_param("/exploration_state", "IDLE")
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()
        self.path_pub.publish(path)
        self.cmd_vel_pub.publish(Twist())


    def project_store_to_map(self, store):
        store_pt = np.array([store[0], store[1]])
        with self.lock:
            R = self.R
            T = self.T
        store_map = np.dot(R, store_pt) + T
        return store_map[0], store_map[1]

    def semantic_cb(self, msg):
        try:
            obs = json.loads(msg.data)
            if not obs.get("has_signboard", False):
                category = obs.get("category", "Unknown")
                storefront = obs.get("storefront", category)
                
                # Find the closest store among the 8 loaded store coordinates
                with self.lock:
                    rx, ry, yaw = self.robot_x, self.robot_y, self.robot_yaw
                    
                closest_idx = -1
                best_score = float('inf')
                
                for idx, store in enumerate(self.stores):
                    s_x, s_y = self.project_store_to_map(store)
                    dist = np.hypot(rx - s_x, ry - s_y)
                    if dist < 1.5:
                        angle_to_store = math.atan2(s_y - ry, s_x - rx)
                        # Normalize angle diff to [-pi, pi]
                        angle_diff = (angle_to_store - yaw + math.pi) % (2 * math.pi) - math.pi
                        angle_diff = abs(angle_diff)
                        
                        # Score prioritizes stores we are directly looking at
                        score = dist + 2.0 * angle_diff
                        
                        if score < best_score:
                            best_score = score
                            closest_idx = idx
                            
                if closest_idx != -1 and best_score < 3.0:
                    self.shop_categories[closest_idx] = {
                        "storefront": storefront,
                        "category": category
                    }
                    rospy.loginfo(f"Successfully mapped Shop S{closest_idx+1} to storefront {storefront} ({category})")
        except Exception as e:
            rospy.logerr(f"Error in web server semantic_cb: {e}")



    def load_bundles(self, yaml_path):
        # Try multiple potential paths
        paths = [
            yaml_path,
            "/home/linusv/project_5/HW4/tags.yaml",
            "/home/linusv/project_5/catkin_ws/src/AprilTagLocalization/tags.yaml",
            os.path.join(os.path.dirname(__file__), '../../AprilTagLocalization/tags.yaml')
        ]
        
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        config = yaml.safe_load(f)
                    bundles = {}
                    if config and 'tag_bundles' in config:
                        for bundle in config['tag_bundles']:
                            name = bundle.get('name', '')
                            layout = bundle.get('layout', [])
                            # Create a sorted tuple of tag IDs in this bundle
                            ids = tuple(sorted([item['id'] for item in layout]))
                            bundles[ids] = {
                                'name': name,
                                'layout': {item['id']: item for item in layout}
                            }
                        rospy.loginfo(f"Successfully loaded {len(bundles)} AprilTag bundles from {path}")
                        return bundles
                except Exception as e:
                    rospy.logwarn(f"Failed to load yaml from {path}: {e}")
        rospy.logwarn("Could not load AprilTag bundles config. Falling back to standalone tags only.")
        return {}

    def draw_tags(self, img):
        K = np.array(self.camera_info.K).reshape((3, 3))
        dist_coeffs = np.zeros((4,1)) # Assume rectified image
        
        def draw_single_tag(img, tag_id, size, rvec, tvec):
            # Define object points: corners and normal vector tip
            # Tag frame: Z out (normal), X right, Y down. Center is 0,0,0
            half = size / 2.0
            obj_pts = np.array([
                [-half, -half, 0], # Top-left
                [ half, -half, 0], # Top-right
                [ half,  half, 0], # Bottom-right
                [-half,  half, 0], # Bottom-left
                [    0,     0, 0], # Center
                [    0,     0, 0.3] # Normal vector tip (30cm length)
            ], dtype=np.float32)
            
            # Project 3D points to 2D image plane
            img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist_coeffs)
            img_pts = np.int32(img_pts).reshape(-1, 2)
            
            # Draw boundary
            cv2.polylines(img, [img_pts[:4]], True, (0, 255, 255), 2)
            
            # Draw normal vector (Arrow from center to tip)
            center = tuple(img_pts[4])
            tip = tuple(img_pts[5])
            cv2.arrowedLine(img, center, tip, (0, 255, 0), 2, tipLength=0.2)
            
            # Draw ID
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"ID: {tag_id}", (img_pts[0][0], img_pts[0][1] - 10), font, 0.7, (0, 255, 255), 2)
            
        for detection in self.tag_detections:
            pose = detection.pose.pose.pose
            det_ids = tuple(sorted(list(detection.id)))
            bundle = self.bundles.get(det_ids)
            
            if bundle:
                for tag_id, tag_cfg in bundle['layout'].items():
                    size = tag_cfg.get('size', 0.15)
                    # Compute relative transform
                    qx = tag_cfg.get('qx', 0.0)
                    qy = tag_cfg.get('qy', 0.0)
                    qz = tag_cfg.get('qz', 0.0)
                    qw = tag_cfg.get('qw', 1.0)
                    tx = tag_cfg.get('x', 0.0)
                    ty = tag_cfg.get('y', 0.0)
                    tz = tag_cfg.get('z', 0.0)
                    
                    T_bundle_tag = tf.transformations.quaternion_matrix([qx, qy, qz, qw])
                    T_bundle_tag[0, 3] = tx
                    T_bundle_tag[1, 3] = ty
                    T_bundle_tag[2, 3] = tz
                    
                    q_bundle = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                    T_cam_bundle = tf.transformations.quaternion_matrix(q_bundle)
                    T_cam_bundle[0, 3] = pose.position.x
                    T_cam_bundle[1, 3] = pose.position.y
                    T_cam_bundle[2, 3] = pose.position.z
                    
                    T_cam_tag = np.dot(T_cam_bundle, T_bundle_tag)
                    
                    tvec = T_cam_tag[:3, 3].reshape(3, 1)
                    R = T_cam_tag[:3, :3]
                    rvec, _ = cv2.Rodrigues(R)
                    
                    draw_single_tag(img, tag_id, size, rvec, tvec)
            else:
                tag_id = detection.id[0]
                size = detection.size[0] if len(detection.size) > 0 else 0.15
                
                tvec = np.array([[pose.position.x], [pose.position.y], [pose.position.z]])
                q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                R = tf.transformations.quaternion_matrix(q)[:3, :3]
                rvec, _ = cv2.Rodrigues(R)
                
                draw_single_tag(img, tag_id, size, rvec, tvec)
            
        return img

    def generate_frames(self, camera_type):
        rate = rospy.Rate(30) # 30 FPS
        while not rospy.is_shutdown():
            frame = None
            if camera_type == 'color' and self.color_image is not None:
                frame = self.color_image
            elif camera_type == 'depth' and self.depth_image is not None:
                frame = self.depth_image
                
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            rate.sleep()

    def map_cb(self, msg):
        with self.lock:
            self.grid_map = msg

    def local_map_cb(self, msg):
        with self.lock:
            self.local_costmap = msg

    def load_stores_from_txt(self):
        file_path = "/home/linusv/project_5/HW4/Store coordinates.txt"
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    match = re.search(r'\(([^,]+),\s*([^)]+)\)', line)
                    if match:
                        self.stores.append((float(match.group(1)), float(match.group(2))))
                rospy.loginfo(f"Web UI loaded {len(self.stores)} stores.")
            except Exception as e:
                rospy.logwarn(f"Failed to load stores in Web UI: {e}")

    def load_tag_true_poses(self):
        active_yaml = "/home/linusv/project_5/catkin_ws/src/AprilTagLocalization/config/2025/re540_n1_room111.yaml"
        self.tag_true_poses = {}
        if os.path.exists(active_yaml):
            try:
                with open(active_yaml, 'r') as f:
                    config = yaml.safe_load(f)
                if config and 'TAG_TRUE_RT' in config and 'TAGS' in config['TAG_TRUE_RT']:
                    for tag_def in config['TAG_TRUE_RT']['TAGS']:
                        name = tag_def[0]
                        x_true = float(tag_def[2])
                        y_true = float(tag_def[3])
                        psi_deg = float(tag_def[5]) if len(tag_def) > 5 else 0.0
                        self.tag_true_poses[name] = (x_true, y_true, psi_deg)
                rospy.loginfo(f"Web UI loaded true poses for {len(self.tag_true_poses)} signboards.")
            except Exception as e:
                rospy.logwarn(f"Failed to load true poses: {e}")

    def estimate_rigid_transform_2d(self, pts_true, pts_meas):
        n = len(pts_true)
        if n == 0:
            return self.R, self.T
        elif n == 1:
            tx = pts_meas[0][0] - pts_true[0][0]
            ty = pts_meas[0][1] - pts_true[0][1]
            return np.eye(2), np.array([tx, ty])
            
        pts_true = np.array(pts_true)
        pts_meas = np.array(pts_meas)
        
        centroid_true = np.mean(pts_true, axis=0)
        centroid_meas = np.mean(pts_meas, axis=0)
        
        pts_true_c = pts_true - centroid_true
        pts_meas_c = pts_meas - centroid_meas
        
        H = np.dot(pts_true_c.T, pts_meas_c)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = np.dot(Vt.T, U.T)
            
        T = centroid_meas - np.dot(R, centroid_true)
        return R, T

    def generate_map_frames(self):
        rate = rospy.Rate(5) # 5 Hz is perfect
        while not rospy.is_shutdown():
            map_img = None
            with self.lock:
                grid_map = self.grid_map
                
            if grid_map is not None:
                width = grid_map.info.width
                height = grid_map.info.height
                resolution = grid_map.info.resolution
                origin = grid_map.info.origin
                
                if width > 0 and height > 0:
                    data = np.array(grid_map.data, dtype=np.int8).reshape((height, width))
                    
                    # render map: 
                    # -1 (unknown) -> slate-800 [59, 41, 30] BGR
                    # 0 (free) -> slate-950 [42, 23, 15] BGR
                    # 100 (occupied) -> slate-300 [184, 163, 148] BGR
                    color_map = np.zeros((height, width, 3), dtype=np.uint8)
                    color_map[data == -1] = [59, 41, 30]
                    color_map[data == 0] = [42, 23, 15]
                    color_map[(data > 0) & (data <= 100)] = [184, 163, 148]
                    
                    # Flip vertically for image coordinates
                    color_map = cv2.flip(color_map, 0)
                    
                    # Try to lookup robot's pose in map frame
                    if self.tf_listener.canTransform('map', 'base_footprint', rospy.Time(0)):
                        try:
                            (trans, rot) = self.tf_listener.lookupTransform('map', 'base_footprint', rospy.Time(0))
                            rx, ry = trans[0], trans[1]
                            euler = tf.transformations.euler_from_quaternion(rot)
                            yaw = euler[2]
                            
                            # Save robot pose for stats API
                            self.robot_x = rx
                            self.robot_y = ry
                            self.robot_yaw = yaw
                            
                            # Translate to pixel coordinates
                            r_col = int((rx - origin.position.x) / resolution)
                            r_row = int((ry - origin.position.y) / resolution)
                            r_row_flipped = height - 1 - r_row
                            
                            if 0 <= r_col < width and 0 <= r_row_flipped < height:
                                # Path Trail trajectory
                                if not hasattr(self, 'trail'):
                                    self.trail = []
                                self.trail.append((r_col, r_row_flipped))
                                if len(self.trail) > 1000:
                                    self.trail.pop(0)
                                    
                                # Draw Trail (slate-500 line)
                                for j in range(1, len(self.trail)):
                                    cv2.line(color_map, self.trail[j-1], self.trail[j], (100, 116, 139), 1)
                                    
                                # Draw robot body (solid rose-red circle)
                                cv2.circle(color_map, (r_col, r_row_flipped), 8, (68, 68, 239), -1)
                                cv2.circle(color_map, (r_col, r_row_flipped), 8, (255, 255, 255), 1)
                                
                                # Draw heading indicator
                                render_yaw = -yaw
                                arrow_len = 12
                                arrow_end_x = int(r_col + arrow_len * np.cos(render_yaw))
                                arrow_end_y = int(r_row_flipped + arrow_len * np.sin(render_yaw))
                                cv2.arrowedLine(color_map, (r_col, r_row_flipped), (arrow_end_x, arrow_end_y), (255, 255, 255), 2, tipLength=0.3)
                        except Exception:
                            pass
                        
                    # Build matched point sets for 2D rigid transform estimation
                    pts_true = []
                    pts_meas = []
                    
                    # Try to lookup transform map -> odom
                    if self.tf_listener.canTransform('map', 'odom', rospy.Time(0)):
                        try:
                            (trans_mo, rot_mo) = self.tf_listener.lookupTransform('map', 'odom', rospy.Time(0))
                            T_map_odom = tf.transformations.quaternion_matrix(rot_mo)
                            T_map_odom[:3, 3] = trans_mo
                            self.T_map_odom = T_map_odom
                        except Exception:
                            pass
                    T_map_odom = self.T_map_odom
                        
                    for tag_id, tag_pose_odom in self.detected_tags.items():
                        signboard_name = None
                        for ids, bundle in self.bundles.items():
                            if tag_id in ids:
                                signboard_name = bundle['name']
                                break
                        if signboard_name and signboard_name in self.tag_true_poses:
                            # Transform tag_pose_odom from odom to map
                            pt_odom = np.array([tag_pose_odom[0], tag_pose_odom[1], 0.0, 1.0])
                            pt_map = np.dot(T_map_odom, pt_odom)
                            
                            pts_true.append((self.tag_true_poses[signboard_name][0], self.tag_true_poses[signboard_name][1]))
                            pts_meas.append((pt_map[0], pt_map[1]))
                            
                    # Estimate the transform from ground-truth/world coordinates to current SLAM map coordinates
                    R, T = self.estimate_rigid_transform_2d(pts_true, pts_meas)
                    with self.lock:
                        self.R = R
                        self.T = T
                        
                    if not hasattr(self, 'last_transform_log_time'):
                        self.last_transform_log_time = 0.0
                    now_sec = rospy.Time.now().to_sec()
                    if now_sec - self.last_transform_log_time > 5.0:
                        rospy.loginfo(f"[RIGID TRANSFORM] Match count: {len(pts_true)}, T: {T.tolist()}")
                        self.last_transform_log_time = now_sec
                    

                                        
                    # Draw persistently detected AprilTags
                    for tag_id, tag_pose_odom in self.detected_tags.items():
                        pt_odom = np.array([tag_pose_odom[0], tag_pose_odom[1], 0.0, 1.0])
                        pt_map = np.dot(T_map_odom, pt_odom)
                        
                        t_col = int((pt_map[0] - origin.position.x) / resolution)
                        t_row = int((pt_map[1] - origin.position.y) / resolution)
                        t_row_flipped = height - 1 - t_row
                        
                        if 0 <= t_col < width and 0 <= t_row_flipped < height:
                            # Orange square marker
                            cv2.rectangle(color_map, (t_col - 5, t_row_flipped - 5), (t_col + 5, t_row_flipped + 5), (0, 127, 255), -1)
                            cv2.rectangle(color_map, (t_col - 5, t_row_flipped - 5), (t_col + 5, t_row_flipped + 5), (255, 255, 255), 1)
                            cv2.putText(color_map, f"T{tag_id}", (t_col + 8, t_row_flipped + 4), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                                        
                    # Calculate explored area in square meters
                    explored_cells = np.sum((data == 0) | ((data > 0) & (data <= 100)))
                    self.map_coverage_m2 = explored_cells * (resolution ** 2)
                    
                    map_img = color_map
                    
            if map_img is not None:
                target_h = 350
                target_w = int(map_img.shape[1] * (target_h / map_img.shape[0]))
                if target_w > 650:
                    target_w = 650
                resized_map = cv2.resize(map_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                
                ret, buffer = cv2.imencode('.jpg', resized_map)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                placeholder = np.zeros((350, 600, 3), dtype=np.uint8)
                placeholder[:] = [42, 23, 15]
                cv2.putText(placeholder, "Waiting for SLAM Map...", (180, 180), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (148, 163, 184), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            rate.sleep()

    def generate_local_planner_frames(self):
        """Renders local occupancy grid and motion primitives."""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            canvas = None
            with self.lock:
                grid = self.local_costmap
                candidates = list(self.motion_candidates)
                selected = list(self.selected_motion_pts)
            
            if grid is not None:
                width = grid.info.width
                height = grid.info.height
                res = grid.info.resolution
                origin = grid.info.origin
                
                # Render local costmap data with continuous gradient
                try:
                    # ROS OccupancyGrid data is int8 (-1 unknown, 0-100 occupancy)
                    raw_data = np.array(grid.data, dtype=np.int8).reshape((height, width))
                    
                    # Base view: Dark background for free space
                    viz = np.full((height, width, 3), (30, 20, 10), dtype=np.uint8)
                    
                    # 1. Unknown space (-1) -> very dark
                    viz[raw_data == -1] = [5, 5, 5]
                    
                    # 2. Gradient space (0 < data < 100)
                    # We use a orange/red tint for the repulsion field
                    mask_gradient = (raw_data > 0) & (raw_data < 100)
                    if np.any(mask_gradient):
                        occ_vals = raw_data[mask_gradient].astype(np.float32)
                        viz[mask_gradient, 2] = np.clip(occ_vals * 2.5, 0, 255).astype(np.uint8) # Red
                        viz[mask_gradient, 1] = np.clip(occ_vals * 1.5, 0, 255).astype(np.uint8) # Green (Orange tint)
                    
                    # 3. Solid walls (data >= 100) -> White
                    viz[raw_data >= 100] = [220, 220, 220]
                    
                    viz = cv2.flip(viz, 0) # Flip to match robot orientation (Up is forward)
                except Exception as e:
                    rospy.logerr(f"Local Planner Rendering Error: {e}")
                    viz = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Draw motion primitives (all candidates in dim green)
                for primitive in candidates:
                    for i in range(1, len(primitive)):
                        p1 = primitive[i-1]
                        p2 = primitive[i]
                        # Convert local meters to pixel coordinates (centered on robot at origin)
                        # Local origin in heightmap is usually bottom-left
                        pt1 = (int((p1[0] - origin.position.x) / res), height - 1 - int((p1[1] - origin.position.y) / res))
                        pt2 = (int((p2[0] - origin.position.x) / res), height - 1 - int((p2[1] - origin.position.y) / res))
                        cv2.line(viz, pt1, pt2, (0, 100, 0), 1)
                
                # Draw selected path (thick cyan)
                for i in range(1, len(selected)):
                    p1 = selected[i-1]
                    p2 = selected[i]
                    pt1 = (int((p1[0] - origin.position.x) / res), height - 1 - int((p1[1] - origin.position.y) / res))
                    pt2 = (int((p2[0] - origin.position.x) / res), height - 1 - int((p2[1] - origin.position.y) / res))
                    cv2.line(viz, pt1, pt2, (255, 255, 0), 2)
                
                # Draw robot at frame center (0,0 local)
                r_col = int((0 - origin.position.x) / res)
                r_row = height - 1 - int((0 - origin.position.y) / res)
                cv2.circle(viz, (r_col, r_row), 3, (0, 0, 255), -1)
                
                canvas = viz
            
            if canvas is not None:
                # Resize and Zoom: Original is usually 200x200 (20m @ 0.1m res). 
                # We crop the center (robot at origin) and resize to zoom in by factor 2
                h, w = canvas.shape[:2]
                ch, cw = h // 4, w // 4 # One quarter of the area is factor 2 zoom
                
                # Find robot position in pixels to center the zoom
                r_col = int((0 - origin.position.x) / res)
                r_row = h - 1 - int((0 - origin.position.y) / res)
                
                y1 = max(0, r_row - ch)
                y2 = min(h, r_row + ch)
                x1 = max(0, r_col - cw)
                x2 = min(w, r_col + cw)
                
                canvas_zoomed = canvas[y1:y2, x1:x2]
                disp = cv2.resize(canvas_zoomed, (400, 400), interpolation=cv2.INTER_NEAREST)
                ret, buffer = cv2.imencode('.jpg', disp)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                placeholder = np.zeros((400, 400, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No Local Grid", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            rate.sleep()


# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_color')
def video_color():
    return Response(server.generate_frames('color'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_depth')
def video_depth():
    return Response(server.generate_frames('depth'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_map')
def video_map():
    return Response(server.generate_map_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_local_planner')
def video_local_planner():
    return Response(server.generate_local_planner_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    is_paused = rospy.get_param("/exploration_paused", False)
    explore_state = rospy.get_param("/exploration_state", "IDLE")

    # Update robot's pose from TF to ensure status has the absolute latest pose
    rx, ry, ryaw = server.get_current_robot_pose()
    server.robot_x = rx
    server.robot_y = ry
    server.robot_yaw = ryaw

    # Map raw state to simple status string for UI highlighting: 'play', 'pause', or 'stop'
    status_str = "stop"
    if explore_state in ["EXPLORE", "READ_SIGN", "TURN", "CHECK_SHOP"]:
        status_str = "pause" if is_paused else "play"
    elif explore_state == "ARRIVED":
        status_str = "arrived"
    elif explore_state == "IDLE":
        status_str = "stop"

    has_key = bool(rospy.get_param("/openai_api_key", "").strip()) or \
              bool(os.getenv("OPENAI_API_KEY")) or \
              os.path.exists("/home/linusv/project_5/HW4/ChatGPT_API_KEY.txt")

    status = {
        "x": round(server.robot_x, 2) if hasattr(server, 'robot_x') else None,
        "y": round(server.robot_y, 2) if hasattr(server, 'robot_y') else None,
        "yaw": round(server.robot_yaw, 2) if hasattr(server, 'robot_yaw') else None,
        "explored_area": round(server.map_coverage_m2, 1) if hasattr(server, 'map_coverage_m2') else 0.0,
        "shops_detected": len(server.shop_categories) if hasattr(server, 'shop_categories') else 0,
        "tags_detected": len(server.detected_tags) if hasattr(server, 'detected_tags') else 0,
        "exploration_status": status_str,
        "exploration_state": explore_state,
        "has_api_key": has_key,
        "searching_tag": server.searching_tag if hasattr(server, 'searching_tag') else False,
        "navigating_to_tag": server.navigating_to_tag if hasattr(server, 'navigating_to_tag') else False
    }
    return jsonify(status)

@app.route('/api/logs')
def api_logs():
    with server.lock:
        logs = list(server.planner_logs)
        server.planner_logs = []
    return jsonify({"logs": logs})

@app.route('/api/set_api_key', methods=['POST'])
def set_api_key():
    data = request.json or {}
    key = data.get('api_key', '').strip()
    if key:
        rospy.set_param("/openai_api_key", key)
        rospy.loginfo("OpenAI API key dynamically updated by the user interface.")
        return jsonify({"status": "success", "message": "API key successfully set for this session!"})
    else:
        return jsonify({"status": "error", "message": "API key cannot be empty!"}), 400

@app.route('/api/cmd_vel', methods=['POST'])
def cmd_vel():
    data = request.json
    linear = float(data.get('linear', 0.0))
    angular = float(data.get('angular', 0.0))
    
    twist = Twist()
    twist.linear.x = linear
    twist.angular.z = angular
    
    server.cmd_vel_pub.publish(twist)
    return jsonify({"status": "success"})

@app.route('/api/set_ai_mode', methods=['POST'])
def set_ai_mode():
    data = request.json
    mode = data.get('mode', 'local') # 'local' or 'remote'
    use_local = (mode == 'local')
    rospy.set_param("/use_local_ai", use_local)
    rospy.loginfo(f"Set /use_local_ai to {use_local}")
    return jsonify({"status": "success", "mode": mode})

@app.route('/api/detect', methods=['POST'])
def api_detect():
    rospy.loginfo("Detect Mode triggered (Backend placeholder).")
    return jsonify({"status": "success", "message": "Detection triggered (Backend placeholder)."})

@app.route('/api/find_tag', methods=['POST'])
def api_find_tag():
    if server.searching_tag:
        server.searching_tag = False
        server.stop_robot()
        return jsonify({"status": "cancelled", "message": "Tag search cancelled."})
    else:
        server.searching_tag = True
        server.search_thread = threading.Thread(target=server.find_tag_thread)
        server.search_thread.daemon = True
        server.search_thread.start()
        return jsonify({"status": "started", "message": "Tag search started."})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    server.stop_robot()
    return jsonify({"status": "success", "message": "Emergency Stop Triggered!"})

@app.route('/api/navigate_to_tag', methods=['POST'])
def api_navigate_to_tag():
    data = request.json or {}
    tag_name = data.get('tag_name', '').strip()
    if not tag_name or tag_name not in server.tag_true_poses:
        return jsonify({"status": "error", "message": f"Invalid tag name: {tag_name}"}), 400
        
    success = server.start_navigation_to_tag(tag_name)
    if success:
        return jsonify({"status": "ok", "message": f"Navigating to landmark {tag_name}..."})
    else:
        return jsonify({"status": "error", "message": "Failed to start navigation."}), 500

@app.route('/api/delivery/send', methods=['POST'])
def delivery_send():
    data = request.json
    message = data.get('message', '').strip()
    if not message:
        return jsonify({"reply": "I did not receive a message. Please say something!"})
        
    target = ""
    reply = ""
    
    # Call the /llm_query service of the stateless client to parse the user's intent!
    try:
        rospy.wait_for_service("llm_query", timeout=2.0)
        llm_query_srv = rospy.ServiceProxy("llm_query", LLMQuery)
        
        prompt = (
            f"The user is talking to a delivery robot. They said: '{message}'.\n"
            f"We want to match their request to one of our mapped storefront categories or store names: "
            f"BLUE CAFE, BLUE STORE, GREEN STORE, ORANGE CAFE, RED BURGER, RED PHARMACY, WHITE CAFE, YELLOW BURGER.\n"
            f"Business Category categories are Cafe, Convenience store, Fast-food restaurant, Pharmacy.\n"
            f"Determine what they want, and return ONLY a raw JSON object with two keys:\n"
            f"1. 'target' (string, the exact target category or store name, e.g. 'Café' or 'BLUE CAFE' or 'RED BURGER')\n"
            f"2. 'reply' (string, a cute, polite conversational response to display to the user, explaining where you will go to get this)."
        )
        
        llm_req = LLMQueryRequest()
        llm_req.prompt = prompt
        llm_res = llm_query_srv(llm_req)
        
        # Parse reply safely
        cleaned_resp = llm_res.response.strip()
        if cleaned_resp.startswith("```json"):
            cleaned_resp = cleaned_resp[7:]
        if cleaned_resp.endswith("```"):
            cleaned_resp = cleaned_resp[:-3]
        cleaned_resp = cleaned_resp.strip()
        
        parsed = json.loads(cleaned_resp)
        target = parsed.get("target", "")
        reply = parsed.get("reply", "Understood! Moving to target.")
        
    except Exception as e:
        rospy.logwarn(f"LLM Query failed or timed out: {e}")
        # Standard local backup parser for simple keywords
        reply = "I'm in offline Local Mode. Let me parse that... "
        m = message.lower()
        if "burger" in m or "food" in m or "restaurant" in m:
            target = "Fast-food restaurant"
            reply += "Ah! You want a burger. I will navigate to the Fast-food restaurant!"
        elif "coffee" in m or "cafe" in m or "drink" in m:
            target = "Café"
            reply += "Ah! You want coffee. I will navigate to the Café!"
        elif "med" in m or "pharm" in m or "pill" in m or "sick" in m:
            target = "Pharmacy"
            reply += "Ah! You need a pharmacy. I will navigate to the Pharmacy!"
        elif "store" in m or "shop" in m or "item" in m or "convenience" in m:
            target = "Convenience store"
            reply += "Ah! You need the Convenience store. I will navigate there!"
        else:
            reply = "I'm not sure which storefront matches that request. Try asking for coffee, a burger, medicine, or the convenience store!"
            
    # Direct navigation resolution
    resolved_tag = None
    if target:
        target_upper = target.upper().strip()
        target_key = target_upper.replace(" ", "_")
        if target_key in server.tag_true_poses:
            resolved_tag = target_key
        else:
            # Check if it matches a category value in server.shop_categories (mapped store category)
            for store_name, category in server.shop_categories.items():
                if category.upper().strip() == target_upper or target_upper in category.upper().strip():
                    store_key = store_name.upper().replace(" ", "_")
                    if store_key in server.tag_true_poses:
                        resolved_tag = store_key
                        break
                        
            # If still not found, try partial match in store names
            if not resolved_tag:
                for k in server.tag_true_poses.keys():
                    if k in target_key or target_key in k:
                        resolved_tag = k
                        break
                        
    if resolved_tag:
        rospy.loginfo(f"[Delivery API] Starting direct navigation to resolved landmark: {resolved_tag}")
        server.start_navigation_to_tag(resolved_tag)
        
    return jsonify({"reply": reply, "target": target})

@app.route('/api/semantic_map')
def api_semantic_map():
    # Return server.shop_categories for UI legends and mapping
    return jsonify(server.shop_categories)

@app.route('/api/reset', methods=['POST'])
def api_reset():
    # 1. Reset local lists/dictionaries in the web server
    server.shop_categories = {}
    server.detected_tags = {}
    server.grid_map = None
    if hasattr(server, 'trail'):
        server.trail = []
    if hasattr(server, 'last_classification_time'):
        server.last_classification_time = {}
        
    rospy.loginfo("Web server local state reset triggered.")
    
    # 2. Reset Gazebo Simulation (models, controllers, and odometry)
    try:
        rospy.wait_for_service('/gazebo/reset_simulation', timeout=1.0)
        reset_sim_srv = rospy.ServiceProxy('/gazebo/reset_simulation', EmptySrv)
        reset_sim_srv()
        rospy.loginfo("Gazebo simulation successfully reset.")
    except Exception as e:
        rospy.logwarn(f"Failed to reset Gazebo simulation: {e}")
        
    # 3. Reset RTAB-Map SLAM completely (reset database + trigger fresh map session)
    try:
        rospy.wait_for_service('/rtabmap/reset', timeout=1.0)
        reset_rtab_srv = rospy.ServiceProxy('/rtabmap/reset', EmptySrv)
        reset_rtab_srv()
        rospy.loginfo("RTAB-Map database successfully reset.")
    except Exception as e:
        rospy.logwarn(f"Failed to reset RTAB-Map SLAM: {e}")
        
    try:
        rospy.wait_for_service('/rtabmap/trigger_new_map', timeout=1.0)
        trigger_new_srv = rospy.ServiceProxy('/rtabmap/trigger_new_map', EmptySrv)
        trigger_new_srv()
        rospy.loginfo("RTAB-Map fresh map session triggered.")
    except Exception as e:
        rospy.logwarn(f"Failed to trigger fresh RTAB-Map session: {e}")
        
    return jsonify({"status": "success", "message": "Robot state, classifications, SLAM map database, and Gazebo simulation successfully reset."})

if __name__ == '__main__':
    global server
    server = RobotWebServer()
    # Run flask in a separate thread to allow ROS to spin (or vice-versa)
    # Using threaded=True allows Flask to handle multiple connections
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)).start()
    rospy.spin()
