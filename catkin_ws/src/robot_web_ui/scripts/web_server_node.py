#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import threading
import yaml
import os
import re
import math
import time
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

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

app = Flask(__name__, static_folder='../static', template_folder='../templates')

class RobotWebServer:
    def __init__(self):
        rospy.init_node('robot_web_ui_server', anonymous=True)
        self.bridge = CvBridge()

        # Try to import YOLO if it failed at top-level (e.g. if installed after node start)
        global YOLO
        if YOLO is None:
            try:
                from ultralytics import YOLO as Y
                YOLO = Y
            except ImportError:
                pass

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
        self.mapped_shops = []  # Replaced self.stores with dynamic mapped_shops
        self.tasks_fulfilled = 0
        self.overshoot_m = 0.0
        self.detected_tags = {}
        self.shop_categories = {}
        self.local_costmap = None
        self.motion_candidates = []
        self.selected_motion_pts = []
        self.trunc_target = None
        self.planner_logs = []
        self.R = np.eye(2)
        self.T = np.array([0.0, -3.125])
        self.T_map_odom = np.eye(4)
        self.searching_tag = False
        self.search_thread = None
        self.navigating_to_tag = False
        
        # Set default parameter: Local AI mode active on startup
        rospy.set_param("/use_local_ai", True)
        rospy.set_param("/local_planner_type", "teb")
        
        self.lock = threading.Lock()
        self.tf_listener = tf.TransformListener()

        # Load AprilTag bundles configuration & Store coordinates
        self.yaml_path = "/home/linusv/project_5/HW4/tags.yaml"
        self.bundles = self.load_bundles(self.yaml_path)
        self.load_stores_from_txt()
        self.load_tag_true_poses()
        self.sign_database = self.load_sign_database()
        
        # YOLO Visualization Cache
        self.last_yolo_viz_results = []
        self.last_yolo_viz_time = 0

        # YOLO initialization
        model_path = "/home/linusv/project_5/catkin_ws/src/robot_web_ui/yolo_models/navigation.pt"
        self.yolo_model = None
        if YOLO and os.path.exists(model_path):
            self.yolo_model = YOLO(model_path)
            rospy.loginfo(f"YOLO model loaded from {model_path}")
        else:
            rospy.logwarn(f"YOLO model not found at {model_path} or ultralytics not installed.")

        # Complex Action State
        self.active_delivery_task = None
        self.task_thread = None
        self.navigating_to_pose_active = False
        
        self.delivery_chat_history = [
            {
                "sender": "bot",
                "text": "Hello! I am your AI delivery robot. Tell me what product or storefront you want me to find, and I will search the map or interpret signboards to bring it to you!"
            }
        ]
        self.active_todo_list = {
            "status": "idle",
            "stores": []
        }
        
        # Visualization Toggles
        self.viz_apriltag = True
        self.viz_yolo = True

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
        rospy.Subscriber('/car/trunc_target', PoseStamped, self.target_cb)
        rospy.Subscriber('/rosout', Log, self.rosout_cb)
        
        if AprilTagDetectionArray is not None:
            rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.tags_cb)

        rospy.loginfo("Web Server Node initialized.")

    def append_bot_chat_message(self, text):
        with self.lock:
            self.delivery_chat_history.append({"sender": "bot", "text": text})
            if len(self.delivery_chat_history) > 100:
                self.delivery_chat_history.pop(0)

    def append_user_chat_message(self, text):
        with self.lock:
            self.delivery_chat_history.append({"sender": "user", "text": text})
            if len(self.delivery_chat_history) > 100:
                self.delivery_chat_history.pop(0)

    def stop_robot(self):
        """Immediately stops all autonomous navigation and the robot movement."""
        with self.lock:
            self.searching_tag = False
            self.navigating_to_tag = False
        
        self.navigating_to_pose_active = False
        self.active_delivery_task = None
        
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
            # Depth images are typically 16UC1 or 32FC1.
            cv_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            cv_image = np.nan_to_num(cv_image, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Store raw depth for visual servoing
            with self.lock:
                self.depth_raw = cv_image
            
            # Normalize to 0-255 for visualization
            max_val = np.max(cv_image)
            if max_val > 0:
                vis_image = (cv_image / max_val * 255.0).astype(np.uint8)
            else:
                vis_image = cv_image.astype(np.uint8)
                
            # Apply colormap for better visualization
            cv_image_color = cv2.applyColorMap(vis_image, cv2.COLORMAP_JET)
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
                        self.detected_tags[tag_id] = (trans, rot)
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
                        trans_ot = T_odom_tag[:3, 3]
                        rot_ot = tf.transformations.quaternion_from_matrix(T_odom_tag)
                        
                        self.detected_tags[tag_id] = (trans_ot, rot_ot)
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

    def target_cb(self, msg):
        with self.lock:
            self.trunc_target = (msg.pose.position.x, msg.pose.position.y)

    def rosout_cb(self, msg):
        """Filters logs to only keep those from the motion planner and related safety services."""
        # Relevant node name substrings
        relevant_keys = ["control_space", "heightmap", "agent", "teb", "custom_vector", "robot_web_ui", "web_server"]
        name_lower = msg.name.lower()
        if any(key in name_lower for key in relevant_keys):
            with self.lock:
                level_str = "INFO"
                if msg.level == 4: level_str = "WARN"
                elif msg.level == 8: level_str = "ERROR"
                elif msg.level == 16: level_str = "FATAL"
                elif msg.level == 1: level_str = "DEBUG"
                
                node_short = msg.name.split('/')[-1]
                # Strip anonymous digits suffix
                node_short = re.sub(r'_\d+_\d+$', '', node_short)
                log_entry = f"[{level_str}] [{node_short}] {msg.msg}"
                
                self.planner_logs.append(log_entry)
                # Keep only last 150 logs to avoid memory bloat
                if len(self.planner_logs) > 150:
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

    def is_pose_reachable(self, x, y, radius=0.3, threshold=70):
        """Checks the local costmap to see if a circle of given radius at (x, y) is blocked."""
        with self.lock:
            grid = self.local_costmap
            
        if grid is None:
            return True
            
        try:
            # Transform Map Point to Costmap Frame
            ps = PointStamped()
            ps.header.frame_id = "map"
            ps.header.stamp = rospy.Time(0)
            ps.point.x = x
            ps.point.y = y
            
            ps_local = self.tf_listener.transformPoint(grid.header.frame_id, ps)
            lx, ly = ps_local.point.x, ps_local.point.y
            
            info = grid.info
            origin = info.origin.position
            res = info.resolution
            
            # Check a small patch around the point to account for robot footprint
            gx_center = int((lx - origin.x) / res)
            gy_center = int((ly - origin.y) / res)
            
            pixel_radius = int(radius / res)
            for dy in range(-pixel_radius, pixel_radius + 1):
                for dx in range(-pixel_radius, pixel_radius + 1):
                    if dx*dx + dy*dy > pixel_radius*pixel_radius: continue
                    
                    gx = gx_center + dx
                    gy = gy_center + dy
                    
                    if 0 <= gx < info.width and 0 <= gy < info.height:
                        cost = grid.data[gy * info.width + gx]
                        if cost > threshold: # Custom occupied or near wall threshold
                            return False
                    else:
                        return False # Out of bounds is unreachable
            return True
        except Exception as e:
            rospy.logerr(f"Costmap Check Error: {e}")
            return False

    def get_distance_to_wall_ahead(self, max_dist=6.0):
        """Casts a ray forward in the local costmap to find the exact physical wall distance."""
        rx, ry, ryaw = self.get_current_robot_pose()
        step_size = 0.05
        current_dist = 0.3  # start 30cm in front (beyond robot footprint)
        
        while current_dist <= max_dist:
            test_x = rx + current_dist * math.cos(ryaw)
            test_y = ry + current_dist * math.sin(ryaw)
            
            # Use radius 0.0 to check the single point along the ray
            if not self.is_pose_reachable(test_x, test_y, radius=0.0): 
                return current_dist
            current_dist += step_size
            
        return None

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

    def start_navigation_to_tag(self, tag_name, is_part_of_task=False):
        # Cancel any active search
        self.searching_tag = False
        self.is_part_of_task = is_part_of_task
        
        # Only clear delivery task if this navigation call was NOT part of a larger task
        if not is_part_of_task:
            self.active_delivery_task = None
        
        # Stop any existing navigation thread
        self.navigating_to_tag = False
        if hasattr(self, 'nav_thread') and self.nav_thread.is_alive():
            self.nav_thread.join(timeout=1.0)
            
        self.navigating_to_tag = True
        self.nav_thread = threading.Thread(target=self.nav_to_tag_thread, args=(tag_name,))
        self.nav_thread.daemon = True
        self.nav_thread.start()
        return True

    def navigate_to_pose(self, x, y, yaw, ignore_yaw=False, dist_tol=0.2, cost_thresh=70):
        """Standard method to navigate to a specific map pose using the global planner."""
        # 0. Safety Pre-check
        if not self.is_pose_reachable(x, y, threshold=cost_thresh):
            rospy.logerr(f"[Navigation] Goal ({x:.2f}, {y:.2f}) is blocked in local costmap. Aborting.")
            self.navigating_to_pose_active = False
            return False

        self.navigating_to_pose_active = True
        
        # Enable C++ planner
        rospy.set_param("/exploration_state", "EXPLORE")
        rospy.set_param("/exploration_paused", False)
        
        rate = rospy.Rate(10)
        start_time = rospy.Time.now()
        arrived = False
        
        while not rospy.is_shutdown() and self.navigating_to_pose_active and self.active_delivery_task:
            rx, ry, ryaw = self.get_current_robot_pose()
            dist = math.hypot(rx - x, ry - y)
            
            if dist < dist_tol:
                if ignore_yaw:
                    arrived = True
                    break
                # Check orientation near end
                angle_diff = (yaw - ryaw + math.pi) % (2 * math.pi) - math.pi
                if abs(angle_diff) < 0.20:
                    arrived = True
                    break
            
            # Publish single-segment path to goal
            path = Path()
            path.header.frame_id = "map"
            path.header.stamp = rospy.Time.now()
            
            # Start
            p0 = PoseStamped()
            p0.header = path.header
            p0.pose.position.x = rx
            p0.pose.position.y = ry
            angle_to_goal = math.atan2(y - ry, x - rx)
            q0 = tf.transformations.quaternion_from_euler(0, 0, angle_to_goal)
            p0.pose.orientation = Quaternion(*q0)
            path.poses.append(p0)
            
            # End
            p1 = PoseStamped()
            p1.header = path.header
            p1.pose.position.x = x
            p1.pose.position.y = y
            q1 = tf.transformations.quaternion_from_euler(0, 0, yaw)
            p1.pose.orientation = Quaternion(*q1)
            path.poses.append(p1)
            
            self.path_pub.publish(path)
            
            if (rospy.Time.now() - start_time).to_sec() > 60.0:
                rospy.logwarn("[Navigation] Timeout reached.")
                break
            rate.sleep()
            
        self.navigating_to_pose_active = False
        self.stop_search(keep_delivery=True) # Clears path but keeps workflow target
        return arrived

    def check_for_shop(self, target_category):
        """Single-frame check for the target shop. Returns detections if found."""
        target_norm = self.normalize_category(target_category)
        with self.lock:
            img = self.color_image.copy() if self.color_image is not None else None
            depth_raw = getattr(self, 'depth_raw', None)
            
        if img is not None:
            results = self.yolo_model.predict(img, conf=0.8, verbose=False)
            for res in results:
                for box in res.boxes:
                    label = res.names[int(box.cls[0])].upper()
                    clean_label = label.replace("STORE_", "").replace("_", " ")
                    label_norm = self.normalize_category(label)
                    
                    is_cafe = "CAF" in target_norm and ("CAF" in label or "CAF" in clean_label)
                    is_hamb = ("HAMB" in target_norm or "BURG" in target_norm) and ("HAMB" in label or "BURG" in label)
                    is_pharm = "PHARM" in target_norm and ("PHARM" in label or "PHARM" in clean_label)
                    is_store = "STORE" in target_norm and "STORE" in label and not ("CAF" in label or "PHARM" in label or "BURG" in label or "HAMB" in label)
                    
                    match = (target_norm in ["ANY", "ALL"]) or \
                            (target_norm == label_norm or target_norm == label or target_norm == clean_label) or \
                            is_cafe or is_hamb or is_pharm or is_store

                    if match:
                        shop_img_x = (box.xyxy[0][0] + box.xyxy[0][2]) / 2.0
                        y1, x1, y2, x2 = int(box.xyxy[0][1]), int(box.xyxy[0][0]), int(box.xyxy[0][3]), int(box.xyxy[0][2])
                        
                        depth_val = 0.0
                        if depth_raw is not None:
                            h, w = depth_raw.shape
                            roi = depth_raw[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                            valid_depths = roi[roi > 0.1]
                            if valid_depths.size > 0:
                                # Use 15th percentile to ensure we are looking at the 'front-most' edge
                                dist_to_shop = np.percentile(valid_depths, 15)
                                if dist_to_shop > 50.0: dist_to_shop /= 1000.0
                                depth_val = dist_to_shop
                        
                        if depth_val == 0.0: depth_val = 1.8 
                        if depth_val > 6.0: continue

                        return {
                            'label': label,
                            'depth': depth_val,
                            'img_x': shop_img_x,
                            'img_w': float(img.shape[1]),
                            'pose': self.get_current_robot_pose()
                        }
        return None

    def center_on_shop(self, target_category, timeout=12.0):
        """Turn to face the store perfectly centered in the camera."""
        rospy.loginfo("[Visual Servoing] Centering on the shop for perfect alignment...")
        rate = rospy.Rate(15)
        start_time = rospy.Time.now()
        
        while not rospy.is_shutdown() and self.active_delivery_task:
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logwarn("[Visual Servoing] Timeout while centering.")
                return False
                
            det = self.check_for_shop(target_category)
            if not det:
                self.cmd_vel_pub.publish(Twist())
                rate.sleep()
                continue
                
            img_w = det['img_w']
            x_center = det['img_x']
            error_norm = (x_center - img_w/2.0) / (img_w/2.0) # -1 to 1
            
            if abs(error_norm) < 0.025: # Extremely tight centering (~1.2% offset allowed)
                self.cmd_vel_pub.publish(Twist())
                rospy.sleep(0.5) # Wait for complete stop
                return True
                
            cmd = Twist()
            cmd.angular.z = -1.0 * error_norm # smooth PI
            cmd.angular.z = np.clip(cmd.angular.z, -0.4, 0.4) # limit speed for precision
            self.cmd_vel_pub.publish(cmd)
            rate.sleep()
            
        self.cmd_vel_pub.publish(Twist())
        return False

    def approach_shop_via_waypoint(self, target_category, corridor_yaw=None):
        """RADICALLY NEW STRATEGY: Discover -> Visual Center -> Raycast -> TEB -> Align.
        Supports retry with forward nudge on focus failure (centering timeout) or local costmap blockage.
        """
        if not self.yolo_model:
            return False

        rospy.loginfo(f"[Radical Overhaul] Commencing precision approach for {target_category}")
        _, _, start_yaw = self.get_current_robot_pose()
        ref_yaw = corridor_yaw if corridor_yaw is not None else start_yaw
        
        max_attempts = 4
        for attempt in range(max_attempts):
            if not self.active_delivery_task:
                break
                
            rospy.loginfo(f"[Approach Workflow] Sweep-and-Nudge Attempt {attempt + 1}/{max_attempts}")
            
            # PHASE 1: DISCOVER THE STORE
            found = False
            # Check pre-sweep
            det = self.check_for_shop(target_category)
            if det:
                rospy.loginfo("[Approach Workflow] Found shop in pre-sweep!")
                found = True
            else:
                # Do continuous sweep left-to-right
                rospy.loginfo("[Approach Workflow] Snapping to left boundary to begin continuous sweep...")
                left_yaw = (ref_yaw + math.radians(60) + math.pi) % (2*math.pi) - math.pi
                self.rotate_to_yaw(left_yaw, p_gain=2.5, speed_limit=1.5, threshold=0.03)
                self.cmd_vel_pub.publish(Twist())
                rospy.sleep(0.2)
                
                if self.check_for_shop(target_category):
                    rospy.loginfo("[Approach Workflow] Discovered shop at left boundary.")
                    found = True
                else:
                    rospy.loginfo("[Approach Workflow] Commencing smooth continuous visual sweep...")
                    rate = rospy.Rate(15)
                    cmd = Twist()
                    cmd.angular.z = -0.5 # Smooth rotation rightwards
                    
                    sweep_start = rospy.Time.now()
                    while not rospy.is_shutdown() and self.active_delivery_task:
                        self.cmd_vel_pub.publish(cmd)
                        if self.check_for_shop(target_category):
                            rospy.loginfo("[Approach Workflow] Discovered shop dynamically on-the-fly!")
                            self.cmd_vel_pub.publish(Twist()) # Halt instantly
                            found = True
                            break
                        if (rospy.Time.now() - sweep_start).to_sec() > 4.5:
                            break
                        rate.sleep()
                    self.cmd_vel_pub.publish(Twist())
            
            # If we didn't find the shop during the sweep at all
            if not found:
                rospy.logwarn("[Approach Workflow] No shop found in sweep. Resetting and translating forward...")
                self.nudge_forward_and_recover(ref_yaw)
                continue
                
            # PHASE 2: VISUAL CENTERING (Face the store strictly based on bounding box)
            rospy.loginfo("[Approach Workflow] Attempting to center on shop...")
            centered = self.center_on_shop(target_category, timeout=8.0)
            if not centered:
                rospy.logwarn("[Approach Workflow] Centering failed (lost focus). Assuming not detected, nudging forward to retry sweep...")
                self.nudge_forward_and_recover(ref_yaw)
                continue
                
            # PHASE 3: CALCULATE EXACT STORE LOCATION
            rx, ry, ryaw = self.get_current_robot_pose()
            wall_dist = self.get_distance_to_wall_ahead(max_dist=6.0)
            
            if wall_dist is not None:
                rospy.loginfo(f"[Approach Workflow] LiDAR/Costmap hit physical wall at {wall_dist:.2f}m straight ahead.")
            else:
                det = self.check_for_shop(target_category)
                wall_dist = det['depth'] + 0.15 if det else 2.0
                rospy.logwarn(f"[Approach Workflow] Raycast missed. Using Camera Depth: {wall_dist:.2f}m")
                
            # The physical center of the storefront on the map
            shop_x = rx + wall_dist * math.cos(ryaw)
            shop_y = ry + wall_dist * math.sin(ryaw)
            
            # PHASE 4: CALCULATE 90-DEGREE WAYPOINT
            n1 = (ref_yaw + math.pi/2.0 + math.pi) % (2*math.pi) - math.pi
            n2 = (ref_yaw - math.pi/2.0 + math.pi) % (2*math.pi) - math.pi
            
            vec_to_robot = np.array([rx - shop_x, ry - shop_y])
            vec_n1 = np.array([math.cos(n1), math.sin(n1)])
            vec_n2 = np.array([math.cos(n2), math.sin(n2)])
            
            outward_normal = n1 if np.dot(vec_to_robot, vec_n1) > np.dot(vec_to_robot, vec_n2) else n2
            
            target_dist = 0.60
            target_x = shop_x + target_dist * math.cos(outward_normal)
            target_y = shop_y + target_dist * math.sin(outward_normal)
            
            # Apply overshoot along the corridor direction
            overshoot_m = getattr(self, 'overshoot_m', 0.0)
            target_x += overshoot_m * math.cos(ref_yaw)
            target_y += overshoot_m * math.sin(ref_yaw)
            
            target_yaw = (outward_normal + math.pi)
            target_yaw = (target_yaw + math.pi) % (2*math.pi) - math.pi
            
            rospy.loginfo(f"[Approach Workflow] Target Pose: ({target_x:.2f}, {target_y:.2f}) facing {math.degrees(target_yaw):.1f} deg")
            
            # PHASE 5: SAFETY PUSH (GUARANTEE NO CRASH)
            for push in range(10):
                if self.is_pose_reachable(target_x, target_y, radius=0.20, threshold=98):
                    break
                rospy.logwarn(f"[Approach Workflow] Perpendicular goal blocked. Pushing outward by 10cm...")
                target_dist += 0.10
                target_x = shop_x + target_dist * math.cos(outward_normal)
                target_y = shop_y + target_dist * math.sin(outward_normal)
            else:
                rospy.logerr("[Approach Workflow] Completely blocked up to 1.5m. Cannot find safe position. Nudging forward to retry sweep...")
                self.nudge_forward_and_recover(ref_yaw)
                continue
                
            # PHASE 6: EXECUTE (Slow TEB)
            v_max = rospy.get_param("/navigation/max_vel_x", 0.3)
            rospy.set_param("/navigation/max_vel_x", 0.08)
            rospy.set_param("/move_base/TebLocalPlannerROS/max_vel_x", 0.08)
            rospy.set_param("/move_base/TebLocalPlannerROS/max_vel_x_backwards", 0.08)
            rospy.set_param("/move_base/TebLocalPlannerROS/max_vel_theta", 0.3)
            
            rospy.loginfo("[Approach Workflow] Engaging TEB planner to exact spatial dot...")
            nav_success = self.navigate_to_pose(target_x, target_y, target_yaw, ignore_yaw=True, dist_tol=0.10, cost_thresh=98)
            
            # Restore speeds
            rospy.set_param("/navigation/max_vel_x", v_max)
            rospy.set_param("/move_base/TebLocalPlannerROS/max_vel_x", v_max)
            rospy.set_param("/move_base/TebLocalPlannerROS/max_vel_x_backwards", 0.15)
            rospy.set_param("/move_base/TebLocalPlannerROS/max_vel_theta", 0.8)
            
            if not nav_success:
                rospy.logerr("[Approach Workflow] Navigation to shopfront failed or was blocked. Nudging forward to retry sweep...")
                self.nudge_forward_and_recover(ref_yaw)
                continue
                
            # FINAL SQUARING UP
            rospy.loginfo("[Approach Workflow] Executing final exact rotation snap to 90 degrees...")
            self.rotate_to_yaw(target_yaw, threshold=0.015)
            return True
            
        rospy.logerr(f"[Approach Workflow] Failed to approach shop {target_category} after {max_attempts} attempts.")
        return False

    def nudge_forward_and_recover(self, corridor_yaw):
        """Helper to recover from lost/failed sweep: rotates back to corridor, rotates to clear costmap, then translates forward."""
        rospy.loginfo("[Nudge Recovery] Rotating back to corridor direction...")
        self.rotate_to_yaw(corridor_yaw, threshold=0.05)
        self.stop_search(keep_delivery=True)
        
        # In-place rotation recovery to clear costmaps
        self.recover_robot()
        
        # Nudge forward
        rx, ry, ryaw = self.get_current_robot_pose()
        # Try a 0.5m nudge first, check reachability
        nudge_dist = 0.50
        
        # Try steps: 0.5m, 0.35m, 0.20m
        for step in [0.50, 0.35, 0.20]:
            fx = rx + step * math.cos(corridor_yaw)
            fy = ry + step * math.sin(corridor_yaw)
            if self.is_pose_reachable(fx, fy, threshold=98):
                rospy.loginfo(f"[Nudge Recovery] Nudging forward by {step:.2f}m...")
                self.navigate_to_pose(fx, fy, corridor_yaw, cost_thresh=98)
                break
        else:
            # If all are blocked in costmap pre-check, execute a small blind nudge to clear the costmap inflation
            rospy.logwarn("[Nudge Recovery] Nudge path blocked in costmap. Executing safe blind nudge...")
            cmd = Twist()
            cmd.linear.x = 0.1
            self.cmd_vel_pub.publish(cmd)
            rospy.sleep(2.0)
            self.cmd_vel_pub.publish(Twist())



    def start_delivery_search(self, category):
        """Starts the complex workflow: Find sign -> Go to sign -> Align -> Crab search."""
        self.stop_robot() # Reset all states
        
        self.active_delivery_task = category.upper()
        self.task_thread = threading.Thread(target=self.delivery_search_workflow, args=(self.active_delivery_task,))
        self.task_thread.daemon = True
        self.task_thread.start()
        return True

    def delivery_search_workflow(self, target_category, is_part_of_task=False):
        success = False
        try:
            rospy.loginfo(f"[Delivery Task] Starting workflow for: {target_category}")
            
            # 1. FIND CLOSEST CROSSROAD SIGN (among known signs that point to this category)
            rx, ry, _ = self.get_current_robot_pose()
            best_tag = None
            min_dist = float('inf')
            target_dir = 0.0
            
            # Use local copies to avoid lock contention during heavy logic
            with self.lock:
                current_R = self.R.copy()
                current_T = self.T.copy()
                
            target_norm = self.normalize_category(target_category)
            for entry in self.sign_database:
                entry_norm = self.normalize_category(entry['category'])
                if target_norm == entry_norm:
                    tag_name = entry['tag']
                    if tag_name in self.tag_true_poses:
                        pose_info = self.tag_true_poses[tag_name]
                        # Project to map using current estimation
                        tag_pt = np.array([pose_info[0], pose_info[1]])
                        tag_map = np.dot(current_R, tag_pt) + current_T
                        dist = math.hypot(rx - tag_map[0], ry - tag_map[1])
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_tag = tag_name
                            target_dir = entry['direction']
                            
            if not best_tag:
                rospy.logwarn(f"[Delivery Task] No sign found for {target_category} in database.")
                if not is_part_of_task:
                    self.active_delivery_task = None
                return False

            # 2. GO TO THE APRITAG (using motion planner)
            rospy.loginfo(f"[Delivery Task] Step 1: Navigating to {best_tag} (dist: {min_dist:.2f}m)")
            self.start_navigation_to_tag(best_tag, is_part_of_task=True)
            
            # Wait until arrived (navigation logic sets navigating_to_tag to False on arrival)
            # We add a small delay to ensure the thread has started
            rospy.sleep(0.2)
            while not rospy.is_shutdown() and self.navigating_to_tag and self.active_delivery_task:
                rospy.sleep(0.1)
                
            if not self.active_delivery_task: 
                rospy.loginfo("[Delivery Task] Task cancelled during navigation.")
                return False

            # 3. ALIGN TO THE DIRECTION FROM THE ROAD SIGN
            rospy.loginfo(f"[Delivery Task] Step 2: Aligning to sign direction ({target_dir} deg)")
            if best_tag not in self.tag_true_poses:
                rospy.logerr(f"[Delivery Task] Critical error: {best_tag} missing from tag_true_poses")
                if not is_part_of_task:
                    self.active_delivery_task = None
                return False
                
            pose_info = self.tag_true_poses[best_tag]
            tag_true_yaw_deg = pose_info[2]
            
            # Map yaw = true_yaw + rotation from R
            rot_angle = math.atan2(current_R[1, 0], current_R[0, 0])
            target_yaw = math.radians(tag_true_yaw_deg) + rot_angle + math.radians(target_dir)
            
            rospy.loginfo(f"[Delivery Task] Rotating to target yaw: {math.degrees(target_yaw):.1f} deg")
            # Fast alignment with corridor before sweep
            self.rotate_to_yaw(target_yaw, p_gain=2.0, speed_limit=1.5)
            
            if not self.active_delivery_task:
                return False
                
            # 4. DISCOVER SHOP & APPROACH USING WAYPOINT
            rospy.loginfo(f"[Delivery Task] Step 3: Discovering and approaching {target_category}...")
            # Use the new waypoint-based approach for reliability, passing the corridor alignment
            success = self.approach_shop_via_waypoint(target_category, corridor_yaw=target_yaw)
            
            if success:
                rospy.loginfo(f"[Delivery Task] SUCCESS: Arrived at {target_category}!")
            else:
                rospy.logwarn(f"[Delivery Task] Failed to final-approach {target_category} storefront.")
        except Exception as e:
            rospy.logerr(f"[Delivery Task] Unexpected error in workflow: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            success = False
            
        if not is_part_of_task:
            self.active_delivery_task = None
        return success

    def start_shopping_list_workflow(self, tasks):
        """Starts sequential traversal of multiple store destinations (shopping list)."""
        self.stop_robot() # Reset all states
        
        self.active_delivery_task = "SHOPPING_LIST"
        self.task_thread = threading.Thread(target=self.shopping_list_worker, args=(tasks,))
        self.task_thread.daemon = True
        self.task_thread.start()
        return True

    def shopping_list_worker(self, tasks):
        rospy.loginfo(f"[Shopping List] Commencing execution of {len(tasks)} tasks sequentially.")
        overshoot_m = getattr(self, 'overshoot_m', 0.0)
        
        for task in tasks:
            if not self.active_delivery_task:
                rospy.logwarn("[Shopping List] Task execution halted (cancelled).")
                break
                
            target = task.get("target", "")
            items = task.get("items", [])
            target_upper = target.upper().strip()
            target_norm = self.normalize_category(target)
            
            rospy.loginfo(f"[Shopping List] Moving to next target: {target_upper} to pick up: {items}")
            self.append_bot_chat_message(f"Headed to the {target}...")
            
            # Update store status in active_todo_list to "navigating"
            with self.lock:
                if hasattr(self, 'active_todo_list') and self.active_todo_list:
                    for s in self.active_todo_list["stores"]:
                        if self.normalize_category(s["category"]) == target_norm:
                            s["status"] = "navigating"
            
            max_attempts = 3
            success = False
            for attempt in range(max_attempts):
                if not self.active_delivery_task:
                    break
                if attempt > 0:
                    rospy.logwarn(f"[Shopping List] Navigation to {target_upper} failed or timed out. Retrying (Attempt {attempt+1}/{max_attempts})...")
                    self.append_bot_chat_message(f"Goal blocked or navigation timed out. Recovering and retrying ({attempt+1}/{max_attempts})...")
                    self.stop_search(keep_delivery=True)
                    self.recover_robot()
                    
                # 1. Check if store is already dynamically mapped
                resolved_shop = None
                if hasattr(self, 'mapped_shops'):
                    for s in self.mapped_shops:
                        s_norm = self.normalize_category(s['type'])
                        if s_norm == target_norm:
                            resolved_shop = s
                            break
                            
                if resolved_shop:
                    # Direct navigation to the mapped storefront
                    rospy.loginfo(f"[Shopping List] Target {target_norm} is already mapped. Navigating directly.")
                    # Calculate corridor axis perpendicular to resolved_shop['yaw']
                    syaw = resolved_shop['yaw']
                    axis_x = -math.sin(syaw)
                    axis_y = math.cos(syaw)
                    
                    # Project robot's current position to determine approach direction
                    rx, ry, _ = self.get_current_robot_pose()
                    dx = resolved_shop['x'] - rx
                    dy = resolved_shop['y'] - ry
                    projection = dx * axis_x + dy * axis_y
                    sign = 1.0 if projection >= 0 else -1.0
                    
                    tx = resolved_shop['x'] + overshoot_m * sign * axis_x
                    ty = resolved_shop['y'] + overshoot_m * sign * axis_y
                    
                    success = self.navigate_to_pose(tx, ty, resolved_shop['yaw'], cost_thresh=98)
                else:
                    # 2. Check if we should do the complex sign-to-store workflow
                    if any(target_norm == self.normalize_category(entry['category']) for entry in self.sign_database):
                        rospy.loginfo(f"[Shopping List] Target {target_norm} is in sign database. Starting search.")
                        success = self.delivery_search_workflow(target_norm, is_part_of_task=True)
                    else:
                        # 3. Fallback to direct navigation to landmark tag
                        resolved_tag = None
                        target_key = target_norm.replace(" ", "_")
                        if target_key in self.tag_true_poses:
                            resolved_tag = target_key
                        else:
                            for k in self.tag_true_poses.keys():
                                if k in target_key or target_key in k:
                                    resolved_tag = k
                                    break
                                    
                        if resolved_tag:
                            rospy.loginfo(f"[Shopping List] Navigating directly to tag/landmark: {resolved_tag}")
                            self.navigating_to_tag = True
                            self.nav_to_tag_thread(resolved_tag)
                            
                            # Wait for arrival
                            pose_info = self.tag_true_poses[resolved_tag]
                            if "STORE_" in resolved_tag:
                                gx, gy = pose_info[0], pose_info[1]
                            else:
                                with self.lock:
                                    R = self.R.copy()
                                    T = self.T.copy()
                                tag_map = np.dot(R, np.array([pose_info[0], pose_info[1]])) + T
                                gx, gy = tag_map[0], tag_map[1]
                                
                            rx, ry, _ = self.get_current_robot_pose()
                            success = (math.hypot(rx - gx, ry - gy) < 0.35) and (self.active_delivery_task is not None)
                        else:
                            rospy.logerr(f"[Shopping List] Could not resolve target for: {target_upper}")
                            break # Break retry loop if category is completely unresolvable
                            
                if success:
                    break
                        
            if success and self.active_delivery_task:
                rospy.loginfo(f"[Shopping List] Successfully arrived at {target_upper}. Simulating pick up.")
                self.append_bot_chat_message(f"Arrived at the {target}. Commencing item pick up.")
                
                with self.lock:
                    if hasattr(self, 'active_todo_list') and self.active_todo_list:
                        for s in self.active_todo_list["stores"]:
                            if self.normalize_category(s["category"]) == target_norm:
                                s["status"] = "arrived"
                                
                for item in items:
                    if not self.active_delivery_task:
                        break
                    # Set item to picking_up
                    with self.lock:
                        if hasattr(self, 'active_todo_list') and self.active_todo_list:
                            for s in self.active_todo_list["stores"]:
                                if self.normalize_category(s["category"]) == target_norm:
                                    for it in s["items"]:
                                        if it["name"] == item:
                                            it["status"] = "picking_up"
                    self.append_bot_chat_message(f"Picking up {item}...")
                    rospy.sleep(2.0)
                    # Set item to completed
                    with self.lock:
                        if hasattr(self, 'active_todo_list') and self.active_todo_list:
                            for s in self.active_todo_list["stores"]:
                                if self.normalize_category(s["category"]) == target_norm:
                                    for it in s["items"]:
                                        if it["name"] == item:
                                            it["status"] = "completed"
                    self.append_bot_chat_message(f"Loaded {item}!")
                    self.tasks_fulfilled += 1
                    
                # Set store status to completed
                with self.lock:
                    if hasattr(self, 'active_todo_list') and self.active_todo_list:
                        for s in self.active_todo_list["stores"]:
                            if self.normalize_category(s["category"]) == target_norm:
                                s["status"] = "completed"
                self.append_bot_chat_message(f"Finished loading items from the {target}.")
            else:
                rospy.logwarn(f"[Shopping List] Failed to arrive at target category {target_upper}.")
                self.append_bot_chat_message(f"Failed to navigate to the {target}. Skipping items: {', '.join(items)}.")
                with self.lock:
                    if hasattr(self, 'active_todo_list') and self.active_todo_list:
                        for s in self.active_todo_list["stores"]:
                            if self.normalize_category(s["category"]) == target_norm:
                                s["status"] = "failed"
                                
        # FINAL STEP: Navigate to Pickup Point
        if self.active_delivery_task:
            rospy.loginfo("[Shopping List] All shopping list stores visited. Guiding robot to the Pickup Point.")
            self.append_bot_chat_message("All shopping items loaded! Guiding the robot to the Pickup Point for delivery.")
            
            target_norm = "PICKUP POINT"
            
            max_attempts = 3
            success = False
            for attempt in range(max_attempts):
                if not self.active_delivery_task:
                    break
                if attempt > 0:
                    rospy.logwarn(f"[Shopping List] Navigation to Pickup Point failed or timed out. Retrying (Attempt {attempt+1}/{max_attempts})...")
                    self.append_bot_chat_message(f"Pickup Point blocked or navigation timed out. Recovering and retrying ({attempt+1}/{max_attempts})...")
                    self.stop_search(keep_delivery=True)
                    self.recover_robot()
                    
                # Find in mapped shops
                resolved_shop = None
                if hasattr(self, 'mapped_shops'):
                    for s in self.mapped_shops:
                        s_norm = self.normalize_category(s['type'])
                        if s_norm == target_norm:
                            resolved_shop = s
                            break
                            
                if resolved_shop:
                    rospy.loginfo("[Shopping List] Navigating directly to mapped Pickup Point.")
                    syaw = resolved_shop['yaw']
                    axis_x = -math.sin(syaw)
                    axis_y = math.cos(syaw)
                    
                    rx, ry, _ = self.get_current_robot_pose()
                    dx = resolved_shop['x'] - rx
                    dy = resolved_shop['y'] - ry
                    projection = dx * axis_x + dy * axis_y
                    sign = 1.0 if projection >= 0 else -1.0
                    
                    tx = resolved_shop['x'] + overshoot_m * sign * axis_x
                    ty = resolved_shop['y'] + overshoot_m * sign * axis_y
                    
                    success = self.navigate_to_pose(tx, ty, resolved_shop['yaw'], cost_thresh=98)
                else:
                    if any(target_norm == self.normalize_category(entry['category']) for entry in self.sign_database):
                        success = self.delivery_search_workflow(target_norm, is_part_of_task=True)
                    else:
                        # Fallback to tag navigation
                        resolved_tag = None
                        for k in self.tag_true_poses.keys():
                            if "STORE_2" in k or "STORE_8" in k or "STORE_22" in k or "STORE_26" in k:
                                resolved_tag = k
                                break
                        if resolved_tag:
                            self.navigating_to_tag = True
                            self.nav_to_tag_thread(resolved_tag)
                            # Wait for arrival...
                            pose_info = self.tag_true_poses[resolved_tag]
                            gx, gy = pose_info[0], pose_info[1]
                            rx, ry, _ = self.get_current_robot_pose()
                            success = (math.hypot(rx - gx, ry - gy) < 0.35) and (self.active_delivery_task is not None)
                if success:
                    break
            
            if success and self.active_delivery_task:
                self.append_bot_chat_message("I have successfully arrived at the Pickup Point! Here are all your items. Enjoy!")
                with self.lock:
                    if hasattr(self, 'active_todo_list') and self.active_todo_list:
                        self.active_todo_list["status"] = "completed"
            else:
                self.append_bot_chat_message("I was unable to reach the Pickup Point. Please guide me manually or clear the path.")
                
        rospy.loginfo("[Shopping List] Sequential shopping list workflow completed.")
        self.active_delivery_task = None

    def visual_servoing_to_shop(self, target_category):
        """Rotates to find the shop, then uses visual servoing to approach perpendicularly."""
        if not self.yolo_model:
            rospy.logerr("YOLO model not loaded. Skipping visual servoing.")
            return False

        rate = rospy.Rate(10)
        found = False
        start_time = rospy.Time.now()
        
        # --- PHASE 1: ROTATE TO FIND ---
        rospy.loginfo("[Visual Servoing] Phase 1: Rotating to find target...")
        while not rospy.is_shutdown() and self.active_delivery_task and not found:
            if (rospy.Time.now() - start_time).to_sec() > 25.0: # Timeout
                break
                
            with self.lock:
                img = self.color_image.copy() if self.color_image is not None else None
            
            if img is not None:
                results = self.yolo_model.predict(img, conf=0.8, verbose=False)
                for res in results:
                    for box in res.boxes:
                        label = res.names[int(box.cls[0])].upper()
                        if target_category in label or label in target_category or \
                           ("CAFE" in target_category and "CAFE" in label) or \
                           ("HAMBURGER" in target_category and "HAMBURGER" in label) or \
                           ("PHARMACY" in target_category and "PHARMACY" in label) or \
                           ("PICKUP" in target_category and "PICK" in label):
                            found = True
                            break
                if found: break

            # Slow search rotation
            cmd = Twist()
            cmd.angular.z = 0.5
            self.cmd_vel_pub.publish(cmd)
            rate.sleep()
            
        if not found:
            self.cmd_vel_pub.publish(Twist())
            return False

        # --- PHASE 2: APPROACH PERPENDICULARLY ---
        rospy.loginfo("[Visual Servoing] Phase 2: Approaching shopfront...")
        arrived = False
        consecutive_lost_frames = 0
        
        while not rospy.is_shutdown() and self.active_delivery_task and not arrived:
            with self.lock:
                img = self.color_image.copy() if self.color_image is not None else None
                depth_raw = getattr(self, 'depth_raw', None)
            
            if img is None: 
                rate.sleep()
                continue
                
            results = self.yolo_model.predict(img, conf=0.8, verbose=False)
            best_box = None
            for res in results:
                for box in res.boxes:
                    label = res.names[int(box.cls[0])].upper()
                    if target_category in label or label in target_category or \
                       ("CAFE" in target_category and "CAFE" in label) or \
                       ("HAMBURGER" in target_category and "HAMBURGER" in label):
                        best_box = box.xyxy[0]
                        break
            
            if best_box is None:
                consecutive_lost_frames += 1
                # If we've already reached 60cm, just consider it arrived rather than failing
                if dist_to_shop < 0.6:
                    rospy.loginfo("[Visual Servoing] Target filled view/lost at close range. Assuming arrival.")
                    arrived = True
                    break
                
                if consecutive_lost_frames > 20: # Lost for 2 seconds
                    rospy.logwarn("[Visual Servoing] Lost target visibility.")
                    break
                self.cmd_vel_pub.publish(Twist())
                rate.sleep()
                continue
            
            consecutive_lost_frames = 0
            
            # 1. Horizontal Centering (Angular)
            img_w = img.shape[1]
            box_center_x = (best_box[0] + best_box[2]) / 2.0
            x_offset = (box_center_x - img_w/2.0) / (img_w/2.0) # -1 to 1
            
            # 2. Distance and Perpendicularity (Depth)
            dist_to_shop = 2.0 # Default
            angle_error = 0.0
            
            if depth_raw is not None:
                # Extract depth ROI around the box
                y1, x1, y2, x2 = int(best_box[1]), int(best_box[0]), int(best_box[3]), int(best_box[2])
                
                # Use a larger slice of the image to check depth even if box is at edge
                roi_y1 = max(0, y1)
                roi_y2 = min(depth_raw.shape[0], y2)
                roi_x1 = max(0, x1)
                roi_x2 = min(depth_raw.shape[1], x2)
                roi = depth_raw[roi_y1:roi_y2, roi_x1:roi_x2]
                
                if roi.size > 0:
                    valid_depths = roi[roi > 0.1]
                    if valid_depths.size > 0:
                        # Use 15th percentile to ensure we are looking at the 'front-most' edge
                        # This prevents overshooting into the shop interior
                        dist_to_shop = np.percentile(valid_depths, 15) 
                        
                        # Sanity check for Millimeters vs Meters
                        if dist_to_shop > 50.0: dist_to_shop /= 1000.0
                        
                        # Check perpendicularity: depth left vs right
                        half_w = roi.shape[1] // 2
                        if half_w > 5:
                            left_roi = roi[:, :half_w]
                            right_roi = roi[:, half_w:]
                            left_v = left_roi[left_roi > 0.1]
                            right_v = right_roi[right_roi > 0.1]
                            
                            if left_v.size > 0 and right_v.size > 0:
                                left_depth = np.median(left_v)
                                right_depth = np.median(right_v)
                                angle_error = (left_depth - right_depth) # Slant

            # 3. Control Logic
            cmd = Twist()
            
            # Distance stop (60cm)
            if dist_to_shop < 0.60:
                rospy.loginfo(f"[Visual Servoing] Target reached at {dist_to_shop:.2f}m")
                arrived = True
                break
                
            # Forward velocity (Proportional to distance)
            # P-controller for distance: goal is 0.6m
            dist_error = dist_to_shop - 0.60
            cmd.linear.x = 0.3 * dist_error
            
            # Side strafe (To get perpendicular)
            # Higher gain for alignment
            cmd.linear.y = -0.6 * angle_error 
            
            # Turn to keep centered
            cmd.angular.z = -1.5 * x_offset 
            
            # Safety checks/Clamping
            cmd.linear.x = np.clip(cmd.linear.x, 0.0, 0.2)
            cmd.linear.y = np.clip(cmd.linear.y, -0.2, 0.2)
            cmd.angular.z = np.clip(cmd.angular.z, -0.6, 0.6)
            
            self.cmd_vel_pub.publish(cmd)
            rate.sleep()
            
        self.cmd_vel_pub.publish(Twist()) # Final stop
        return arrived

    def rotate_to_yaw(self, target_yaw, p_gain=1.5, speed_limit=0.8, threshold=0.03):
        """Simple proportional controller for rotation in-place."""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self.active_delivery_task:
            _, _, ryaw = self.get_current_robot_pose()
            diff = (target_yaw - ryaw + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) < threshold: break
            
            cmd = Twist()
            cmd.angular.z = p_gain * diff 
            # Clamp
            cmd.angular.z = np.clip(cmd.angular.z, -speed_limit, speed_limit)
            self.cmd_vel_pub.publish(cmd)
            rate.sleep()
        self.cmd_vel_pub.publish(Twist()) # Stop

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
        
        if "STORE_" in tag_name:
            # Store coordinates are already in Map frame
            gx = tag_pt[0]
            gy = tag_pt[1]
            gyaw = psi_rad
        else:
            # AprilTag signboards are in World frame, transformed via R, T
            with self.lock:
                R = self.R
                T = self.T
            
            tag_map = np.dot(R, tag_pt) + T
            tag_x, tag_y = tag_map[0], tag_map[1]
            
            # Transform heading (psi) to map coordinates
            rot_angle = math.atan2(R[1, 0], R[0, 0])
            psi_rad_map = psi_rad + rot_angle
            
            gx = tag_x
            gy = tag_y
            gyaw = psi_rad_map
        
        rospy.loginfo(f"[Tag Nav] Target Pose in Map: ({gx:.2f}, {gy:.2f}, heading: {math.degrees(gyaw):.1f}°)")
        
        rate = rospy.Rate(10)
        start_time = rospy.Time.now()
        
        while not rospy.is_shutdown() and self.navigating_to_tag:
            rx, ry, ryaw = self.get_current_robot_pose()
            self.robot_x = rx
            self.robot_y = ry
            self.robot_yaw = ryaw
            
            # Distance to the target position
            dist = math.hypot(rx - gx, ry - gy)
            
            # Check if arrived at position (under signboard)
            if dist < 0.25:
                rospy.loginfo("[Tag Nav] Arrived at target signboard position successfully.")
                break
                
            # Publish path to goal
            path = Path()
            path.header.frame_id = "map"
            path.header.stamp = rospy.Time.now()
            
            p0 = PoseStamped()
            p0.header.frame_id = "map"
            p0.pose.position.x = rx
            p0.pose.position.y = ry
            
            # Orientation facing next point
            angle_to_next = math.atan2(gy - ry, gx - rx)
            q0 = tf.transformations.quaternion_from_euler(0, 0, angle_to_next)
            p0.pose.orientation = Quaternion(*q0)
            path.poses.append(p0)
            
            p1 = PoseStamped()
            p1.header.frame_id = "map"
            p1.pose.position.x = gx
            p1.pose.position.y = gy
            q1 = tf.transformations.quaternion_from_euler(0, 0, gyaw)
            p1.pose.orientation = Quaternion(*q1)
            path.poses.append(p1)
            
            self.path_pub.publish(path)
            
            now = rospy.Time.now()
            if (now - start_time).to_sec() > 120.0:
                rospy.logwarn("[Tag Nav] Navigation timeout (120 seconds).")
                break
                
            rate.sleep()
            
        self.navigating_to_tag = False
        self.stop_search(keep_delivery=getattr(self, 'is_part_of_task', False))

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

    def stop_search(self, keep_delivery=False):
        self.navigating_to_pose_active = False # Signal any while loops to stop
        if not keep_delivery:
            self.active_delivery_task = None
        
        rospy.set_param("/exploration_state", "IDLE")
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()
        self.path_pub.publish(path)
        self.cmd_vel_pub.publish(Twist())

    def recover_robot(self):
        """Costmap clearing recovery by rotating in place."""
        rospy.loginfo("[Recovery] Attempting costmap clearing recovery by rotating in place.")
        cmd = Twist()
        cmd.angular.z = 0.5
        self.cmd_vel_pub.publish(cmd)
        rospy.sleep(1.5)
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        rospy.sleep(1.0)


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
            # semantic logic is disabled as it relies on old pre-mapped stores
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
        if self.camera_info is None:
            return img
            
        K = np.array(self.camera_info.K).reshape((3, 3))
        dist_coeffs = np.zeros((4,1)) # Assume rectified image
        
        # --- Optimized YOLO Detections for Visualization ---
        # Only run at ~2Hz to prevent web UI lag during intense robot tasks
        now = time.time()
        if self.yolo_model is not None and self.viz_yolo:
            if now - self.last_yolo_viz_time > 0.5: # 2 FPS
                # Lower confidence for visualization helps see distant/partial shops
                self.last_yolo_viz_results = self.yolo_model.predict(img, conf=0.25, verbose=False)
                self.last_yolo_viz_time = now
                
            for res in self.last_yolo_viz_results:
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = res.names[cls]
                    conf = float(box.conf[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        def draw_single_tag(img, tag_id, size, rvec, tvec):
            if not self.viz_apriltag:
                return
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
                        pass # Ignore file stores, we use dynamic mapped_shops now
                rospy.loginfo("Web UI ignored file stores (using dynamic mapped_shops now).")
            except Exception as e:
                rospy.logwarn(f"Failed to load stores in Web UI: {e}")

    def load_tag_true_poses(self):
        active_yaml = "/home/linusv/project_5/catkin_ws/src/AprilTagLocalization/config/2025/re540_simulation.yaml"
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
        
        # Stores are dynamically mapped now, do not insert them into tag_true_poses

    def normalize_category(self, cat):
        if not cat:
            return ""
        c = cat.upper().strip()
        if "BURG" in c or "FAST" in c or "RESTAURANT" in c or "HAMB" in c:
            return "HAMBURGER"
        if "CAF" in c or "COFFEE" in c:
            return "CAFE"
        if "PHARM" in c or "MED" in c or "PILL" in c or "DRUG" in c:
            return "PHARMACY"
        if "CONV" in c or "STORE" in c or "SHOP" in c:
            return "CONVENIENCE STORE"
        if "PICK" in c or "POINT" in c:
            return "PICKUP POINT"
        return c

    def load_sign_database(self):
        db = []
        path = "/home/linusv/project_5/HW4/signboards.yaml"
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                if config:
                    for board_name, tags in config.items():
                        if not isinstance(tags, dict):
                            continue
                        for tag_id, data in tags.items():
                            # Map directions to angles relative to facing the tag (0 = up, 90 = left, -90 = right)
                            dir_str = data.get('direction', 'Up').strip().lower()
                            angle = 0.0
                            if dir_str == 'left': angle = 90.0
                            elif dir_str == 'right': angle = -90.0
                            elif dir_str == 'down': angle = 180.0
                            
                            db.append({
                                "tag": board_name,
                                "category": self.normalize_category(data.get('store_type', '')),
                                "direction": angle
                            })
                rospy.loginfo(f"Loaded {len(db)} sign entries from {path}")
            except Exception as e:
                rospy.logerr(f"Error loading sign database (YAML): {e}")
        return db

    def estimate_rigid_transform_2d(self, pts_true, pts_meas, true_yaws=None, meas_yaws=None):
        n = len(pts_true)
        if n == 0:
            return self.R, self.T
        elif n == 1:
            if true_yaws is not None and meas_yaws is not None and len(true_yaws) > 0 and len(meas_yaws) > 0:
                theta = meas_yaws[0] - true_yaws[0]
                R = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]
                ])
                T = np.array(pts_meas[0]) - np.dot(R, np.array(pts_true[0]))
                return R, T
            else:
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
                    true_yaws = []
                    meas_yaws = []
                    
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
                            # Get true pose position and yaw
                            x_true, y_true, psi_deg = self.tag_true_poses[signboard_name]
                            pts_true.append((x_true, y_true))
                            
                            # True yaw in world frame including orientation correction
                            q_x90 = tf.transformations.quaternion_about_axis(math.radians(90), [0, 1, 0])
                            q_y_90 = tf.transformations.quaternion_about_axis(math.radians(-90), [1, 0, 0])
                            q_correction = tf.transformations.quaternion_multiply(q_x90, q_y_90)
                            q_heading = tf.transformations.quaternion_about_axis(math.radians(psi_deg), [0, 0, 1])
                            q_true = tf.transformations.quaternion_multiply(q_heading, q_correction)
                            yaw_true = tf.transformations.euler_from_quaternion(q_true)[2]
                            true_yaws.append(yaw_true)
                            
                            # Measured pose position and yaw in map frame
                            tag_trans_odom, tag_rot_odom = tag_pose_odom
                            
                            # T_odom_tag
                            T_odom_tag = tf.transformations.quaternion_matrix(tag_rot_odom)
                            T_odom_tag[:3, 3] = tag_trans_odom
                            
                            # T_map_tag = T_map_odom * T_odom_tag
                            T_map_tag = np.dot(T_map_odom, T_odom_tag)
                            
                            pt_map_x = T_map_tag[0, 3]
                            pt_map_y = T_map_tag[1, 3]
                            pts_meas.append((pt_map_x, pt_map_y))
                            
                            q_meas = tf.transformations.quaternion_from_matrix(T_map_tag)
                            yaw_meas = tf.transformations.euler_from_quaternion(q_meas)[2]
                            meas_yaws.append(yaw_meas)
                            
                    # Dynamic rigid transform estimation removed for simulation alignment consistency.
                    # R, T = self.estimate_rigid_transform_2d(pts_true, pts_meas, true_yaws, meas_yaws)
                    # with self.lock:
                    #     self.R = R
                    #     self.T = T
                    
 
                                        
                    # Draw persistently detected AprilTags
                    for tag_id, tag_pose_odom in self.detected_tags.items():
                        tag_trans_odom, tag_rot_odom = tag_pose_odom
                        pt_odom = np.array([tag_trans_odom[0], tag_trans_odom[1], 0.0, 1.0])
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

                    # Draw shops on the SLAM map
                    if hasattr(self, 'mapped_shops'):
                        for idx, store in enumerate(self.mapped_shops):
                            # Store coords are directly in Map frame
                            s_x, s_y = store['x'], store['y']
                            s_col = int((s_x - origin.position.x) / resolution)
                            s_row = int((s_y - origin.position.y) / resolution)
                            s_row_flipped = height - 1 - s_row
                            
                            if 0 <= s_col < width and 0 <= s_row_flipped < height:
                                color = (128, 0, 128) # Purple if mapped
                                cv2.circle(color_map, (s_col, s_row_flipped), 6, color, -1)
                                cv2.circle(color_map, (s_col, s_row_flipped), 6, (255, 255, 255), 1)
                                
                                label = f"S{idx+1}: {store['type'][:3]}"
                                cv2.putText(color_map, label, (s_col + 8, s_row_flipped + 4), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                                        
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
                
                # Draw local target (Large Yellow Cross)
                with self.lock:
                    target = self.trunc_target
                if target:
                    tx, ty = target
                    t_col = int((tx - origin.position.x) / res)
                    t_row = height - 1 - int((ty - origin.position.y) / res)
                    if 0 <= t_col < width and 0 <= t_row < height:
                        cv2.drawMarker(viz, (t_col, t_row), (0, 255, 255), cv2.MARKER_CROSS, 8, 2)
                        cv2.putText(viz, "GOAL", (t_col + 5, t_row - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # Draw robot at frame center (0,0 local)
                r_col = int((0 - origin.position.x) / res)
                r_row = height - 1 - int((0 - origin.position.y) / res)
                cv2.circle(viz, (r_col, r_row), 4, (0, 0, 255), -1)
                cv2.circle(viz, (r_col, r_row), 4, (255, 255, 255), 1)
                
                canvas = viz
            
            if canvas is not None:
                # Resize and Zoom: Original is usually 200x200 (20m @ 0.1m res). 
                h, w = canvas.shape[:2]
                ch, cw = h // 4, w // 4 
                
                r_col = int((0 - origin.position.x) / res)
                r_row = h - 1 - int((0 - origin.position.y) / res)
                
                y1 = max(0, r_row - ch)
                y2 = min(h, r_row + ch)
                x1 = max(0, r_col - cw)
                x2 = min(w, r_col + cw)
                
                canvas_zoomed = canvas[y1:y2, x1:x2]
                disp = cv2.resize(canvas_zoomed, (400, 400), interpolation=cv2.INTER_NEAREST)
                
                # Add status text overlay
                cv2.putText(disp, f"Planner: {rospy.get_param('/local_planner_type', 'control_space')}", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if selected:
                    # heuristic to get velocity from selected path length
                    path_len = sum([np.hypot(selected[i][0]-selected[i-1][0], selected[i][1]-selected[i-1][1]) for i in range(1, len(selected))])
                    cv2.putText(disp, f"Moving Path: {path_len:.1f}m", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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

    has_key = False
    param_key = rospy.get_param("/openai_api_key", "").strip()
    if param_key.startswith("sk-"):
        has_key = True
    elif os.getenv("OPENAI_API_KEY", "").strip().startswith("sk-"):
        has_key = True
    elif os.path.exists("/home/linusv/project_5/HW4/ChatGPT_API_KEY.txt"):
        try:
            with open("/home/linusv/project_5/HW4/ChatGPT_API_KEY.txt", "r") as f:
                content = f.read().strip()
            import re
            match = re.search(r'\b(sk-[a-zA-Z0-9_-]+)\b', content)
            if match:
                has_key = True
        except Exception:
            pass

    local_planner = rospy.get_param("/local_planner_type", "teb")

    status = {
        "x": round(server.robot_x, 2) if hasattr(server, 'robot_x') else None,
        "y": round(server.robot_y, 2) if hasattr(server, 'robot_y') else None,
        "yaw": round(server.robot_yaw, 2) if hasattr(server, 'robot_yaw') else None,
        "explored_area": round(server.map_coverage_m2, 1) if hasattr(server, 'map_coverage_m2') else 0.0,
        "shops_detected": len(server.mapped_shops) if hasattr(server, "mapped_shops") else 0,
        "tasks_fulfilled": server.tasks_fulfilled if hasattr(server, 'tasks_fulfilled') else 0,
        "tags_detected": len(server.detected_tags) if hasattr(server, 'detected_tags') else 0,
        "exploration_status": status_str,
        "exploration_state": explore_state,
        "has_api_key": has_key,
        "searching_tag": server.searching_tag if hasattr(server, 'searching_tag') else False,
        "navigating_to_tag": server.navigating_to_tag if hasattr(server, 'navigating_to_tag') else False,
        "local_planner": local_planner,
        "mapped_shops": server.mapped_shops if hasattr(server, "mapped_shops") else [],
        "overshoot_cm": (server.overshoot_m * 100.0) if hasattr(server, 'overshoot_m') else 0.0,
        "chat_messages": server.delivery_chat_history if hasattr(server, 'delivery_chat_history') else [],
        "todo_list": server.active_todo_list if hasattr(server, 'active_todo_list') else None
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

@app.route('/api/set_overshoot', methods=['POST'])
def set_overshoot():
    data = request.json or {}
    overshoot_cm = float(data.get('overshoot_cm', 0.0))
    server.overshoot_m = overshoot_cm / 100.0
    rospy.loginfo(f"Approach overshoot set to {overshoot_cm} cm in backend.")
    return jsonify({"status": "success", "overshoot_cm": overshoot_cm})

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

@app.route('/api/set_planner', methods=['POST'])
def set_planner():
    data = request.json
    planner = data.get('planner', 'control_space') # 'control_space' or 'teb'
    rospy.set_param("/local_planner_type", planner)
    rospy.loginfo(f"Set /local_planner_type parameter to {planner}")
    return jsonify({"status": "success", "planner": planner})

@app.route('/api/detect', methods=['POST'])
def api_detect():
    rospy.loginfo("Detect Mode triggered: Mapping shopfront.")
    import threading
    def detect_thread():
        # Do a quick check
        det = server.check_for_shop("ANY")  # Match any detected storefront!
        if not det:
            rospy.logwarn("No shop detected directly in front.")
            return
            
        rx, ry, ryaw = server.get_current_robot_pose()
        # Snap robot's yaw to nearest 90 degrees (corridors and walls are grid-aligned)
        snapped_yaw = round(ryaw / (math.pi / 2.0)) * (math.pi / 2.0)
        
        wall_dist = server.get_distance_to_wall_ahead(max_dist=6.0)
        if wall_dist is None:
            wall_dist = det['depth'] + 0.15 if det else 2.0
            
        shop_x = rx + wall_dist * math.cos(snapped_yaw)
        shop_y = ry + wall_dist * math.sin(snapped_yaw)
        
        # Outward normal points opposite to the snapped heading (from wall to robot)
        outward_normal = (snapped_yaw + math.pi) % (2*math.pi) - math.pi
        
        # The approach point is 60cm in front of the wall
        target_dist = 0.60
        target_x = shop_x + target_dist * math.cos(outward_normal)
        target_y = shop_y + target_dist * math.sin(outward_normal)
        
        target_yaw = (outward_normal + math.pi) % (2*math.pi) - math.pi
        
        shop_info = {
            "name": det['label'] + f"_{len(server.mapped_shops)+1}",
            "type": det['label'],
            "x": target_x,
            "y": target_y,
            "yaw": target_yaw
        }
        server.mapped_shops.append(shop_info)
        rospy.loginfo(f"Mapped {shop_info['name']} to approach point ({target_x:.2f}, {target_y:.2f})")
        
    threading.Thread(target=detect_thread).start()
    return jsonify({"status": "success", "message": "Shop mapping initiated."})

@app.route('/api/delete_shop', methods=['POST'])
def api_delete_shop():
    data = request.json
    shop_name = data.get('name')
    if hasattr(server, 'mapped_shops'):
        server.mapped_shops = [s for s in server.mapped_shops if s['name'] != shop_name]
        rospy.loginfo(f"Deleted shop: {shop_name}")
    return jsonify({"status": "success"})
    
@app.route('/api/goto_shop', methods=['POST'])
def api_goto_shop():
    data = request.json
    shop_name = data.get('name')
    overshoot_cm = float(data.get('overshoot_cm', 0.0))
    overshoot_m = overshoot_cm / 100.0
    
    if hasattr(server, 'mapped_shops'):
        for s in server.mapped_shops:
            if s['name'] == shop_name:
                # Calculate corridor axis perpendicular to s['yaw'] (which faces the shop)
                axis_x = -math.sin(s['yaw'])
                axis_y = math.cos(s['yaw'])
                
                # Project robot's current position to determine approach direction
                rx, ry, _ = server.get_current_robot_pose()
                dx = s['x'] - rx
                dy = s['y'] - ry
                projection = dx * axis_x + dy * axis_y
                sign = 1.0 if projection >= 0 else -1.0
                
                target_x = s['x'] + overshoot_m * sign * axis_x
                target_y = s['y'] + overshoot_m * sign * axis_y
                
                rospy.loginfo(f"Going to mapped shop {shop_name} (overshoot: {overshoot_cm}cm along corridor)")
                # Navigate to target using relaxed costmap check to allow getting closer to the wall!
                server.navigate_to_pose(target_x, target_y, s['yaw'], cost_thresh=98)
                return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Shop not found"})


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

@app.route('/api/viz_toggle', methods=['POST'])
def api_viz_toggle():
    data = request.json or {}
    layer = data.get('layer', '')
    enabled = data.get('enabled', True)
    
    if layer == 'apriltag':
        server.viz_apriltag = enabled
    elif layer == 'yolo':
        server.viz_yolo = enabled
        
    return jsonify({"status": "success", "layer": layer, "enabled": enabled})

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

@app.route('/api/delivery/clear', methods=['POST'])
def delivery_clear():
    server.active_todo_list = None
    server.stop_robot()
    return jsonify({"status": "success", "message": "Delivery plan cleared."})

@app.route('/api/delivery/send', methods=['POST'])
def delivery_send():
    data = request.json
    message = data.get('message', '').strip()
    overshoot_cm = float(data.get('overshoot_cm', 0.0))
    server.overshoot_m = overshoot_cm / 100.0
    if not message:
        return jsonify({"reply": "I did not receive a message. Please say something!"})
        
    server.append_user_chat_message(message)
    
    tasks = []
    reply = ""
    
    use_local = rospy.get_param("/use_local_ai", True)
    
    if not use_local:
        # Call the /llm_query service of the stateless client to parse the user's intent!
        try:
            rospy.wait_for_service("llm_query", timeout=2.0)
            llm_query_srv = rospy.ServiceProxy("llm_query", LLMQuery)
            
            prompt = (
                f"The user wants to get some items. They said: '{message}'.\n"
                f"Please match the items they requested to one of our 5 store categories:\n"
                f"- 'Cafe' (for coffee, ice coffee, tea, latte, cappuccino, espresso, drinks, etc.)\n"
                f"- 'Convenience store' (for onigiri, banana, snacks, chips, tissue, water, general shop items, etc.)\n"
                f"- 'Fast-food restaurant' (for burger, fries, hamburger, food, pizza, chicken nuggets, etc.)\n"
                f"- 'Pharmacy' (for medicine, aspirin, cough drop, band-aid, pills, etc.)\n"
                f"- 'Pickup Point' (for packages, parcels, etc.)\n\n"
                f"If there are items, group them by category.\n"
                f"Return ONLY a raw JSON object with two keys:\n"
                f"1. 'tasks': A list of objects, where each object has:\n"
                f"   - 'target': The exact category string ('Cafe', 'Convenience store', 'Fast-food restaurant', 'Pharmacy', or 'Pickup Point')\n"
                f"   - 'items': A list of strings containing the items matched to this category.\n"
                f"2. 'reply': A polite conversational response to display to the user, listing the items and where you will pick them up (e.g., 'sounds good, here is where I will go to get your items: Cafe for coffee and ice coffee, Convenience store for onigiri and banana, Fast-food restaurant for burger and fries.').\n\n"
                f"Example JSON output:\n"
                f"{{\n"
                f"  \"tasks\": [\n"
                f"    {{\"target\": \"Cafe\", \"items\": [\"coffee\", \"ice coffee\"]}},\n"
                f"    {{\"target\": \"Convenience store\", \"items\": [\"onigiri\", \"banana\"]}},\n"
                f"    {{\"target\": \"Fast-food restaurant\", \"items\": [\"burger\", \"fries\"]}}\n"
                f"  ],\n"
                f"  \"reply\": \"sounds good, here is where I will go to get your items: Cafe for coffee and ice coffee, Convenience store for onigiri and banana, Fast-food restaurant for burger and fries.\"\n"
                f"}}"
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
            tasks = parsed.get("tasks", [])
            reply = parsed.get("reply", "Understood! Executing tasks.")
            
        except Exception as e:
            rospy.logwarn(f"LLM Query failed or timed out: {e}. Falling back to offline parser.")
            use_local = True
            
    if use_local:
        # Standard local backup parser for simple keywords
        m = message.lower()
        tasks = []
        cafe_items = []
        conv_items = []
        burger_items = []
        pharm_items = []
        
        # Simple local keywords mapping
        if "coffee" in m or "cafe" in m or "drink" in m or "tea" in m:
            items = []
            if "ice coffee" in m or "ice-coffee" in m:
                items.append("ice coffee")
            if "coffee" in m and "ice coffee" not in m:
                items.append("coffee")
            if not items:
                items.append("coffee")
            cafe_items.extend(items)
            
        if "onigiri" in m or "banana" in m or "convenience" in m or "store" in m or "shop" in m:
            items = []
            if "onigiri" in m: items.append("onigiri")
            if "banana" in m: items.append("banana")
            if not items: items.append("general item")
            conv_items.extend(items)
            
        if "burger" in m or "fries" in m or "food" in m or "restaurant" in m or "hamburger" in m:
            items = []
            if "burger" in m or "hamburger" in m: items.append("burger")
            if "fries" in m: items.append("fries")
            if not items: items.append("burger")
            burger_items.extend(items)
            
        if "med" in m or "pharm" in m or "pill" in m or "sick" in m or "drug" in m:
            pharm_items.append("medicine")
            
        if cafe_items:
            tasks.append({"target": "Cafe", "items": cafe_items})
        if conv_items:
            tasks.append({"target": "Convenience store", "items": conv_items})
        if burger_items:
            tasks.append({"target": "Fast-food restaurant", "items": burger_items})
        if pharm_items:
            tasks.append({"target": "Pharmacy", "items": pharm_items})
            
        if tasks:
            reply = "sounds good, here is where I will go to get your items: "
            parts = []
            for t in tasks:
                parts.append(f"{t['target']} for {', '.join(t['items'])}")
            reply += ", ".join(parts) + "."
        else:
            reply = "I'm not sure which storefront matches that request. Try asking for coffee, a burger, medicine, or convenience store items!"
            
    server.append_bot_chat_message(reply)
    
    if tasks:
        # Build active_todo_list
        server.active_todo_list = {
            "status": "in_progress",
            "stores": [
                {
                    "category": t["target"],
                    "items": [{"name": item, "status": "pending"} for item in t["items"]],
                    "status": "pending"
                } for t in tasks
            ]
        }
        server.start_shopping_list_workflow(tasks)
    else:
        server.active_todo_list = {
            "status": "idle",
            "stores": []
        }
        
    return jsonify({"reply": reply, "tasks": tasks})

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
