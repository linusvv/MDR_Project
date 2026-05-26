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
        self.stores = []
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
        # Relevant nodes for navigation safety
        relevant_nodes = [
            "/control_space_planner_node", 
            "/heightmap_costmap_node",
            "/heightmap_node",
            "/agent_node",
            "/control_space_planner_python",
            "/teb_planner_node",
            "/custom_vector_planner"
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

    def navigate_to_pose(self, x, y, yaw):
        """Standard method to navigate to a specific map pose using the global planner."""
        self.navigating_to_pose_active = True
        
        # Enable C++ planner
        rospy.set_param("/exploration_state", "EXPLORE")
        rospy.set_param("/exploration_paused", False)
        
        rate = rospy.Rate(10)
        start_time = rospy.Time.now()
        
        while not rospy.is_shutdown() and self.navigating_to_pose_active and self.active_delivery_task:
            rx, ry, ryaw = self.get_current_robot_pose()
            dist = math.hypot(rx - x, ry - y)
            
            if dist < 0.4: # Distance threshold
                # Check orientation near end
                angle_diff = (yaw - ryaw + math.pi) % (2 * math.pi) - math.pi
                if abs(angle_diff) < 0.15:
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
                break
            rate.sleep()
            
        self.navigating_to_pose_active = False
        self.stop_search(keep_delivery=True) # Clears path but keeps workflow target

    def approach_shop_via_waypoint(self, target_category):
        """Find shop via sweep -> Compute Waypoint -> Go there -> Face shop."""
        if not self.yolo_model:
            return False

        # PHASE 1: SEARCH (Sweep 90 deg left and right)
        rate = rospy.Rate(10)
        found = False
        shop_img_x = 0
        shop_depth = 0
        shop_label = ""
        
        _, _, start_yaw = self.get_current_robot_pose()
        rospy.loginfo(f"[Waypoint Appr] Phase 1: Scanning for {target_category} (90 deg sweep)...")
        
        # Target sequence: +90 (Left), back to 0 (Center), then -90 (Right)
        sweep_rel_targets = [math.radians(90), 0, math.radians(-90)]
        target_idx = 0
        
        start_time = rospy.Time.now()
        while not rospy.is_shutdown() and self.active_delivery_task and not found:
            if (rospy.Time.now() - start_time).to_sec() > 50.0: break
            
            with self.lock:
                img = self.color_image.copy() if self.color_image is not None else None
                depth_raw = getattr(self, 'depth_raw', None)
                
            if img is not None:
                results = self.yolo_model.predict(img, conf=0.35, verbose=False)
                for res in results:
                    for box in res.boxes:
                        label = res.names[int(box.cls[0])].upper()
                        if target_category in label or label in target_category or \
                           ("CAFE" in target_category and "CAFE" in label) or \
                           ("HAMBURGER" in target_category and "HAMBURGER" in label) or \
                           ("PHARMACY" in target_category and "PHARMACY" in label) or \
                           ("PICKUP" in target_category and "PICK" in label):
                            
                            shop_img_x = (box.xyxy[0][0] + box.xyxy[0][2]) / 2.0
                            y1, x1, y2, x2 = int(box.xyxy[0][1]), int(box.xyxy[0][0]), int(box.xyxy[0][3]), int(box.xyxy[0][2])
                            if depth_raw is not None:
                                h, w = depth_raw.shape
                                roi = depth_raw[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                                valids = roi[roi > 0.1]
                                if valids.size > 0:
                                    shop_depth = np.median(valids)
                                    shop_label = label
                                    found = True
                                    break
                if found: break
            
            # Rotation logic: reach target 1 (+90), then target 2 (-90)
            _, _, curr_yaw = self.get_current_robot_pose()
            target_yaw = (start_yaw + sweep_rel_targets[target_idx] + math.pi) % (2 * math.pi) - math.pi
            diff = (target_yaw - curr_yaw + math.pi) % (2 * math.pi) - math.pi
            
            if abs(diff) < 0.2: # Match threshold
                target_idx += 1
                if target_idx >= len(sweep_rel_targets):
                    break
                    
            cmd = Twist()
            # If we are heading from +90 to -90, the diff will be -180. 
            # Force it to take the "Right" turn (shortest path)
            cmd.angular.z = 0.5 * np.sign(diff)
            self.cmd_vel_pub.publish(cmd)
            rate.sleep()
            
        if not found: 
            self.cmd_vel_pub.publish(Twist())
            return False
        
        self.cmd_vel_pub.publish(Twist())
        rospy.loginfo(f"[Waypoint Appr] Found {shop_label} at {shop_depth:.2f}m.")
        rospy.sleep(0.5)
            cmd.angular.z = 0.5 * rot_dir
            self.cmd_vel_pub.publish(cmd)
            rate.sleep()
            
        if not found: 
            self.cmd_vel_pub.publish(Twist())
            return False
        
        # Stop briefly to finalize position
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(0.5)

        # PHASE 2: COMPUTE GLOBAL COORDINATES OF SHOP
        rospy.loginfo(f"[Waypoint Appr] Phase 2: Computing pose for {shop_label}...")
        rx, ry, ryaw = self.get_current_robot_pose()
        
        # Approximate offset from center to angle (FOV is ~85 deg)
        img_w = 640.0 # Standard
        pix_offset = shop_img_x - (img_w/2.0)
        angle_to_shop_rel = -(pix_offset / (img_w/2.0)) * (math.radians(42.0)) # Rough mapping
        angle_to_shop_global = ryaw + angle_to_shop_rel
        
        # Shop map location
        shop_map_x = rx + shop_depth * math.cos(angle_to_shop_global)
        shop_map_y = ry + shop_depth * math.sin(angle_to_shop_global)
        
        # Waypoint: 0.5m in front of shop along the approach vector
        # Approach vector is from robot to shop (angle_to_shop_global)
        # We want to be at a position that is (shop_depth - 0.5) away from robot
        target_x = rx + (shop_depth - 0.5) * math.cos(angle_to_shop_global)
        target_y = ry + (shop_depth - 0.5) * math.sin(angle_to_shop_global)
        target_yaw = angle_to_shop_global # Face the shop
        
        rospy.loginfo(f"[Waypoint Appr] Shop at ({shop_map_x:.2f}, {shop_map_y:.2f}). Waypoint at ({target_x:.2f}, {target_y:.2f})")
        
        # PHASE 3: ACTUALLY NAVIGATE THERE
        self.navigate_to_pose(target_x, target_y, target_yaw)
        
        # PHASE 4: FINAL ROTATION TO FACE SHOP EXACTLY
        self.rotate_to_yaw(target_yaw)
        return True


    def start_delivery_search(self, category):
        """Starts the complex workflow: Find sign -> Go to sign -> Align -> Crab search."""
        self.stop_robot() # Reset all states
        
        self.active_delivery_task = category.upper()
        self.task_thread = threading.Thread(target=self.delivery_search_workflow, args=(self.active_delivery_task,))
        self.task_thread.daemon = True
        self.task_thread.start()
        return True

    def delivery_search_workflow(self, target_category):
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
                
            for entry in self.sign_database:
                if target_category == entry['category'] or target_category in entry['category']:
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
                self.active_delivery_task = None
                return

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
                return

            # 3. ALIGN TO THE DIRECTION FROM THE ROAD SIGN
            rospy.loginfo(f"[Delivery Task] Step 2: Aligning to sign direction ({target_dir} deg)")
            if best_tag not in self.tag_true_poses:
                rospy.logerr(f"[Delivery Task] Critical error: {best_tag} missing from tag_true_poses")
                self.active_delivery_task = None
                return
                
            pose_info = self.tag_true_poses[best_tag]
            tag_true_yaw_deg = pose_info[2]
            
            # Map yaw = true_yaw + rotation from R
            rot_angle = math.atan2(current_R[1, 0], current_R[0, 0])
            target_yaw = math.radians(tag_true_yaw_deg) + rot_angle + math.radians(target_dir)
            
            rospy.loginfo(f"[Delivery Task] Rotating to target yaw: {math.degrees(target_yaw):.1f} deg")
            self.rotate_to_yaw(target_yaw)
            
            # 4. DISCOVER SHOP & APPROACH USING WAYPOINT
            rospy.loginfo(f"[Delivery Task] Step 3: Discovering and approaching {target_category}...")
            # Use the new waypoint-based approach for reliability
            success = self.approach_shop_via_waypoint(target_category)
            
            if success:
                rospy.loginfo(f"[Delivery Task] SUCCESS: Arrived at {target_category}!")
            else:
                rospy.logwarn(f"[Delivery Task] Failed to final-approach {target_category} storefront.")
        except Exception as e:
            rospy.logerr(f"[Delivery Task] Unexpected error in workflow: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            
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
                results = self.yolo_model.predict(img, conf=0.4, verbose=False)
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
                
            results = self.yolo_model.predict(img, conf=0.4, verbose=False)
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
                        dist_to_shop = np.percentile(valid_depths, 30) # Use 30th percentile to get the front face
                        
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
            
            # Distance stop (40cm)
            if dist_to_shop < 0.40:
                rospy.loginfo(f"[Visual Servoing] Target reached at {dist_to_shop:.2f}m")
                arrived = True
                break
                
            # Forward velocity (Proportional to distance)
            # P-controller for distance: goal is 0.4m
            dist_error = dist_to_shop - 0.40
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

    def rotate_to_yaw(self, target_yaw):
        """Simple proportional controller for rotation in-place."""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self.active_delivery_task:
            _, _, ryaw = self.get_current_robot_pose()
            diff = (target_yaw - ryaw + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) < 0.05: break
            
            cmd = Twist()
            cmd.angular.z = 1.0 * diff # P controller
            # Clamp
            if cmd.angular.z > 0.6: cmd.angular.z = 0.6
            if cmd.angular.z < -0.6: cmd.angular.z = -0.6
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
            if dist < 0.5:
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
        if self.camera_info is None:
            return img
            
        K = np.array(self.camera_info.K).reshape((3, 3))
        dist_coeffs = np.zeros((4,1)) # Assume rectified image
        
        # --- Optimized YOLO Detections for Visualization ---
        # Only run at ~5Hz to prevent web UI lag
        now = time.time()
        if self.yolo_model is not None:
            if now - self.last_yolo_viz_time > 0.2: # 5 FPS
                self.last_yolo_viz_results = self.yolo_model.predict(img, conf=0.4, verbose=False)
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
        
        # Manually add the store coordinates to navigation list
        store_names = ["STORE_1", "STORE_2", "STORE_3", "STORE_4", "STORE_5", "STORE_6", "STORE_7", "STORE_8"]
        for idx, coord in enumerate(self.stores):
            if idx < len(store_names):
                name = store_names[idx]
                # In simulation, these provided coordinates are actually already in the Map frame
                # They are not true-world GPS coords that need R/T transformation.
                self.tag_true_poses[name] = (coord[0], coord[1], 0.0)

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
                                "category": data.get('store_type', '').upper(),
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
                    for idx, store in enumerate(self.stores):
                        # Store coords are directly in Map frame
                        s_x, s_y = store[0], store[1]
                        s_col = int((s_x - origin.position.x) / resolution)
                        s_row = int((s_y - origin.position.y) / resolution)
                        s_row_flipped = height - 1 - s_row
                        
                        if 0 <= s_col < width and 0 <= s_row_flipped < height:
                            is_mapped = idx in self.shop_categories
                            color = (128, 0, 128) if is_mapped else (255, 255, 0) # Purple if mapped, Cyan if unmapped (BGR)
                            cv2.circle(color_map, (s_col, s_row_flipped), 6, color, -1)
                            cv2.circle(color_map, (s_col, s_row_flipped), 6, (255, 255, 255), 1)
                            
                            label = f"S{idx+1}"
                            if is_mapped:
                                label += f": {self.shop_categories[idx]['storefront'][:3]}"
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

    has_key = bool(rospy.get_param("/openai_api_key", "").strip()) or \
              bool(os.getenv("OPENAI_API_KEY")) or \
              os.path.exists("/home/linusv/project_5/HW4/ChatGPT_API_KEY.txt")

    local_planner = rospy.get_param("/local_planner_type", "control_space")

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
        "navigating_to_tag": server.navigating_to_tag if hasattr(server, 'navigating_to_tag') else False,
        "local_planner": local_planner
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

@app.route('/api/set_planner', methods=['POST'])
def set_planner():
    data = request.json
    planner = data.get('planner', 'control_space') # 'control_space' or 'teb'
    rospy.set_param("/local_planner_type", planner)
    rospy.loginfo(f"Set /local_planner_type parameter to {planner}")
    return jsonify({"status": "success", "planner": planner})

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
            f"1. 'target' (string, the exact target category or store name, e.g. 'Café' or 'BLUE CAFE' or 'STORE_1')\n"
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
        if "burger" in m or "food" in m or "restaurant" in m or "hungry" in m:
            target = "Hamburger"
            reply += "Ah! You want a burger. I will navigate to the Hamburger restaurant!"
        elif "coffee" in m or "cafe" in m or "drink" in m or "ice" in m or "tea" in m:
            target = "Cafe"
            reply += "Ah! You want coffee. I will navigate to the Cafe!"
        elif "med" in m or "pharm" in m or "pill" in m or "sick" in m or "drug" in m:
            target = "Pharmacy"
            reply += "Ah! You need a pharmacy. I will navigate to the Pharmacy!"
        elif "store" in m or "shop" in m or "item" in m or "convenience" in m:
            target = "Convenience Store"
            reply += "Ah! You need the Convenience store. I will navigate there!"
        elif "pickup" in m or "parcel" in m or "package" in m:
            target = "Pickup Point"
            reply += "Ah! You have a package. I will navigate to the Pickup Point!"
        else:
            reply = "I'm not sure which storefront matches that request. Try asking for coffee, a burger, medicine, or the convenience store!"
            
    # Direct navigation resolution
    resolved_tag = None
    if target:
        target_upper = target.upper().strip()
        
        # Check if we should use the complex sign-to-store workflow
        if any(target_upper in entry['category'] for entry in server.sign_database):
            rospy.loginfo(f"[Delivery API] Starting complex workflow for category: {target_upper}")
            server.start_delivery_search(target_upper)
            return jsonify({"reply": reply, "target": target})

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
