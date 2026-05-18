#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import threading
import yaml
import os
import re
from flask import Flask, render_template, Response, request, jsonify
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import tf
import tf.transformations

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
        
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.map_coverage_m2 = 0.0
        self.stores = []
        
        self.lock = threading.Lock()
        self.tf_listener = tf.TransformListener()

        # Load AprilTag bundles configuration & Store coordinates
        self.yaml_path = "/home/linusv/project_5/HW4/tags.yaml"
        self.bundles = self.load_bundles(self.yaml_path)
        self.load_stores_from_txt()

        # ROS Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # ROS Subscribers
        rospy.Subscriber('/camera/color/image_raw', Image, self.color_cb)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_cb)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.cam_info_cb)
        rospy.Subscriber('/rtabmap/grid_map', OccupancyGrid, self.map_cb)
        
        if AprilTagDetectionArray is not None:
            rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.tags_cb)

        rospy.loginfo("Web Server Node initialized.")

    def color_cb(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Draw AprilTags if we have camera info
            with self.lock:
                if self.camera_info is not None and len(self.tag_detections) > 0:
                    cv_image = self.draw_tags(cv_image)
                    
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
                self.camera_info = msg.K # 3x3 intrinsic matrix

    def tags_cb(self, msg):
        with self.lock:
            self.tag_detections = msg.detections

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
        K = np.array(self.camera_info).reshape(3, 3)
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
                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                        pass
                        
                    # Draw shops
                    for idx, store in enumerate(self.stores):
                        s_col = int((store[0] - origin.position.x) / resolution)
                        s_row = int((store[1] - origin.position.y) / resolution)
                        s_row_flipped = height - 1 - s_row
                        
                        if 0 <= s_col < width and 0 <= s_row_flipped < height:
                            # Cyan circle marker
                            cv2.circle(color_map, (s_col, s_row_flipped), 6, (255, 255, 0), -1)
                            cv2.circle(color_map, (s_col, s_row_flipped), 6, (255, 255, 255), 1)
                            cv2.putText(color_map, f"S{idx+1}", (s_col + 8, s_row_flipped + 4), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                        
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

@app.route('/api/status')
def api_status():
    status = {
        "x": round(server.robot_x, 2) if hasattr(server, 'robot_x') else None,
        "y": round(server.robot_y, 2) if hasattr(server, 'robot_y') else None,
        "yaw": round(server.robot_yaw, 2) if hasattr(server, 'robot_yaw') else None,
        "explored_area": round(server.map_coverage_m2, 1) if hasattr(server, 'map_coverage_m2') else 0.0,
        "shops_detected": len(server.tag_detections) if hasattr(server, 'tag_detections') else 0
    }
    return jsonify(status)

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

if __name__ == '__main__':
    global server
    server = RobotWebServer()
    # Run flask in a separate thread to allow ROS to spin (or vice-versa)
    # Using threaded=True allows Flask to handle multiple connections
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)).start()
    rospy.spin()
