#!/usr/bin/python3
# coding=utf8
"""
YOLO + RealSense Depth Integrated Detection Node

Pipeline:
  1. /camera/color/image_raw              -> Run YOLO inference
  2. /camera/aligned_depth_to_color/image_raw  -> Get depth value at the bounding box center
  3. /camera/color/camera_info            -> Extract camera intrinsics (fx, fy, cx, cy)

  -> Calculate 3D camera coordinates (X_cam, Y_cam, Z_cam)
  -> Apply calibration offset to transform into Robot Arm coordinates
  -> Publish /yolo/arm_point (geometry_msgs/Point)   → Consumed by pick_arm_node

Published Topics:
  - /yolo/image_result  : Annotated image (for visualization)
  - /yolo/object_point  : 3D coordinates in Camera frame (PointStamped)
  - /yolo/arm_point     : 3D coordinates in Robot Arm frame (Point)

Subscribed Topics:
  - /yolo/class (std_msgs/String) : Set target class name at runtime
      - Empty string ("") -> Select the object with the highest confidence among all classes
      - Specify a class name -> Select the object with the highest confidence within that class
      Examples:
        rostopic pub /yolo/class std_msgs/String "data: 'hamburger'"
        rostopic pub /yolo/class std_msgs/String "data: ''"
"""

import os

# ── CPU-spike control at init ─────────────────────────────────────────────
# Loading torch/ultralytics/TensorRT spins up OpenMP/MKL/OpenBLAS thread pools
# that briefly grab every core during init. Cap them BEFORE importing torch,
# numpy & cv2 (these read the env vars at import time, so this must run first).
# Inference is on CUDA, so the CPU thread count barely affects throughput.
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
os.environ.setdefault("OMP_WAIT_POLICY",      "PASSIVE")  # idle threads sleep, don't spin

import threading

import rospy
import numpy as np
import cv2
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
from ultralytics import YOLO

# ── Parameters ────────────────────────────────────────────────────────────
MODEL_PATH   = "/home/ee478_team1/catkin_ws/src/MDR_Project/catkin_ws/src/controller/best.engine"
CONF_THRESH  = 0.8
DEPTH_SCALE  = 0.001      # RealSense: mm -> meters
DEPTH_RADIUS = 3          # NxN sampling radius around bbox center (for noise reduction)
MAX_DEPTH    = 1.5        # Ignore detections beyond this distance (meters)
MIN_DEPTH    = 0.05       # Ignore detections closer than this distance (meters)

# ── Activation Gate ──────────────────────────────────────────────────────
# Controlled via /yolo_grab/activate (Bool). Starts INACTIVE so the arm does
# NOT pick anything until the mission orchestrator switches to grab mode.
# Without this gate the node would run inference on every frame and publish
# /yolo/arm_point immediately, making pick_arm grab objects with no LLM command.
_active      = False
_active_lock = threading.Lock()
_last_inference_time = 0.0
_inference_interval = 0.15   # cap inference rate to ~6.7 Hz to prevent CPU spikes

def activate_cb(msg):
    global _active
    with _active_lock:
        _active = msg.data
    rospy.loginfo(f"[yolo_detector] {'ACTIVATED' if msg.data else 'DEACTIVATED'}")

# ── Target Class (set at runtime via /target_item, sent by the orchestrator) ─
# Empty string -> all classes allowed; a value -> only that class is targeted.
_target_class      = ""            # current target class
_target_class_lock = threading.Lock()

# ── Camera-to-Robot Arm Coordinate Transformation Offsets ─────────────────
# Camera Frame:     X = Right, Y = Down,  Z = Forward (Depth)
# Robot Arm Frame:  x = Left(-)/Right(+), y = Forward/Backward, z = Height
#
# 3-Point Calibration Coefficients (Linear Regression)
#   arm_x = Ax * X_cam + Bx * Z_cam + Cx
#   arm_y = Ay * X_cam + By * Z_cam + Cy
Ax =  1.445;  Bx = -0.350;  Cx =  0.035
Ay =  0.015;  By = -0.316;  Cy =  0.299

ARM_Z_FIXED = -0.015   # Fixed picking height (m) — target object height adjust to match the target object height

# ── Initialize ────────────────────────────────────────────────────────────
# Keep CPU-side libs single-threaded; heavy work runs on CUDA.
cv2.setNumThreads(1)
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

bridge    = CvBridge()
model     = YOLO(MODEL_PATH, task="detect")

# Camera intrinsics (Updated from camera_info callback)
fx = fy = cx = cy = 0.0
intrinsic_ready = False


# ════════════════════════════════════════════════════════════════════════════
# ── Callbacks ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def camera_info_cb(msg):
    global fx, fy, cx, cy, intrinsic_ready
    if not intrinsic_ready:
        fx = msg.K[0]
        fy = msg.K[4]
        cx = msg.K[2]
        cy = msg.K[5]
        intrinsic_ready = True
        rospy.loginfo(f"Camera intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")


def class_cb(msg):
    """
    Receives /yolo/class or /target_item (std_msgs/String).
    Updates the target class at runtime.
    Empty string -> allow all classes.
    """
    global _target_class
    raw_class = msg.data.strip()
    new_class = raw_class
    c = raw_class.lower()
    if c:
        if any(x in c for x in ["burger", "hamburger", "fries", "fast food", "food"]):
            new_class = "hamburger"
        elif any(x in c for x in ["med", "pharm", "pill", "sick", "drug", "first aid", "first-aid", "aspirin", "band-aid", "kit"]):
            new_class = "drug"
        elif any(x in c for x in ["coffee", "drink", "tea", "latte", "cappuccino", "espresso"]):
            new_class = "iceCoffee"
        elif any(x in c for x in ["mug", "cup"]):
            new_class = "mug"
            
    with _target_class_lock:
        _target_class = new_class
    if raw_class:
        rospy.loginfo(f"[yolo] Target class set: '{new_class}' (raw input: '{raw_class}')")
    else:
        rospy.loginfo("[yolo] Target class cleared (all classes allowed)")


# ════════════════════════════════════════════════════════════════════════════
# ── Coordinate Helpers ───────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def pixel_to_camera_frame(u, v, d):
    """
    Pixel coordinates + depth -> camera coordinates (m).
    Camera Frame: X=Right, Y=Down, Z=Forward(Depth)
    """
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d
    return X, Y, Z


def camera_to_robot_frame(X_cam, Y_cam, Z_cam):
    """
    Camera coordinates -> robot arm coordinates (linear calibration).
    """
    arm_x = Ax * X_cam + Bx * Z_cam + Cx
    arm_y = Ay * X_cam + By * Z_cam + Cy
    arm_z = ARM_Z_FIXED
    return arm_x, arm_y, arm_z


def get_robust_depth(depth_img, u, v, radius=DEPTH_RADIUS):
    """
    Return the median depth from an NxN region around the bbox center.
    Exclude invalid pixels (0).
    """
    h, w = depth_img.shape
    u0, u1 = max(0, u - radius), min(w, u + radius + 1)
    v0, v1 = max(0, v - radius), min(h, v + radius + 1)
    patch = depth_img[v0:v1, u0:u1].astype(np.float32)
    valid = patch[patch > 0]
    if len(valid) < 3:
        return 0.0
    return float(np.median(valid)) * DEPTH_SCALE


# ════════════════════════════════════════════════════════════════════════════
# ── Main Sync Callback ───────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def sync_callback(color_msg, depth_msg):
    global _last_inference_time
    # ── Activation gate: do nothing until the orchestrator switches to grab ──
    # Returns BEFORE any cv_bridge conversion or inference, so an inactive node
    # costs almost no CPU/GPU and never drives the arm.
    with _active_lock:
        if not _active:
            return

    # ── Rate Limiter ──
    # Limit inference rate to prevent overloading the system
    now = rospy.Time.now().to_sec()
    if now - _last_inference_time < _inference_interval:
        return
    _last_inference_time = now

    if not intrinsic_ready:
        rospy.logwarn_throttle(5, "Camera intrinsics not yet received")
        return

    # Image conversion
    frame = bridge.imgmsg_to_cv2(color_msg, "bgr8")
    depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

    # Read current target class (thread-safe)
    with _target_class_lock:
        current_target = _target_class

    # YOLO inference
    results = model(frame, conf=CONF_THRESH, device=0, verbose=False)
    boxes   = results[0].boxes

    # ── Process detection results ────────────────────────────────────────────────────
    best_box  = None
    best_conf = 0.0
    vis_labels = []   # [(x1,y1,x2,y2, label_str, color)]

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        u        = (x1 + x2) // 2
        v        = (y1 + y2) // 2
        cls_name = model.names[int(box.cls)]
        conf     = float(box.conf)

        # ── Class filter ──────────────────────────────────────────────────
        # If current_target is empty, allow all classes
        # If a value is specified, only that class is considered a target candidate
        is_target = (not current_target) or (cls_name == current_target)

        if not is_target:
            # Non-target objects are visualized in gray only (excluded from target selection)
            vis_labels.append((
                x1, y1, x2, y2,
                f"[skip] {cls_name} {conf:.2f}",
                (100, 100, 100)   # Gray
            ))
            continue

        # ── Depth validity check ─────────────────────────────────────────────
        d = get_robust_depth(depth, u, v)
        if not (MIN_DEPTH < d < MAX_DEPTH):
            rospy.logwarn_throttle(2,
                f"[{cls_name}] Depth out of range: {d:.3f}m")
            vis_labels.append((
                x1, y1, x2, y2,
                f"N/A ({d:.2f}m)",
                (128, 128, 128)   # Gray
            ))
            continue

        # ── Coordinate transformation ────────────────────────────────────────────────────
        X_cam, Y_cam, Z_cam = pixel_to_camera_frame(u, v, d)
        arm_x, arm_y, arm_z = camera_to_robot_frame(X_cam, Y_cam, Z_cam)

        rospy.loginfo(
            f"[{cls_name} {conf:.2f}]  "
            f"pixel=({u},{v})  dist={d:.3f}m  "
            f"arm=({arm_x:.3f},{arm_y:.3f},{arm_z:.3f})"
        )

        vis_labels.append((
            x1, y1, x2, y2,
            f"{cls_name} {conf:.2f}  {d:.2f}m  ({arm_x:+.2f},{arm_y:+.2f})",
            (0, 255, 255)   # Yellow tone (valid target)
        ))

        # ── Best selection: highest confidence ───────────────────────────────────
        # (Select the most confident object as the target, not the closest one)
        if conf > best_conf:
            best_conf = conf
            best_box  = {
                'cls':    cls_name,
                'conf':   conf,
                'dist':   d,
                'X_cam':  X_cam, 'Y_cam': Y_cam, 'Z_cam': Z_cam,
                'arm_x':  arm_x, 'arm_y': arm_y, 'arm_z': arm_z,
                'header': color_msg.header,
            }

    # ── Publish the best detection result in arm coordinates (consumed by pick_arm_node) ─────────────
    if best_box is not None:
        cam_pt         = PointStamped()
        cam_pt.header  = best_box['header']
        cam_pt.point.x = best_box['X_cam']
        cam_pt.point.y = best_box['Y_cam']
        cam_pt.point.z = best_box['Z_cam']
        obj_pub.publish(cam_pt)

        arm_pt   = Point()
        arm_pt.x = best_box['arm_x']
        arm_pt.y = best_box['arm_y']
        arm_pt.z = best_box['arm_z']
        arm_pub.publish(arm_pt)   # /yolo/arm_point → pick_arm_node

        rospy.loginfo_throttle(1.0,
            f"[yolo] BEST → [{best_box['cls']} conf={best_box['conf']:.2f}]  "
            f"dist={best_box['dist']:.3f}m  "
            f"arm=({best_box['arm_x']:.3f},{best_box['arm_y']:.3f})")

    # ── Publish visualization image ────────────────────────────────────────────────
    annotated = results[0].plot()
    for (x1, y1, x2, y2, label, color) in vis_labels:
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)[0]
        tx, ty = x1, min(y2 + th + 6, annotated.shape[0] - 4)
        cv2.rectangle(annotated,
                      (tx - 2, ty - th - 4), (tx + tw + 2, ty + 2),
                      (0, 0, 0), -1)   # Black background
        cv2.putText(annotated, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

    # Highlight the best box separately (green border)
    if best_box is not None:
        bx = vis_labels[[l[4] for l in vis_labels].index(
            next(l[4] for l in vis_labels
                 if best_box['cls'] in l[4] and f"{best_box['conf']:.2f}" in l[4])
        )]
        cv2.rectangle(annotated, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 3)

    vis_pub.publish(bridge.cv2_to_imgmsg(annotated, "bgr8"))


# ════════════════════════════════════════════════════════════════════════════
# ── Main ─────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    rospy.init_node('yolo_detector', anonymous=True)

    # ── Publishers ────────────────────────────────────────────────────────
    vis_pub = rospy.Publisher('/yolo/image_result', Image,        queue_size=1)
    obj_pub = rospy.Publisher('/yolo/object_point', PointStamped, queue_size=1)
    arm_pub = rospy.Publisher('/yolo/arm_point',    Point,        queue_size=1)

    # ── Subscribers ───────────────────────────────────────────────────────
    # Camera intrinsics
    rospy.Subscriber('/camera/color/camera_info', CameraInfo, camera_info_cb)

    # Target class — set at runtime by the orchestrator on /target_item.
    # (For standalone YOLO testing, publish to /yolo/class instead.)
    rospy.Subscriber('/target_item', String, class_cb)
    #rospy.Subscriber('/yolo/class', String, class_cb)

    # Activation gate (controlled by the orchestrator). Starts inactive.
    rospy.Subscriber('/yolo_grab/activate', Bool, activate_cb)

    # Subscribe to synchronized RGB + Depth streams
    color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    depth_sub = message_filters.Subscriber(
        '/camera/aligned_depth_to_color/image_raw', Image)
    sync = message_filters.ApproximateTimeSynchronizer(
        [color_sub, depth_sub], queue_size=5, slop=0.05)
    sync.registerCallback(sync_callback)

    rospy.loginfo("YOLO detector ready.")
    rospy.loginfo("  Subscribe: /camera/color/image_raw  /camera/aligned_depth_to_color/image_raw")
    rospy.loginfo("  Subscribe: /yolo/class  (std_msgs/String — 타겟 클래스 실시간 지정)")
    rospy.loginfo("  Publish: /yolo/image_result  /yolo/object_point  /yolo/arm_point")
    rospy.loginfo("  Selection criterion: highest-confidence object within the target class")
    rospy.loginfo("")
    rospy.loginfo("  타겟 클래스 지정 Examples:")
    rospy.loginfo("    rostopic pub /yolo/class std_msgs/String \"data: 'hamburger'\"")
    rospy.loginfo("    rostopic pub /yolo/class std_msgs/String \"data: ''\"  (allow all classes)")
    rospy.spin()