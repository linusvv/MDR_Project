#!/usr/bin/python3
# coding=utf8
"""
YOLO + RealSense Depth Integrated Detection Node

Pipeline:
  1. /camera/color/image_raw                   -> Run YOLO inference
  2. /camera/aligned_depth_to_color/image_raw  -> Get depth at the bbox center
  3. /camera/color/camera_info                 -> Camera intrinsics (fx, fy, cx, cy)

  -> Compute 3D camera coordinates (X_cam, Y_cam, Z_cam)
  -> Apply calibration offset to transform into Robot Arm coordinates
  -> Publish /yolo/arm_point (geometry_msgs/Point)   -> consumed by pick_arm_node

Published Topics:
  - /yolo/image_result  : Annotated image (for visualization)
  - /yolo/object_point  : 3D coordinates in the Camera frame (PointStamped)
  - /yolo/arm_point     : 3D coordinates in the Robot Arm frame (Point)

Subscribed Topics:
  - /target_item (std_msgs/String) : target class name, set at runtime
      - empty string ("") -> pick the highest-confidence object of ANY class
      - class name        -> pick the highest-confidence object of that class
      Examples:
        rostopic pub /target_item std_msgs/String "data: 'hamburger'"
        rostopic pub /target_item std_msgs/String "data: ''"

CPU NOTE:
  The camera publishes at ~30 Hz, but running YOLO + depth lookup + drawing the
  annotated image on every frame is expensive on a Jetson Orin Nano. This node
  therefore RATE-LIMITS processing to ~TARGET_FPS (default 5 Hz, see the
  ~target_fps param) and skips building the visualization image when no one is
  subscribed to /yolo/image_result. Both are tunable to trade CPU for latency.
"""

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
MODEL_PATH   = "./best.engine"
CONF_THRESH  = 0.8
DEPTH_SCALE  = 0.001      # RealSense: mm -> meters
DEPTH_RADIUS = 3          # NxN sampling radius around bbox center (noise reduction)
MAX_DEPTH    = 1.5        # Ignore detections beyond this distance (meters)
MIN_DEPTH    = 0.05       # Ignore detections closer than this distance (meters)

# ── Rate limiting (CPU control) ───────────────────────────────────────────
# Maximum number of frames processed per second. The camera runs much faster
# than this; frames arriving inside one period are dropped before any heavy
# work (cv_bridge conversion, inference, drawing). Overridden by ~target_fps.
TARGET_FPS   = 5.0
# Build & publish the annotated /yolo/image_result image. Drawing it
# (results.plot() + per-box text) is one of the biggest CPU costs here, so it
# can be disabled entirely. Overridden by ~enable_viz.
ENABLE_VIZ   = True

# Internal rate-limit state (set up in __main__ from the params above)
_proc_period = 1.0 / TARGET_FPS   # minimum seconds between processed frames
_last_proc_t = None               # rospy.Time of the last processed frame

# ── Activation Gate ──────────────────────────────────────────────────────
# Controlled via /yolo_grab/activate (Bool). Starts INACTIVE.
_active      = False
_active_lock = threading.Lock()

def activate_cb(msg):
    global _active
    with _active_lock:
        _active = msg.data
    rospy.loginfo(f"[yolo_detector] {'ACTIVATED' if msg.data else 'DEACTIVATED'}")

# ── Target Class (set at runtime via the /target_item topic) ──────────────
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

ARM_Z_FIXED = -0.015   # Fixed picking height (m) — tune to the target object height

# ── Initialize ────────────────────────────────────────────────────────────
bridge    = CvBridge()
model     = YOLO(MODEL_PATH, task="detect")

# Camera intrinsics (updated from the camera_info callback)
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
    Receive /target_item (std_msgs/String) and update the target class at
    runtime. An empty string allows all classes.
    """
    global _target_class
    new_class = msg.data.strip()
    with _target_class_lock:
        _target_class = new_class
    if new_class:
        rospy.loginfo(f"[yolo] Target class set: '{new_class}'")
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
    Return the median depth of the NxN region around the bbox center,
    ignoring invalid (0) pixels.
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
    global _last_proc_t

    # ── Activation gate: skip everything while deactivated ────────────────
    with _active_lock:
        if not _active:
            return

    # ── Rate limit: drop frames that arrive faster than TARGET_FPS ────────
    # This check runs BEFORE any cv_bridge conversion or inference, so dropped
    # frames cost almost nothing. This is the main CPU lever.
    now = rospy.Time.now()
    if _last_proc_t is not None and (now - _last_proc_t).to_sec() < _proc_period:
        return
    _last_proc_t = now

    if not intrinsic_ready:
        rospy.logwarn_throttle(5, "Camera intrinsics not yet received")
        return

    # Decide up front whether to spend CPU on the visualization image. When
    # nobody subscribes to /yolo/image_result, we skip plot()/drawing/publish.
    want_viz = ENABLE_VIZ and (vis_pub.get_num_connections() > 0)

    # Convert images
    frame = bridge.imgmsg_to_cv2(color_msg, "bgr8")
    depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

    # Read the current target class (thread-safe)
    with _target_class_lock:
        current_target = _target_class

    # YOLO inference
    results = model(frame, conf=CONF_THRESH, device='cuda', verbose=False)
    boxes   = results[0].boxes

    # ── Process detections ────────────────────────────────────────────────
    best_box  = None
    best_conf = 0.0
    vis_labels = []   # [(x1,y1,x2,y2, label_str, color)] — only filled when want_viz

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        u        = (x1 + x2) // 2
        v        = (y1 + y2) // 2
        cls_name = model.names[int(box.cls)]
        conf     = float(box.conf)

        # ── Class filter ─────────────────────────────────────────────────
        # Empty target -> all classes allowed; otherwise only the target class
        # is considered as a pick candidate.
        is_target = (not current_target) or (cls_name == current_target)

        if not is_target:
            # Non-target objects are only drawn (grey), never selected.
            if want_viz:
                vis_labels.append((
                    x1, y1, x2, y2,
                    f"[skip] {cls_name} {conf:.2f}",
                    (100, 100, 100)   # grey
                ))
            continue

        # ── Depth validity check ─────────────────────────────────────────
        d = get_robust_depth(depth, u, v)
        if not (MIN_DEPTH < d < MAX_DEPTH):
            rospy.logwarn_throttle(2,
                f"[{cls_name}] depth out of range: {d:.3f}m")
            if want_viz:
                vis_labels.append((
                    x1, y1, x2, y2,
                    f"N/A ({d:.2f}m)",
                    (128, 128, 128)   # grey
                ))
            continue

        # ── Coordinate transform ─────────────────────────────────────────
        X_cam, Y_cam, Z_cam = pixel_to_camera_frame(u, v, d)
        arm_x, arm_y, arm_z = camera_to_robot_frame(X_cam, Y_cam, Z_cam)

        rospy.loginfo_throttle(0.5,
            f"[{cls_name} {conf:.2f}]  "
            f"pixel=({u},{v})  dist={d:.3f}m  "
            f"arm=({arm_x:.3f},{arm_y:.3f},{arm_z:.3f})"
        )

        if want_viz:
            vis_labels.append((
                x1, y1, x2, y2,
                f"{cls_name} {conf:.2f}  {d:.2f}m  ({arm_x:+.2f},{arm_y:+.2f})",
                (0, 255, 255)   # yellow (valid target)
            ))

        # ── Best selection: highest confidence ───────────────────────────
        # (Selecting the most confident object, not the closest one.)
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

    # ── Publish the best detection in Arm coordinates (pick_arm_node) ──────
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
        arm_pub.publish(arm_pt)   # /yolo/arm_point -> pick_arm_node

        rospy.loginfo_throttle(1.0,
            f"[yolo] BEST -> [{best_box['cls']} conf={best_box['conf']:.2f}]  "
            f"dist={best_box['dist']:.3f}m  "
            f"arm=({best_box['arm_x']:.3f},{best_box['arm_y']:.3f})")

    # ── Publish the visualization image (skipped entirely when not wanted) ─
    if not want_viz:
        return

    annotated = results[0].plot()
    for (x1, y1, x2, y2, label, color) in vis_labels:
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)[0]
        tx, ty = x1, min(y2 + th + 6, annotated.shape[0] - 4)
        cv2.rectangle(annotated,
                      (tx - 2, ty - th - 4), (tx + tw + 2, ty + 2),
                      (0, 0, 0), -1)   # black background
        cv2.putText(annotated, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

    # Outline the best box (green border)
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

    # ── Load CPU-control params ───────────────────────────────────────────
    TARGET_FPS = float(rospy.get_param('~target_fps', TARGET_FPS))
    ENABLE_VIZ = bool(rospy.get_param('~enable_viz', ENABLE_VIZ))
    _proc_period = (1.0 / TARGET_FPS) if TARGET_FPS > 0 else 0.0
    rospy.loginfo(f"[yolo_detector] Rate limit: {TARGET_FPS:.1f} FPS "
                  f"(period {_proc_period*1000:.0f} ms), viz={'on' if ENABLE_VIZ else 'off'}")

    # ── Publishers ────────────────────────────────────────────────────────
    vis_pub = rospy.Publisher('/yolo/image_result', Image,        queue_size=1)
    obj_pub = rospy.Publisher('/yolo/object_point', PointStamped, queue_size=1)
    arm_pub = rospy.Publisher('/yolo/arm_point',    Point,        queue_size=1)

    # ── Subscribers ───────────────────────────────────────────────────────
    # Camera intrinsics
    rospy.Subscriber('/camera/color/camera_info', CameraInfo, camera_info_cb)

    # Target class, set at runtime
    # (for standalone YOLO testing you can publish to /yolo/class instead)
    #rospy.Subscriber('/yolo/class', String, class_cb)
    rospy.Subscriber('/target_item', String, class_cb)

    # Activation gate (controlled by the orchestrator)
    rospy.Subscriber('/yolo_grab/activate', Bool, activate_cb)

    # ── GPU warm-up ──────────────────────────────────────────────────────────
    # The model is already resident on the GPU (loaded at import). Run ONE dummy
    # inference now so the CUDA context, cuDNN/TensorRT kernels and workspace are
    # built up-front — otherwise the first real frame after activation stalls for
    # seconds. After this the node sits IDLE: sync_callback returns at the
    # activation gate (before any cv_bridge conversion or inference) while
    # _active is False, so it consumes effectively no CPU/GPU until the
    # orchestrator flips /yolo_grab/activate to True.
    # try:
    #     dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    #     model(dummy, conf=CONF_THRESH, device='cuda', verbose=False)
    #     rospy.loginfo("[yolo_detector] GPU warm-up complete — model resident on "
    #                   "GPU, node IDLE until /yolo_grab/activate is True.")
    # except Exception as e:
    #     rospy.logwarn(f"[yolo_detector] GPU warm-up failed (continuing): {e}")

    # RGB + Depth synchronized subscription
    color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    depth_sub = message_filters.Subscriber(
        '/camera/aligned_depth_to_color/image_raw', Image)
    sync = message_filters.ApproximateTimeSynchronizer(
        [color_sub, depth_sub], queue_size=5, slop=0.05)
    sync.registerCallback(sync_callback)

    rospy.loginfo("YOLO detector ready.")
    rospy.loginfo("  Subscribing: /camera/color/image_raw  /camera/aligned_depth_to_color/image_raw")
    rospy.loginfo("  Subscribing: /target_item  (std_msgs/String — runtime target class)")
    rospy.loginfo("  Publishing : /yolo/image_result  /yolo/object_point  /yolo/arm_point")
    rospy.loginfo("  Selection  : highest-confidence object within the target class")
    rospy.loginfo("")
    rospy.loginfo("  Set target class example:")
    rospy.loginfo("    rostopic pub /target_item std_msgs/String \"data: 'hamburger'\"")
    rospy.loginfo("    rostopic pub /target_item std_msgs/String \"data: ''\"  (allow all)")
    rospy.spin()