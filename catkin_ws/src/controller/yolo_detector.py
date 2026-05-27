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
  -> Publish /pick_target (geometry_msgs/Point)

Published Topics:
  - /yolo/image_result    : Annotated image (for visualization)
  - /yolo/object_point    : 3D coordinates of the detected object in the Camera frame
  - /pick_target          : 3D coordinates in the Robot Arm frame (Input for auto_pick.py)
"""

import rospy
import numpy as np
import cv2
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO

# ── 파라미터 ─────────────────────────────────────────────────────────
MODEL_PATH   = "./best.engine"
CONF_THRESH  = 0.8
DEPTH_SCALE  = 0.001      # RealSense: mm → m
DEPTH_RADIUS = 3          # bbox 중심 주변 NxN 샘플링 반경 (노이즈 제거)
MAX_DEPTH    = 1.5        # 이 거리(m) 이상은 무시
MIN_DEPTH    = 0.05       # 이 거리(m) 이하는 무시

# ── Parameters ────────────────────────────────────────────────────────
MODEL_PATH   = "./best.engine"
CONF_THRESH  = 0.8
DEPTH_SCALE  = 0.001      # RealSense: mm -> meters
DEPTH_RADIUS = 3          # NxN sampling radius around bbox center (for noise reduction)
MAX_DEPTH    = 1.5        # Ignore detections beyond this distance (meters)
MIN_DEPTH    = 0.05       # Ignore detections closer than this distance (meters)

# ── Camera-to-Robot Arm Coordinate Transformation Offsets ─────────────
# Assumption: The camera is securely mounted, looking down at the workspace in front of the robot.
# Please tune these empirical values based on actual physical measurements.
#
# Camera Frame conventions:   X = Right, Y = Down,  Z = Forward (Depth)
# Robot Arm Frame conventions: x = Left(-)/Right(+), y = Forward/Backward, z = Up(+)/Down(-) Height
#
# Conceptual Transformation:
#   arm_y =  cam_Z (Depth correlates to robot's forward distance)
#   arm_x = -cam_X (Inverted left/right axis)
#   arm_z =  CAM_HEIGHT - cam_Y * sin(tilt) (Camera mounting height minus vertical component)
#
# 3-Point Calibration Coefficients (Linear Regression Model)
#   arm_x = Ax * X_cam + Bx * Z_cam + Cx
#   arm_y = Ay * X_cam + By * Z_cam + Cy
Ax =  1.445;  Bx = -0.350;  Cx =  0.035
Ay =  0.015;  By = -0.316;  Cy =  0.299

ARM_Z_FIXED = -0.015  # Fixed picking height (meters) - Adjust this based on target object height

# ── Initialize ───────────────────────────────────────────────────────────
bridge    = CvBridge()
model     = YOLO(MODEL_PATH, task="detect")

# camera intrinsic (Updated from camera_info callback)
fx = fy = cx = cy = 0.0
intrinsic_ready = False


def camera_info_cb(msg):
    global fx, fy, cx, cy, intrinsic_ready
    if not intrinsic_ready:
        fx = msg.K[0]
        fy = msg.K[4]
        cx = msg.K[2]
        cy = msg.K[5]
        intrinsic_ready = True
        rospy.loginfo(f"Camera intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")


def pixel_to_camera_frame(u, v, d):
    """
    Convert pixel coordinates and depth to 3D camera coordinates (meters).
    Camera Frame Convention:
      X = Right, Y = Down, Z = Forward (Depth)
    """
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d
    return X, Y, Z


def camera_to_robot_frame(X_cam, Y_cam, Z_cam):
    """
    Linear transformation from Camera Frame to Robot Arm Frame.
    Uses empirical coefficients derived from a 3-point calibration.
    """
    arm_x = Ax * X_cam + Bx * Z_cam + Cx
    arm_y = Ay * X_cam + By * Z_cam + Cy
    arm_z = ARM_Z_FIXED
    return arm_x, arm_y, arm_z


def get_robust_depth(depth_img, u, v, radius=DEPTH_RADIUS):
    """
    Return the median depth value within an NxN region around the bbox center.
    Excludes invalid pixels (depth value of 0).
    """
    h, w = depth_img.shape
    u0, u1 = max(0, u - radius), min(w, u + radius + 1)
    v0, v1 = max(0, v - radius), min(h, v + radius + 1)
    patch = depth_img[v0:v1, u0:u1].astype(np.float32)
    valid = patch[patch > 0]
    if len(valid) < 3:
        return 0.0
    return float(np.median(valid)) * DEPTH_SCALE


# ── main callback ────────────────────────────────────────────────────────
def sync_callback(color_msg, depth_msg):
    if not intrinsic_ready:
        rospy.logwarn_throttle(5, "Camera intrinsics not yet received")
        return

    # transform image
    frame = bridge.imgmsg_to_cv2(color_msg, "bgr8")
    depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

    # YOLO inference
    results = model(frame, conf=CONF_THRESH, device=0, verbose=False)
    boxes   = results[0].boxes

    # ── process detection results ────────────────────────────────────────
    best_box   = None
    best_conf  = 0.0
    vis_labels = []   # [(x1,y1,x2,y2, label_str, color)]

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        u = (x1 + x2) // 2
        v = (y1 + y2) // 2
        cls_name = model.names[int(box.cls)]
        conf     = float(box.conf)

        d = get_robust_depth(depth, u, v)

        if MIN_DEPTH < d < MAX_DEPTH:
            X_cam, Y_cam, Z_cam = pixel_to_camera_frame(u, v, d)
            arm_x, arm_y, arm_z = camera_to_robot_frame(X_cam, Y_cam, Z_cam)

            # print detection log result
            rospy.loginfo(
                f"[{cls_name} {conf:.2f}]  "
                f"pixel=({u},{v})  dist={d:.3f}m  "
                f"arm=({arm_x:.3f},{arm_y:.3f},{arm_z:.3f})"
            )

            # 시각화용 라벨 수집 (유효 depth)
            vis_labels.append((
                x1, y1, x2, y2,
                f"{d:.2f}m  ({arm_x:+.2f},{arm_y:+.2f})",
                (0, 255, 255)   # 노란색 계열
            ))

            # pick_target candidate: closest object
            if best_box is None or d < best_box['dist']:
                best_box = {
                    'cls': cls_name, 'conf': conf, 'dist': d,
                    'X_cam': X_cam, 'Y_cam': Y_cam, 'Z_cam': Z_cam,
                    'arm_x': arm_x, 'arm_y': arm_y, 'arm_z': arm_z,
                    'header': color_msg.header,
                }
        else:
            rospy.logwarn_throttle(2,
                f"[{cls_name}] depth out of range: {d:.3f}m")
            vis_labels.append((
                x1, y1, x2, y2,
                f"N/A ({d:.2f}m)",
                (128, 128, 128)  # 회색 (range 밖)
            ))

    # ── Publish the detected object coordinates in Arm frame (Consumed by pick_controller) ──
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
        arm_pub.publish(arm_pt)   # /yolo/arm_point → pick_controller pub

    # annotated image + depth overlay publish (visualization)
    annotated = results[0].plot()
    for (x1, y1, x2, y2, label, color) in vis_labels:
        # bbox 하단에 거리 + arm 좌표 표시
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        tx, ty = x1, min(y2 + th + 6, annotated.shape[0] - 4)
        cv2.rectangle(annotated, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 2),
                      (0, 0, 0), -1)   # 배경 검정
        cv2.putText(annotated, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    vis_pub.publish(bridge.cv2_to_imgmsg(annotated, "bgr8"))


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rospy.init_node('yolo_detector', anonymous=True)

    # Publishers
    vis_pub  = rospy.Publisher('/yolo/image_result', Image,        queue_size=1)
    obj_pub  = rospy.Publisher('/yolo/object_point', PointStamped, queue_size=1)
    arm_pub  = rospy.Publisher('/yolo/arm_point',    Point,        queue_size=1)

    # Camera intrinsics
    rospy.Subscriber('/camera/color/camera_info', CameraInfo, camera_info_cb)

    # subscribe synchronized RGB + Depth
    color_sub = message_filters.Subscriber(
        '/camera/color/image_raw', Image)
    depth_sub = message_filters.Subscriber(
        '/camera/aligned_depth_to_color/image_raw', Image)

    sync = message_filters.ApproximateTimeSynchronizer(
        [color_sub, depth_sub], queue_size=5, slop=0.05)
    sync.registerCallback(sync_callback)

    rospy.loginfo("YOLO detector ready.")
    rospy.loginfo("Subscribing to /camera/color/image_raw + aligned depth")
    rospy.loginfo("Publishing  → /yolo/image_result  /yolo/object_point  /yolo/arm_point")
    rospy.spin()