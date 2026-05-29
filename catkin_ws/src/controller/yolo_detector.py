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
  - /yolo/class (std_msgs/String) : 타겟 클래스 이름 실시간 지정
      - 빈 문자열("") → 모든 클래스 중 confidence 최대 물체 선택
      - 클래스 이름 지정 → 해당 클래스 중 confidence 최대 물체 선택
      예시:
        rostopic pub /yolo/class std_msgs/String "data: 'hamburger'"
        rostopic pub /yolo/class std_msgs/String "data: ''"
"""

import threading

import rospy
import numpy as np
import cv2
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO

# ── Parameters ────────────────────────────────────────────────────────────
MODEL_PATH   = "./best.engine"
CONF_THRESH  = 0.8
DEPTH_SCALE  = 0.001      # RealSense: mm -> meters
DEPTH_RADIUS = 3          # NxN sampling radius around bbox center (for noise reduction)
MAX_DEPTH    = 1.5        # Ignore detections beyond this distance (meters)
MIN_DEPTH    = 0.05       # Ignore detections closer than this distance (meters)

# ── Target Class (동적 설정 — /yolo/class 토픽으로 런타임 변경 가능) ────
# 빈 문자열이면 모든 클래스 허용, 값이 있으면 해당 클래스만 타겟
_target_class      = ""            # 현재 타겟 클래스
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

ARM_Z_FIXED = -0.015   # Fixed picking height (m) — target object height 에 맞게 조정

# ── Initialize ────────────────────────────────────────────────────────────
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
    /yolo/class (std_msgs/String) 수신.
    타겟 클래스를 런타임에 업데이트한다.
    빈 문자열 → 모든 클래스 허용.
    """
    global _target_class
    new_class = msg.data.strip()
    with _target_class_lock:
        _target_class = new_class
    if new_class:
        rospy.loginfo(f"[yolo] 타겟 클래스 설정: '{new_class}'")
    else:
        rospy.loginfo("[yolo] 타겟 클래스 해제 (모든 클래스 허용)")


# ════════════════════════════════════════════════════════════════════════════
# ── Coordinate Helpers ───────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def pixel_to_camera_frame(u, v, d):
    """
    픽셀 좌표 + 깊이 → 카메라 좌표 (m).
    Camera Frame: X=Right, Y=Down, Z=Forward(Depth)
    """
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d
    return X, Y, Z


def camera_to_robot_frame(X_cam, Y_cam, Z_cam):
    """
    카메라 좌표 → 로봇 팔 좌표 (선형 캘리브레이션).
    """
    arm_x = Ax * X_cam + Bx * Z_cam + Cx
    arm_y = Ay * X_cam + By * Z_cam + Cy
    arm_z = ARM_Z_FIXED
    return arm_x, arm_y, arm_z


def get_robust_depth(depth_img, u, v, radius=DEPTH_RADIUS):
    """
    bbox 중심 주변 NxN 영역의 중앙값 깊이 반환.
    유효하지 않은 픽셀(0) 제외.
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
    if not intrinsic_ready:
        rospy.logwarn_throttle(5, "Camera intrinsics not yet received")
        return

    # 이미지 변환
    frame = bridge.imgmsg_to_cv2(color_msg, "bgr8")
    depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

    # 현재 타겟 클래스 읽기 (thread-safe)
    with _target_class_lock:
        current_target = _target_class

    # YOLO 추론
    results = model(frame, conf=CONF_THRESH, device=0, verbose=False)
    boxes   = results[0].boxes

    # ── 감지 결과 처리 ────────────────────────────────────────────────────
    best_box  = None
    best_conf = 0.0
    vis_labels = []   # [(x1,y1,x2,y2, label_str, color)]

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        u        = (x1 + x2) // 2
        v        = (y1 + y2) // 2
        cls_name = model.names[int(box.cls)]
        conf     = float(box.conf)

        # ── 클래스 필터 ──────────────────────────────────────────────────
        # current_target가 비어있으면 모든 클래스 허용
        # 값이 있으면 해당 클래스만 타겟 후보로 인정
        is_target = (not current_target) or (cls_name == current_target)

        if not is_target:
            # 타겟이 아닌 물체는 회색으로 시각화만 (타겟 선정 제외)
            vis_labels.append((
                x1, y1, x2, y2,
                f"[skip] {cls_name} {conf:.2f}",
                (100, 100, 100)   # 회색
            ))
            continue

        # ── 깊이 유효성 검사 ─────────────────────────────────────────────
        d = get_robust_depth(depth, u, v)
        if not (MIN_DEPTH < d < MAX_DEPTH):
            rospy.logwarn_throttle(2,
                f"[{cls_name}] depth 범위 밖: {d:.3f}m")
            vis_labels.append((
                x1, y1, x2, y2,
                f"N/A ({d:.2f}m)",
                (128, 128, 128)   # 회색
            ))
            continue

        # ── 좌표 변환 ────────────────────────────────────────────────────
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
            (0, 255, 255)   # 노란색 계열 (유효 타겟)
        ))

        # ── Best 선정: confidence 최대 ───────────────────────────────────
        # (거리 기준이 아닌 confidence 기준으로 가장 확실한 물체를 타겟)
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

    # ── Best 감지 결과를 Arm 좌표로 발행 (pick_arm_node 소비) ─────────────
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

    # ── 시각화 이미지 발행 ────────────────────────────────────────────────
    annotated = results[0].plot()
    for (x1, y1, x2, y2, label, color) in vis_labels:
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)[0]
        tx, ty = x1, min(y2 + th + 6, annotated.shape[0] - 4)
        cv2.rectangle(annotated,
                      (tx - 2, ty - th - 4), (tx + tw + 2, ty + 2),
                      (0, 0, 0), -1)   # 배경 검정
        cv2.putText(annotated, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

    # Best box에 별도 표시 (초록 테두리)
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

    # 타겟 클래스 동적 설정
    rospy.Subscriber('/yolo/class', String, class_cb)

    # RGB + Depth 동기화 구독
    color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    depth_sub = message_filters.Subscriber(
        '/camera/aligned_depth_to_color/image_raw', Image)
    sync = message_filters.ApproximateTimeSynchronizer(
        [color_sub, depth_sub], queue_size=5, slop=0.05)
    sync.registerCallback(sync_callback)

    rospy.loginfo("YOLO detector 준비 완료.")
    rospy.loginfo("  구독: /camera/color/image_raw  /camera/aligned_depth_to_color/image_raw")
    rospy.loginfo("  구독: /yolo/class  (std_msgs/String — 타겟 클래스 실시간 지정)")
    rospy.loginfo("  발행: /yolo/image_result  /yolo/object_point  /yolo/arm_point")
    rospy.loginfo("  선정 기준: 타겟 클래스 내 confidence 최대 물체")
    rospy.loginfo("")
    rospy.loginfo("  타겟 클래스 지정 예시:")
    rospy.loginfo("    rostopic pub /yolo/class std_msgs/String \"data: 'hamburger'\"")
    rospy.loginfo("    rostopic pub /yolo/class std_msgs/String \"data: ''\"  (전체 허용)")
    rospy.spin()