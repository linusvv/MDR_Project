#!/usr/bin/python3
# coding=utf8
"""
Pick Controller Node  (with Grasp Verification + Failure Recovery)
─────────────────────────────────────────────────────────────────────────────
State machine:

  IDLE
    ↓ (YOLO detects object)
  APPROACHING  → chassis forward / strafe until arm_x≈0, arm_y≈TARGET_Y
    ↓
  CREEPING     → slow forward for CREEP_TIME
    ↓
  ALIGNED      → send /pick_target once
    ↓
  PICKING      → wait PICK_WAIT seconds for auto_pick to finish
    ↓
  VERIFY_ROTATE → rotate chassis 90° (to bring gripper into camera view)
    ↓
  VERIFY_CHECK  → check RealSense depth at gripper ROI
    ├─ depth < GRASP_CHECK_DIST  →  SUCCESS
    │     ↓
    │   PLACE_ROTATE → rotate 90° more (total 180° from pick pose)
    │     ↓  IDLE (basket delivery done)
    │
    └─ depth >= GRASP_CHECK_DIST  →  FAIL
          ↓
        SCAN_L  → rotate left SCAN_TIME s while watching YOLO
        SCAN_R  → rotate right SCAN_TIME s while watching YOLO
          ├─ object found  → back to APPROACHING
          └─ not found     → IDLE (give up)

Chassis SetVelocity:
  velocity  : mm/s
  direction : 0~360° (90=forward, 270=back, 0=right, 180=left)
  angular   : rad/s  (positive = CCW / left turn)
"""

import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from chassis_control.msg import SetVelocity
from cv_bridge import CvBridge

# ── States ────────────────────────────────────────────────────────────────
IDLE          = 'IDLE'
APPROACHING   = 'APPROACHING'
CREEPING      = 'CREEPING'
ALIGNED       = 'ALIGNED'
PICKING       = 'PICKING'
VERIFY_ROTATE = 'VERIFY_ROTATE'   # rotate 90° after pick
VERIFY_CHECK  = 'VERIFY_CHECK'    # depth check: did we grab it?
PLACE_ROTATE  = 'PLACE_ROTATE'    # rotate 90° more → face basket (180° total)
SCAN_L        = 'SCAN_L'          # recovery: rotate left to find object
SCAN_R        = 'SCAN_R'          # recovery: rotate right to find object

# ── Parameters ────────────────────────────────────────────────────────────
TARGET_Y      = 0.250
TARGET_X      = 0.000
TOL_X         = 0.030
TOL_Y         = 0.020

FORWARD_SPEED = 80
STRAFE_SPEED  = 60
CREEP_SPEED   = 50
CREEP_TIME    = 0.6

VALID_Y_MIN   = -0.30
VALID_Y_MAX   =  0.60

DETECTION_TIMEOUT = 2.0
PICK_WAIT         = 12.0

# ── Grasp verification ────────────────────────────────────────────────────
ROTATE_SPEED      = 0.5            # rad/s (in-place rotation)
ROTATE_90_TIME    = 1.571 / ROTATE_SPEED   # π/2 / ω ≈ 3.14s  (tune if overshoot)

# Depth image ROI for grasp check [x1, y1, x2, y2] (640×480 기준)
# 로봇이 90° 회전 후 카메라 시야에 그리퍼가 들어오는 픽셀 영역
# 물리 배치에 따라 튜닝 필요
GRASP_ROI         = (260, 160, 380, 320)   # center region
GRASP_CHECK_DIST  = 0.15           # m — 이 거리보다 가까우면 물체 잡힌 것으로 판단
GRASP_DEPTH_SCALE = 0.001          # RealSense mm→m

# ── Scan recovery ─────────────────────────────────────────────────────────
SCAN_SPEED    = 0.4    # rad/s
SCAN_TIME     = 2.5    # seconds per side

# ── Globals ───────────────────────────────────────────────────────────────
state            = IDLE
last_detect_t    = None
pick_sent_t      = None
creep_start_t    = None
verify_start_t   = None
place_start_t    = None
scan_start_t     = None
last_arm_pt      = None
latest_depth_img = None   # RealSense depth image (for grasp check)

bridge = CvBridge()

# ── Publishers (init in main) ─────────────────────────────────────────────
vel_pub  = None
pick_pub = None


# ── Helpers ───────────────────────────────────────────────────────────────
def chassis_cmd(velocity=0.0, direction=90.0, angular=0.0):
    msg = SetVelocity()
    msg.velocity  = float(velocity)
    msg.direction = float(direction)
    msg.angular   = float(angular)
    vel_pub.publish(msg)

def chassis_stop():
    chassis_cmd(0, 90, 0)

def send_pick(pt):
    pick_pub.publish(pt)
    rospy.loginfo(f'[pick_ctrl] PICK SENT  arm=({pt.x:.3f},{pt.y:.3f},{pt.z:.3f})')

def check_grasp_depth():
    """
    RealSense depth ROI의 중앙값이 GRASP_CHECK_DIST 미만이면 True (잡힘).
    latest_depth_img 없으면 False 반환.
    """
    if latest_depth_img is None:
        rospy.logwarn('[pick_ctrl] No depth image for grasp check')
        return False

    x1, y1, x2, y2 = GRASP_ROI
    h, w = latest_depth_img.shape
    x1, x2 = max(0,x1), min(w,x2)
    y1, y2 = max(0,y1), min(h,y2)

    patch = latest_depth_img[y1:y2, x1:x2].astype(np.float32)
    valid = patch[patch > 0] * GRASP_DEPTH_SCALE   # mm → m
    if len(valid) < 5:
        rospy.logwarn('[pick_ctrl] Grasp ROI has too few valid depth pixels')
        return False

    median_d = float(np.median(valid))
    rospy.loginfo(f'[pick_ctrl] Grasp check: median depth in ROI = {median_d:.3f}m '
                  f'(threshold {GRASP_CHECK_DIST:.2f}m)')
    return median_d < GRASP_CHECK_DIST


# ── Callbacks ─────────────────────────────────────────────────────────────
def arm_point_cb(msg):
    global last_detect_t, last_arm_pt
    if not (VALID_Y_MIN < msg.y < VALID_Y_MAX):
        return
    last_arm_pt   = msg
    last_detect_t = rospy.Time.now()

def depth_cb(msg):
    global latest_depth_img
    try:
        latest_depth_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    except Exception as e:
        rospy.logwarn_throttle(5, f'[pick_ctrl] depth_cb error: {e}')


# ── Main control loop (10 Hz) ─────────────────────────────────────────────
def control_loop(event):
    global state, pick_sent_t, creep_start_t
    global verify_start_t, place_start_t, scan_start_t

    now = rospy.Time.now()

    # detection timeout guard
    if last_detect_t is None:
        if state == APPROACHING:
            rospy.logwarn_throttle(3, '[pick_ctrl] No detection — stopping')
            chassis_stop()
            state = IDLE
        return

    dt = (now - last_detect_t).to_sec()

    # ── IDLE ──────────────────────────────────────────────────────────────
    if state == IDLE:
        if dt < DETECTION_TIMEOUT:
            rospy.loginfo('[pick_ctrl] Object detected → APPROACHING')
            state = APPROACHING

    # ── APPROACHING ───────────────────────────────────────────────────────
    elif state == APPROACHING:
        if dt >= DETECTION_TIMEOUT:
            rospy.logwarn('[pick_ctrl] Detection lost → IDLE')
            chassis_stop()
            state = IDLE
            return

        pt    = last_arm_pt
        err_x = pt.x - TARGET_X
        err_y = pt.y - TARGET_Y

        aligned_x = abs(err_x) < TOL_X
        aligned_y = abs(err_y) < TOL_Y

        rospy.loginfo_throttle(0.5,
            f'[pick_ctrl] APPROACHING  arm=({pt.x:.3f},{pt.y:.3f})  '
            f'err=({err_x:+.3f},{err_y:+.3f})  aligned=({aligned_x},{aligned_y})')

        if aligned_x and aligned_y:
            rospy.loginfo(f'[pick_ctrl] ALIGNED → CREEPING')
            chassis_cmd(CREEP_SPEED, 90, 0)
            state = CREEPING
        elif not aligned_y:
            direction = 270 if err_y > 0 else 90
            chassis_cmd(FORWARD_SPEED, direction, 0)
        else:
            direction = 0 if err_x > 0 else 180
            chassis_cmd(STRAFE_SPEED, direction, 0)

    # ── CREEPING ──────────────────────────────────────────────────────────
    elif state == CREEPING:
        if creep_start_t is None:
            creep_start_t = now
        if (now - creep_start_t).to_sec() >= CREEP_TIME:
            chassis_stop()
            rospy.loginfo('[pick_ctrl] Creep done → ALIGNED')
            state = ALIGNED

    # ── ALIGNED ───────────────────────────────────────────────────────────
    elif state == ALIGNED:
        if last_arm_pt is not None:
            send_pick(last_arm_pt)
        pick_sent_t   = now
        creep_start_t = None
        state = PICKING

    # ── PICKING ───────────────────────────────────────────────────────────
    elif state == PICKING:
        elapsed = (now - pick_sent_t).to_sec() if pick_sent_t else PICK_WAIT
        rospy.loginfo_throttle(2,
            f'[pick_ctrl] PICKING … ({elapsed:.1f}/{PICK_WAIT:.0f}s)')
        if elapsed >= PICK_WAIT:
            chassis_stop()
            rospy.loginfo('[pick_ctrl] Pick wait done → VERIFY_ROTATE')
            verify_start_t = now
            # 90° 회전 시작 (CCW: 양수 angular)
            chassis_cmd(0, 90, ROTATE_SPEED)
            state = VERIFY_ROTATE

    # ── VERIFY_ROTATE ─────────────────────────────────────────────────────
    elif state == VERIFY_ROTATE:
        if verify_start_t is None:
            verify_start_t = now
            chassis_cmd(0, 90, ROTATE_SPEED)

        elapsed = (now - verify_start_t).to_sec()
        rospy.loginfo_throttle(0.5,
            f'[pick_ctrl] VERIFY_ROTATE … ({elapsed:.1f}/{ROTATE_90_TIME:.1f}s)')

        if elapsed >= ROTATE_90_TIME:
            chassis_stop()
            rospy.sleep(0.3)   # 진동 안정화
            rospy.loginfo('[pick_ctrl] Rotation done → VERIFY_CHECK')
            state = VERIFY_CHECK

    # ── VERIFY_CHECK ──────────────────────────────────────────────────────
    elif state == VERIFY_CHECK:
        grabbed = check_grasp_depth()

        if grabbed:
            rospy.loginfo('[pick_ctrl] ✓ Grasp SUCCESS → PLACE_ROTATE')
            place_start_t = now
            chassis_cmd(0, 90, ROTATE_SPEED)   # 90° 더 회전 → 총 180°
            state = PLACE_ROTATE
        else:
            rospy.logwarn('[pick_ctrl] ✗ Grasp FAILED → recovery SCAN_L')
            scan_start_t = now
            chassis_cmd(0, 90, SCAN_SPEED)     # 왼쪽으로 천천히 회전
            state = SCAN_L

    # ── PLACE_ROTATE ─────────────────────────────────────────────────────
    elif state == PLACE_ROTATE:
        if place_start_t is None:
            place_start_t = now
            chassis_cmd(0, 90, ROTATE_SPEED)

        elapsed = (now - place_start_t).to_sec()
        rospy.loginfo_throttle(0.5,
            f'[pick_ctrl] PLACE_ROTATE … ({elapsed:.1f}/{ROTATE_90_TIME:.1f}s)')

        if elapsed >= ROTATE_90_TIME:
            chassis_stop()
            rospy.loginfo('[pick_ctrl] 180° rotation done — drop object into basket → IDLE')
            # TODO: 필요하면 여기서 /place_target publish 또는 그리퍼 open 신호
            verify_start_t = None
            place_start_t  = None
            state = IDLE

    # ── SCAN_L ────────────────────────────────────────────────────────────
    elif state == SCAN_L:
        if scan_start_t is None:
            scan_start_t = now
            chassis_cmd(0, 90, SCAN_SPEED)

        elapsed = (now - scan_start_t).to_sec()

        if dt < DETECTION_TIMEOUT:
            # 스캔 중 물체 재발견
            chassis_stop()
            rospy.loginfo('[pick_ctrl] Object re-detected during SCAN_L → APPROACHING')
            scan_start_t = None
            state = APPROACHING
        elif elapsed >= SCAN_TIME:
            rospy.loginfo('[pick_ctrl] SCAN_L done → SCAN_R')
            scan_start_t = now
            chassis_cmd(0, 90, -SCAN_SPEED)   # 반대 방향 (오른쪽)
            state = SCAN_R

    # ── SCAN_R ────────────────────────────────────────────────────────────
    elif state == SCAN_R:
        if scan_start_t is None:
            scan_start_t = now
            chassis_cmd(0, 90, -SCAN_SPEED)

        elapsed = (now - scan_start_t).to_sec()

        if dt < DETECTION_TIMEOUT:
            chassis_stop()
            rospy.loginfo('[pick_ctrl] Object re-detected during SCAN_R → APPROACHING')
            scan_start_t = None
            state = APPROACHING
        elif elapsed >= SCAN_TIME * 2:   # 복귀 위해 2배 시간 (L에서 온 만큼 되돌아감)
            chassis_stop()
            rospy.logwarn('[pick_ctrl] Object not found after full scan → IDLE')
            scan_start_t = None
            state = IDLE


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rospy.init_node('pick_controller', anonymous=True)

    vel_pub  = rospy.Publisher(
        '/chassis_control/set_velocity', SetVelocity, queue_size=1)
    pick_pub = rospy.Publisher(
        '/pick_target', Point, queue_size=1)

    rospy.Subscriber('/yolo/arm_point', Point, arm_point_cb)
    rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_cb,
                     queue_size=1, buff_size=2**24)

    rospy.sleep(0.5)
    rospy.Timer(rospy.Duration(0.1), control_loop)

    rospy.loginfo('Pick controller ready (with grasp verification + recovery).')
    rospy.loginfo('  States: APPROACHING→CREEPING→PICKING→VERIFY_ROTATE→VERIFY_CHECK')
    rospy.loginfo('          ├─ SUCCESS → PLACE_ROTATE → IDLE')
    rospy.loginfo('          └─ FAIL    → SCAN_L → SCAN_R → APPROACHING or IDLE')
    rospy.loginfo(f'  Grasp ROI : {GRASP_ROI}  threshold : {GRASP_CHECK_DIST}m')
    rospy.loginfo(f'  Rotate 90°: {ROTATE_90_TIME:.1f}s @ {ROTATE_SPEED}rad/s  '
                  f'(tune ROTATE_SPEED if overshoot)')
    rospy.spin()