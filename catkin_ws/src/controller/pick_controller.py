#!/usr/bin/python3
# coding=utf8
"""
Pick Controller Node
─────────────────────────────────────────────────────────────────────────────
Subscribes to /yolo/arm_point (geometry_msgs/Point) and executes:
  1. [APPROACHING] Command chassis to move forward/strafe until the object enters the target window.
  2. [ALIGNED]     Target reached -> Publish /pick_target (Consumed by auto_pick.py).
  3. [PICKING]     Wait for picking to complete -> Return to IDLE state.

Coordinate System (Identical to auto_pick / yolo_detector):
  arm_x : Left/Right (Positive = Robot's Left)
  arm_y : Forward/Backward (Positive = Robot's Front, larger value = further away)
  arm_z : Height (Fixed value, set by yolo_detector)

Target Position: arm_x ≈ 0.000,  arm_y ≈ TARGET_Y (Default: 0.250 m)
  -> This is the optimal position where auto_pick can safely execute the pick operation.

Chassis SetVelocity Interface:
  velocity  : Linear speed in mm/s
  direction : Heading direction in degrees (0~360°)
              (90° = Forward, 270° = Backward, 0° = Strafe Right, 180° = Strafe Left)
  angular   : Angular velocity in rad/s (Positive = Counter-Clockwise / Turn Left)
"""

import rospy
from geometry_msgs.msg import Point
from chassis_control.msg import SetVelocity

import os
import sys
_pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_pkg_path, 'armpi_pro_kinematics'))

# ── Define states ─────────────────────────────────────────────────────────────
IDLE        = 'IDLE'
APPROACHING = 'APPROACHING'
CREEPING    = 'CREEPING'   # align and keep moving
ALIGNED     = 'ALIGNED'
PICKING     = 'PICKING'

# ── Parameters ────────────────────────────────────────────────────────────
TARGET_Y      = 0.250   # Target forward/backward distance (meters) <- Execute pick at this distance
TARGET_X      = 0.000   # Target left/right position (meters)

TOL_X         = 0.030   # Left/Right alignment tolerance: considered aligned within ±3 cm
TOL_Y         = 0.020   # Forward/Backward alignment tolerance: considered aligned within ±2 cm

FORWARD_SPEED = 80      # Forward/Backward speed (mm/s)
STRAFE_SPEED  = 60      # Sideways strafe speed (mm/s)
CREEP_SPEED   = 50      # Approach/Creep speed for fine adjustment after alignment (mm/s)
CREEP_TIME    = 0.6     # Approach/Creep duration (seconds) — 50mm/s × 0.6s ≈ 3.0cm

# Distance Thresholds: Ignore detections outside this range (out of bounds or noise)
# Given arm_y = -0.316 * Z_cam + 0.299, a larger Z_cam (further away) results in a smaller/negative arm_y.
# Based on Z_cam <= MAX_DEPTH (1.5m): arm_y_min ≈ -0.316 * 1.5 + 0.299 = -0.175
VALID_Y_MIN   = -0.30   # Lower bound of valid range (to filter out extreme noise)
VALID_Y_MAX   =  0.60

DETECTION_TIMEOUT = 2.0   # Stop the chassis if no object is detected for this duration (seconds)
PICK_COOLDOWN     = 4.0   # Cooldown time after a pick command before resuming detection (seconds)
                           # Note: auto_pick takes approx. 11s (MOVE_SLEEP × 6 + GRIP_SLEEP × 3)
                           # Since we wait by duration rather than a completion feedback signal, keep it generous.
PICK_WAIT         = 12.0  # Total wait time for the full pick-and-place sequence to complete (seconds)


# ── Global state ─────────────────────────────────────────────────────────────
state           = IDLE
last_detect_t   = None
pick_sent_t     = None
creep_start_t   = None
last_arm_pt     = None   # Most recent /yolo/arm_point


# ── Publisher (Initialized in main ) ───────────────────────────────────────────
vel_pub  = None
pick_pub = None


# ── helper ─────────────────────────────────────────────────────────────────
def chassis_cmd(velocity=0.0, direction=90.0, angular=0.0):
    msg = SetVelocity()
    msg.velocity  = float(velocity)
    msg.direction = float(direction)
    msg.angular   = float(angular)
    vel_pub.publish(msg)


def chassis_stop():
    chassis_cmd(0, 90, 0)


def send_pick(pt):
    """publish arm coordinate exactly one time to /pick_target"""
    pick_pub.publish(pt)
    rospy.loginfo(f'[pick_ctrl] PICK SENT  arm=({pt.x:.3f}, {pt.y:.3f}, {pt.z:.3f})')


# ── callback ─────────────────────────────────────────────────────────────────
def arm_point_cb(msg):
    """
    subscribe /yolo/arm_point
    filter effective range and update globle variable
    """
    global last_detect_t, last_arm_pt

    if not (VALID_Y_MIN < msg.y < VALID_Y_MAX):
        return   # ignore noises out of range

    last_arm_pt  = msg
    last_detect_t = rospy.Time.now()


# ── main loop ─────────────────────────────────────────────────────────────
def control_loop(event):
    """
    rospy.Timer call back (10 Hz).
    state Machine Transition & Chassis Command Issuance
    """
    global state, pick_sent_t, creep_start_t

    now = rospy.Time.now()

    # ── check detection timeout ────────────────────────────────────────────────
    if last_detect_t is None:
        if state == APPROACHING:
            rospy.logwarn_throttle(3, '[pick_ctrl] No detection — stopping chassis')
            chassis_stop()
            state = IDLE
        return

    dt = (now - last_detect_t).to_sec()

    # ─────────────────────────────────────────────────────────────────────
    if state == IDLE:
        if dt < DETECTION_TIMEOUT:
            rospy.loginfo('[pick_ctrl] Object detected → APPROACHING')
            state = APPROACHING

    # ─────────────────────────────────────────────────────────────────────
    elif state == APPROACHING:
        if dt >= DETECTION_TIMEOUT:
            rospy.logwarn('[pick_ctrl] Detection lost → IDLE')
            chassis_stop()
            state = IDLE
            return

        pt = last_arm_pt
        err_x = pt.x - TARGET_X   # Positive = Object is to the left -> Robot must strafe right
        err_y = pt.y - TARGET_Y   # Positive = Object is too far   -> Move forward
                                  # Negative = Object is too close -> Move backward

        aligned_x = abs(err_x) < TOL_X
        aligned_y = abs(err_y) < TOL_Y

        rospy.loginfo_throttle(0.5,
            f'[pick_ctrl] APPROACHING  '
            f'arm=({pt.x:.3f},{pt.y:.3f})  '
            f'err=({err_x:+.3f},{err_y:+.3f})  '
            f'aligned=({aligned_x},{aligned_y})')

        if aligned_x and aligned_y:
            # ── align completed --> creep and pick ──────────────────────────
            rospy.loginfo(f'[pick_ctrl] ALIGNED → CREEPING ({CREEP_SPEED}mm/s × {CREEP_TIME}s ≈ {CREEP_SPEED*CREEP_TIME:.0f}mm)')
            chassis_cmd(CREEP_SPEED, 90, 0)
            state = CREEPING

        else:
            # ── Determine Motion Command ──────────────────────────────
            # Priority: Forward/Backward correction takes precedence.
            # If Left/Right error is significant, execute strafe motion afterwards.
            #
            # Direction Conventions (SetVelocity):
            #   90°  = Forward (+y direction, robot front)
            #   270° = Backward
            #   0°   = Strafe Right (+x direction)
            #   180° = Strafe Left (-x direction)
            #
            # err_y > 0 -> Object is far   -> Move Forward (90°)
            # err_y < 0 -> Object is close -> Move Backward (270°)
            # err_x > 0 -> Object is to the left  -> Robot moves Left (180°)
            # err_x < 0 -> Object is to the right -> Robot moves Right (0°)

            if not aligned_y:
                # Smaller arm_y = Larger Z_cam (camera depth) = Object is far away -> Move Forward
                # arm_y < TARGET_Y -> err_y < 0 -> Move Forward (90°)
                # arm_y > TARGET_Y -> err_y > 0 -> Move Backward (270°)
                direction = 270 if err_y > 0 else 90
                chassis_cmd(FORWARD_SPEED, direction, 0)
            else:
                # Y-axis aligned, compensating X-axis error
                # Robot moving Right (0°) -> Camera moves right -> Object shifts left in image
                # -> Decreases X_cam -> Decreases arm_x. Therefore:
                # err_x < 0 (arm_x < 0, object is on image left)  -> Robot moves LEFT (180°)  -> Increases arm_x
                # err_x > 0 (arm_x > 0, object is on image right) -> Robot moves RIGHT (0°) -> Decreases arm_x
                direction = 0 if err_x > 0 else 180
                chassis_cmd(STRAFE_SPEED, direction, 0)

    # ─────────────────────────────────────────────────────────────────────
    elif state == CREEPING:
        if creep_start_t is None:
            creep_start_t = now
        elapsed = (now - creep_start_t).to_sec()
        if elapsed >= CREEP_TIME:
            chassis_stop()
            rospy.loginfo('[pick_ctrl] Creep done → sending pick command')
            state = ALIGNED

    # ─────────────────────────────────────────────────────────────────────
    elif state == ALIGNED:
        if last_arm_pt is not None:
            send_pick(last_arm_pt)
        pick_sent_t  = rospy.Time.now()
        creep_start_t = None
        state = PICKING

    # ─────────────────────────────────────────────────────────────────────
    elif state == PICKING:
        if pick_sent_t is None:
            state = IDLE
            return

        elapsed = (now - pick_sent_t).to_sec()
        if elapsed < PICK_WAIT:
            rospy.loginfo_throttle(2,
                f'[pick_ctrl] PICKING … ({elapsed:.1f}/{PICK_WAIT:.0f}s)')
        else:
            rospy.loginfo('[pick_ctrl] Pick complete → IDLE')
            state = IDLE
            pick_sent_t = None


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rospy.init_node('pick_controller', anonymous=True)

    vel_pub  = rospy.Publisher(
        '/chassis_control/set_velocity', SetVelocity, queue_size=1)
    pick_pub = rospy.Publisher(
        '/pick_target', Point, queue_size=1)

    rospy.Subscriber('/yolo/arm_point', Point, arm_point_cb)

    rospy.sleep(0.5)

    # 10 Hz control loop
    rospy.Timer(rospy.Duration(0.1), control_loop)

    rospy.loginfo('Pick controller ready.')
    rospy.loginfo('  Subscribing : /yolo/arm_point')
    rospy.loginfo('  Publishing  : /chassis_control/set_velocity  /pick_target')
    rospy.loginfo(f'  Target      : x={TARGET_X:.3f} ± {TOL_X:.3f}  '
                  f'y={TARGET_Y:.3f} ± {TOL_Y:.3f}  (m)')
    rospy.spin()