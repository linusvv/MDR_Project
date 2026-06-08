#!/usr/bin/python3
# coding=utf8
"""
Pick Arm Node (Combined auto_pick + pick_controller)
─────────────────────────────────────────────────────────────────────────────
[pick_controller Role]
  Subscribes to /yolo/arm_point (geometry_msgs/Point)
  → Drives/strafes the chassis to align the object with the TARGET position.
  → Once aligned, directly executes pick_and_place() in the main loop thread.

[auto_pick Role]
  Solves IK and executes servo commands sequentially to pick & place.

Coordinate System:
  arm_x : Left(-)/Right(+)   (Left side of the robot is +)
  arm_y : Front(+)/Back(-)   (Larger value means further away)
  arm_z : Up(+)/Down(-)

Chassis SetVelocity:
  velocity  : mm/s
  direction : 0~360°  (90°=Forward, 270°=Backward, 0°=Right, 180°=Left)
  angular   : rad/s   (Positive=Counter-Clockwise=Left turn)

Optional topic:
  /place_target (geometry_msgs/Point) : Real-time update for drop position
"""

import os
import sys

import json
from std_msgs.msg import String, Bool

import rospy
from geometry_msgs.msg import Point
from hiwonder_servo_msgs.msg import MultiRawIdPosDur, RawIdPosDur
from chassis_control.msg import SetVelocity

_pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_pkg_path, 'armpi_pro_kinematics'))
sys.path.insert(0, '/home/ee478_team1/catkin_ws/src/armpi_pro_kinematics')
from kinematics import ik_transform


# ════════════════════════════════════════════════════════════════════════════
# ── Arm Parameters (auto_pick) ───────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
Z_APPROACH     = 0.12       # Approach height (m)
Z_GRASP        = -0.015     # Grasp height (m)  ← Tuned to actual measurements
MOVE_SLEEP     = 1.2        # Wait time after each movement (s)
GRIP_SLEEP     = 0.6        # Wait time after gripper movement (s)

SERVO_DURATION = 800        # Servo travel duration (ms)
GRIPPER_OPEN   = 200        # Servo 1 pulse (Open)
GRIPPER_CLOSE  = 500        # Servo 1 pulse (Close) ← Adjust based on object size
SERVO2_DEFAULT = 500        # Joint 5 fixed value

PITCH          = -90        # End-effector pitch (deg)
PITCH_MIN      = -150
PITCH_MAX      = -30

HOME           = (0.00, 0.15, 0.12)    # Home position
DEFAULT_DROP   = (0.12, 0.15, 0.12)    # Default drop position


# ════════════════════════════════════════════════════════════════════════════
# ── Controller Parameters (pick_controller) ──────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
IDLE        = 'IDLE'
APPROACHING = 'APPROACHING'
CREEPING    = 'CREEPING'
ALIGNED     = 'ALIGNED'

TARGET_Y          = 0.250   # Target distance forward/back (m) — execute pick here
TARGET_X          = 0.000   # Target distance left/right (m)
TOL_X             = 0.030   # Left/right tolerance ±3 cm
TOL_Y             = 0.020   # Forward/back tolerance ±2 cm

FORWARD_SPEED     = 80      # Forward/backward speed (mm/s)
STRAFE_SPEED      = 60      # Left/right strafe speed (mm/s)
CREEP_SPEED       = 50      # Fine approach speed (mm/s)
CREEP_TIME        = 0.4     # Fine approach duration (s)  →  50mm/s × 0.6s ≈ 3 cm

VALID_Y_MIN       = -0.30   # Valid detection range lower limit
VALID_Y_MAX       =  0.60   # Valid detection range upper limit
DETECTION_TIMEOUT = 2.0     # Stop if no detection occurs within this time (s)


# ════════════════════════════════════════════════════════════════════════════
# ── Global State ─────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
state         = IDLE
last_detect_t = None
creep_start_t = None
last_arm_pt   = None        # Most recent /yolo/arm_point

_drop_pos     = list(DEFAULT_DROP)

# ── Publisher handles ─────────────────────────────────────────────────────
_servo_pub = None
_ik        = None
vel_pub    = None

# check variables
target_item = "None"
target_visible = False
heartbeat_pub = None
target_visible_pub = None


# ════════════════════════════════════════════════════════════════════════════
# ── Arm Control Helpers (from auto_pick) ─────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def init_arm(servo_pub):
    global _servo_pub, _ik
    _servo_pub = servo_pub
    _ik = ik_transform.ArmIK()


def send_servos(servo_dict, duration=SERVO_DURATION):
    msg = MultiRawIdPosDur()
    items = []
    for sid, pos in servo_dict.items():
        item          = RawIdPosDur()
        item.id       = int(sid)
        item.position = int(max(0, min(1000, pos)))
        item.duration = int(duration)
        items.append(item)
    msg.id_pos_dur_list = items
    _servo_pub.publish(msg)


def set_gripper(pulse):
    send_servos({1: pulse}, duration=400)


def move_to(x, y, z, label='', step_down=False, current_z=None):
    """
    Solves IK and sends servo commands.
    If step_down=True, descends step-by-step from current_z to target z at 0.02m intervals
    (to prevent IK solution branch shifting).
    """
    if step_down and current_z is not None and current_z > z:
        steps = int((current_z - z) / 0.02)
        for i in range(1, steps + 1):
            mid_z  = current_z - i * 0.02
            target = _ik.setPitchRanges((x, y, mid_z), PITCH, PITCH_MIN, PITCH_MAX)
            if target:
                sd = target[1]
                send_servos({2: SERVO2_DEFAULT, 3: sd['servo3'],
                             4: sd['servo4'],   5: sd['servo5'], 6: sd['servo6']})
                rospy.sleep(0.3)

    target = _ik.setPitchRanges((x, y, z), PITCH, PITCH_MIN, PITCH_MAX)
    if target:
        sd = target[1]
        send_servos({
            2: SERVO2_DEFAULT,
            3: sd['servo3'],
            4: sd['servo4'],
            5: sd['servo5'],
            6: sd['servo6'],
        })
        rospy.loginfo(f'  → {label}  ({x:.3f}, {y:.3f}, {z:.3f})')
        return True
    else:
        rospy.logwarn(f'  No IK solution: {label}  ({x:.3f}, {y:.3f}, {z:.3f})')
        return False


def pick_and_place(px, py, pz, dx, dy, dz):
    """
    Executes the entire sequence: pick (px,py,pz) → place (dx,dy,dz).
    Called directly (blocking) in the ALIGNED state.
    """
    global heartbeat_pub
    if heartbeat_pub is not None:
        heartbeat_pub.publish(String("picking up object"))

    rospy.loginfo(f'=== PICK  ({px:.3f}, {py:.3f}, {pz:.3f})')
    rospy.loginfo(f'=== PLACE ({dx:.3f}, {dy:.3f}, {dz:.3f})')

    # ── PICK PHASE ────────────────────────────────────────────────────────
    # 1. Open gripper
    set_gripper(GRIPPER_OPEN)
    rospy.sleep(GRIP_SLEEP)

    # 2. Move to approach height
    if not move_to(px, py, Z_APPROACH, 'approach'):
        rospy.logerr('Pick aborted: approach IK failed')
        return False
    rospy.sleep(MOVE_SLEEP)

    # 3. Step-down descent to grasp height
    if not move_to(px, py, pz, 'grasp', step_down=True, current_z=Z_APPROACH):
        rospy.logerr('Pick aborted: grasp IK failed')
        return False
    rospy.sleep(MOVE_SLEEP)

    # 4. Close gripper (grasp)
    set_gripper(GRIPPER_CLOSE)
    rospy.sleep(GRIP_SLEEP)

    # 5. Lift up
    if not move_to(px, py, Z_APPROACH, 'lift'):
        rospy.logerr('Lift failed — gripper might still be holding the object')
    rospy.sleep(MOVE_SLEEP)

    # ── PLACE PHASE ───────────────────────────────────────────────────────
    # 6. Move above the drop position
    if not move_to(dx, dy, max(dz + 0.05, Z_APPROACH), 'place_approach'):
        rospy.logerr('Place aborted: approach IK failed')
        return False
    rospy.sleep(MOVE_SLEEP)

    # 7. Descend to drop height
    if not move_to(dx, dy, dz, 'place'):
        rospy.logerr('Place aborted: place IK failed')
        return False
    rospy.sleep(MOVE_SLEEP)

    # 8. Open gripper (release)
    set_gripper(GRIPPER_OPEN)
    rospy.sleep(GRIP_SLEEP)

    # 9. Return to HOME
    move_to(*HOME, 'home')
    rospy.sleep(MOVE_SLEEP)

    rospy.loginfo('=== Pick & Place Completed ===')
    if heartbeat_pub is not None:
        heartbeat_pub.publish(String("picked up object"))
    return True


# ════════════════════════════════════════════════════════════════════════════
# ── Chassis Helpers (from pick_controller) ───────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def chassis_cmd(velocity=0.0, direction=90.0, angular=0.0):
    msg           = SetVelocity()
    msg.velocity  = float(velocity)
    msg.direction = float(direction)
    msg.angular   = float(angular)
    vel_pub.publish(msg)


def chassis_stop():
    chassis_cmd(0, 90, 0)


# ════════════════════════════════════════════════════════════════════════════
# ── ROS Callbacks ────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def arm_point_cb(msg):
    """
    Receive /yolo/arm_point: Update global variables after filtering valid range.
    """
    global last_detect_t, last_arm_pt
    if not (VALID_Y_MIN < msg.y < VALID_Y_MAX):
        return
    last_arm_pt   = msg
    last_detect_t = rospy.Time.now()


def target_item_cb(msg):
    global target_item
    target_item = msg.data
    rospy.loginfo(f'[pick_arm] Target item updated: {target_item}')

def place_target_cb(msg):
    """
    Receive /place_target: Real-time update for drop position.
    """
    global _drop_pos
    _drop_pos = [
        msg.x,
        msg.y,
        msg.z if msg.z > 0.001 else DEFAULT_DROP[2],
    ]
    rospy.loginfo(f'[pick_arm] Drop position updated: {_drop_pos}')


# ════════════════════════════════════════════════════════════════════════════
# ── Control Loop  10 Hz (from pick_controller) ───────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def control_loop(event):
    """
    rospy.Timer callback (10 Hz).
    Handles state machine transitions and issues chassis commands.
    In ALIGNED state, directly calls pick_and_place() (blocking) and returns to IDLE.
    """
    global state, creep_start_t, target_visible

    now = rospy.Time.now()

    # Update target_visible state
    if last_detect_t is None:
        target_visible = False
    else:
        dt = (now - last_detect_t).to_sec()
        target_visible = (dt < DETECTION_TIMEOUT)

    if target_visible_pub is not None:
        target_visible_pub.publish(Bool(target_visible))

    # ── Check detection timeout ───────────────────────────────────────────
    if not target_visible:
        if state in [APPROACHING, CREEPING]:
            rospy.logwarn_throttle(3, '[pick_arm] No detection — Stopping chassis')
            chassis_stop()
            state = IDLE
        if heartbeat_pub is not None and state == IDLE:
            heartbeat_pub.publish(String("idle"))
        return

    # ── IDLE ──────────────────────────────────────────────────────────────
    if state == IDLE:
        rospy.loginfo('[pick_arm] Object detected → APPROACHING')
        state = APPROACHING
        if heartbeat_pub is not None:
            heartbeat_pub.publish(String("detecting object"))

    # ── APPROACHING ───────────────────────────────────────────────────────
    elif state == APPROACHING:
        if dt >= DETECTION_TIMEOUT:
            rospy.logwarn('[pick_arm] Detection lost → IDLE')
            chassis_stop()
            state = IDLE
            return

        pt    = last_arm_pt
        err_x = pt.x - TARGET_X   # + : Object is to the left → Robot needs to move right
        err_y = pt.y - TARGET_Y   # + : Object is far away   → Needs to move forward
                                   # - : Object is close      → Needs to move backward

        aligned_x = abs(err_x) < TOL_X
        aligned_y = abs(err_y) < TOL_Y

        rospy.loginfo_throttle(0.5,
            f'[pick_arm] APPROACHING  '
            f'arm=({pt.x:.3f},{pt.y:.3f})  '
            f'err=({err_x:+.3f},{err_y:+.3f})  '
            f'aligned=({aligned_x},{aligned_y})')

        if aligned_x and aligned_y:
            rospy.loginfo(
                f'[pick_arm] ALIGNED → CREEPING '
                f'({CREEP_SPEED}mm/s × {CREEP_TIME}s ≈ {CREEP_SPEED*CREEP_TIME:.0f}mm)')
            chassis_cmd(CREEP_SPEED, 90, 0)
            state = CREEPING
        else:
            # Prioritize forward/back error → correct left/right error after alignment
            if not aligned_y:
                # err_y < 0 → arm_y < TARGET_Y → Object is far away → Forward (90°)
                # err_y > 0 → arm_y > TARGET_Y → Object is close    → Backward (270°)
                direction = 270 if err_y > 0 else 90
                chassis_cmd(FORWARD_SPEED, direction, 0)
            else:
                # err_x > 0 → Object is to the right (+arm_x) → Strafe right (0°)   → decreases arm_x
                # err_x < 0 → Object is to the left (-arm_x)  → Strafe left (180°)  → increases arm_x
                direction = 0 if err_x > 0 else 180
                chassis_cmd(STRAFE_SPEED, direction, 0)

    # ── CREEPING ──────────────────────────────────────────────────────────
    elif state == CREEPING:
        if heartbeat_pub is not None:
            heartbeat_pub.publish(String("approaching object"))
        if creep_start_t is None:
            creep_start_t = now
        elapsed = (now - creep_start_t).to_sec()
        if elapsed >= CREEP_TIME:
            chassis_stop()
            rospy.loginfo('[pick_arm] Creep completed → ALIGNED')
            state = ALIGNED

    # ── ALIGNED ───────────────────────────────────────────────────────────
    elif state == ALIGNED:
        if last_arm_pt is not None:
            pt = last_arm_pt
            px, py = pt.x, pt.y
            pz     = pt.z if pt.z > -0.020 else Z_GRASP
            dx, dy, dz = _drop_pos
            rospy.loginfo(
                f'[pick_arm] Starting Pick  '
                f'arm=({px:.3f}, {py:.3f}, {pz:.3f})')
            pick_and_place(px, py, pz, dx, dy, dz)   # Blocking call
        creep_start_t = None
        state         = IDLE
        rospy.loginfo('[pick_arm] Pick completed → IDLE')


# ════════════════════════════════════════════════════════════════════════════
# ── Main ─────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    rospy.init_node('pick_arm_node', anonymous=True)

    # ── Publishers ────────────────────────────────────────────────────────
    servo_pub = rospy.Publisher(
        '/servo_controllers/port_id_1/multi_id_pos_dur',
        MultiRawIdPosDur, queue_size=1)
    vel_pub = rospy.Publisher(
        '/chassis_control/set_velocity', SetVelocity, queue_size=1)
    
    target_visible_pub = rospy.Publisher('/target_visible', Bool, queue_size=1)
    heartbeat_pub = rospy.Publisher('/pick_arm_heartbeat', String, queue_size=1)

    rospy.sleep(0.5)

    # ── Arm Initialization + Move to HOME ────────────────────────────────
    init_arm(servo_pub)
    rospy.loginfo('[pick_arm] Moving to HOME position…')
    move_to(*HOME, 'home')
    rospy.sleep(1.5)

    # ── Subscribers ───────────────────────────────────────────────────────
    rospy.Subscriber('/yolo/arm_point', Point, arm_point_cb)
    rospy.Subscriber('/place_target',   Point, place_target_cb)
    rospy.Subscriber('/target_item',    String, target_item_cb)

    # ── 10 Hz Control Loop ────────────────────────────────────────────────
    rospy.Timer(rospy.Duration(0.1), control_loop)

    rospy.loginfo('[pick_arm] Node is ready.')
    rospy.loginfo('  Subscribed: /yolo/arm_point  /place_target')
    rospy.loginfo('  Published:  /chassis_control/set_velocity')
    rospy.loginfo('              /servo_controllers/port_id_1/multi_id_pos_dur')
    rospy.loginfo(
        f'  Target: x={TARGET_X:.3f} ± {TOL_X:.3f}  '
        f'y={TARGET_Y:.3f} ± {TOL_Y:.3f}  (m)')
    rospy.spin()