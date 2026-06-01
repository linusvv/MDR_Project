#!/usr/bin/python3
# coding=utf8
"""
Pick Arm Node  (auto_pick + pick_controller merged)
─────────────────────────────────────────────────────────────────────────────
[pick_controller role]
  Subscribes to /yolo/arm_point (geometry_msgs/Point)
  -> Drives chassis forward/strafe to align object to TARGET position
  -> On alignment, calls pick_and_place() directly (blocking)

[auto_pick role]
  Solves IK and sends servo commands to execute pick & place sequence

Coordinate System:
  arm_x : Left(-) / Right(+)   (positive = robot's left)
  arm_y : Forward(+) / Back(-) (larger = further away)
  arm_z : Up(+) / Down(-)

Chassis SetVelocity:
  velocity  : mm/s
  direction : 0~360°  (90=forward, 270=backward, 0=strafe right, 180=strafe left)
  angular   : rad/s   (positive = counter-clockwise = turn left)

Optional topic:
  /place_target (geometry_msgs/Point) : update drop position at runtime
"""

import os
import sys

import rospy
from geometry_msgs.msg import Point
from hiwonder_servo_msgs.msg import MultiRawIdPosDur, RawIdPosDur
from chassis_control.msg import SetVelocity
from std_msgs.msg import Float64, String, Bool


_pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_pkg_path, 'armpi_pro_kinematics'))
sys.path.insert(0, '/home/ee478_team1/catkin_ws/src/armpi_pro_kinematics')
from kinematics import ik_transform


# ════════════════════════════════════════════════════════════════════════════
# ── Arm Parameters ───────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
Z_APPROACH     = 0.12       # Approach height (m)
# Z_GRASP        = -0.015     # Grasp height (m)  <- tune based on physical measurement
Z_GRASP        = 0.03       # test hight so see if 60° works
MOVE_SLEEP     = 1.2        # Dwell time after each motion (s)
GRIP_SLEEP     = 0.6        # Dwell time after gripper actuation (s)

SERVO_DURATION = 800        # Servo travel time (ms)
GRIPPER_OPEN   = 200        # Servo 1 pulse width (open)
GRIPPER_CLOSE  = 600        # Servo 1 pulse width (closed) <- increased for tighter grip
SERVO2_DEFAULT = 500        # Fixed pulse value for joint 5

PITCH          = -140       # End-effector pitch angle (deg) — tilted downward
PITCH_MIN      = -150
PITCH_MAX      = -90

HOME           = (0.00, 0.15, 0.12)    # Home position (waypoint after lift)


# ════════════════════════════════════════════════════════════════════════════
# ── Controller Parameters ────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
IDLE        = 'IDLE'
APPROACHING = 'APPROACHING'
CREEPING    = 'CREEPING'
ALIGNED     = 'ALIGNED'

TARGET_Y          = 0.250   # Target forward distance (m) — pick executes here
TARGET_X          = 0.000   # Target lateral position (m)
TOL_X             = 0.030   # Lateral alignment tolerance ±3 cm
TOL_Y             = 0.020   # Forward alignment tolerance ±2 cm

FORWARD_SPEED     = 80      # Forward/backward speed (mm/s)
STRAFE_SPEED      = 60      # Lateral strafe speed (mm/s)
CREEP_SPEED       = 50      # Fine approach speed (mm/s)
CREEP_TIME        = 0.2     # Fine approach duration (s)  ->  50mm/s x 0.2s ~ 10mm

VALID_Y_MIN       = -0.30   # Valid detection range lower bound
VALID_Y_MAX       =  0.60   # Valid detection range upper bound
DETECTION_TIMEOUT = 2.0     # Stop chassis if no detection for this duration (s)


# ════════════════════════════════════════════════════════════════════════════
# ── Global State ─────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
state         = IDLE
last_detect_t = None
creep_start_t = None
last_arm_pt   = None        # Latest /yolo/arm_point message

# ── Publisher handles ─────────────────────────────────────────────────────
_servo_pub     = None
_ik            = None
vel_pub        = None
_joint_pubs    = {}     # {1: Publisher, ...} for /joint{n}_controller/command

# check variables
target_item = "None"
target_visible = False
heartbeat_pub = None
target_visible_pub = None


# ════════════════════════════════════════════════════════════════════════════
# ── Arm Control Helpers ──────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def init_arm(servo_pub):
    global _servo_pub, _ik
    _servo_pub = servo_pub
    _ik = ik_transform.ArmIK()


def init_joints():
    """Initialize publishers for joint1~5 controllers."""
    global _joint_pubs
    for n in range(1, 6):
        _joint_pubs[n] = rospy.Publisher(
            f'/joint{n}_controller/command', Float64, queue_size=1)


def set_final_pose():
    """
    Move arm to final place pose using absolute joint angles (rad), then open gripper.
      joint1=0, joint2=-0.4, joint3=1.0, joint4=1.0, joint5=0, gripper open
    """
    rospy.loginfo('[pick_arm] Moving to final place pose...')
    _joint_pubs[1].publish(Float64(data= 0.0))
    _joint_pubs[2].publish(Float64(data=-0.4))
    _joint_pubs[3].publish(Float64(data= 1.0))
    _joint_pubs[4].publish(Float64(data= 1.0))
    _joint_pubs[5].publish(Float64(data= 0.0))
    rospy.sleep(MOVE_SLEEP)
    set_gripper(GRIPPER_OPEN)
    rospy.sleep(GRIP_SLEEP)
    rospy.loginfo('[pick_arm] Final pose reached')


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
    Solve IK and send servo commands to move arm to (x, y, z).
    If step_down=True, descend incrementally from current_z to z in 0.02m steps
    to prevent IK solution branch flipping.
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
        rospy.loginfo(f'  -> {label}  ({x:.3f}, {y:.3f}, {z:.3f})')
        return True
    else:
        rospy.logwarn(f'  No IK solution: {label}  ({x:.3f}, {y:.3f}, {z:.3f})')
        return False


def pick_and_place(px, py, pz):
    """
    Pick object at (px, py, pz), move to place pose via joint controllers, release.
    Called directly (blocking) from ALIGNED state.
    Place position is defined by set_final_pose() joint absolute values.
    """
    
    global heartbeat_pub
    if heartbeat_pub is not None:
        heartbeat_pub.publish(String("picking up object"))
    
    rospy.loginfo(f'=== PICK  ({px:.3f}, {py:.3f}, {pz:.3f})')

    # ── PICK PHASE ────────────────────────────────────────────────────────
    # 1. Open gripper
    set_gripper(GRIPPER_OPEN)
    rospy.sleep(GRIP_SLEEP)

    # 2. Move to approach height
    if not move_to(px, py, Z_APPROACH, 'approach'):
        rospy.logerr('Pick aborted: approach IK failed')
        return False
    rospy.sleep(MOVE_SLEEP)

    # 3. Descend incrementally to grasp height
    if not move_to(px, py, pz, 'grasp', step_down=True, current_z=Z_APPROACH):
        rospy.logerr('Pick aborted: grasp IK failed')
        return False
    rospy.sleep(MOVE_SLEEP)

    # 4. Close gripper (grasp object)
    set_gripper(GRIPPER_CLOSE)
    rospy.sleep(GRIP_SLEEP)


    # 5. Lift object
    if not move_to(px, py, Z_APPROACH, 'lift'):
        rospy.logerr('Lift failed — gripper may still be holding object')
    rospy.sleep(MOVE_SLEEP)

    # ── PLACE PHASE ───────────────────────────────────────────────────────
    # 6. Move to place pose via joint controllers and release object
    set_final_pose()

    rospy.loginfo('=== Pick & Place DONE ===')
    if heartbeat_pub is not None:
        heartbeat_pub.publish(String("picked up object"))
    return True


# ════════════════════════════════════════════════════════════════════════════
# ── Chassis Helpers ──────────────────────────────────────────────────────
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
    Receive /yolo/arm_point, filter by valid range, update global state.
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


# ════════════════════════════════════════════════════════════════════════════
# ── Control Loop  10 Hz ──────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def control_loop(event):
    """
    rospy.Timer callback at 10 Hz.
    Handles state machine transitions and chassis commands.
    In ALIGNED state, calls pick_and_place() directly (blocking) then returns to IDLE.
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

    # ── Detection timeout check ───────────────────────────────────────────
    if not target_visible:
        if state in [APPROACHING, CREEPING]:
            rospy.logwarn_throttle(3, '[pick_arm] No detection — stopping chassis')
            chassis_stop()
            state = IDLE
        if heartbeat_pub is not None and state == IDLE:
            heartbeat_pub.publish(String("idle"))
        return

    # ── IDLE ──────────────────────────────────────────────────────────────
    if state == IDLE:
        rospy.loginfo('[pick_arm] Object detected -> APPROACHING')
        state = APPROACHING
        if heartbeat_pub is not None:
            heartbeat_pub.publish(String("detecting object"))

    # ── APPROACHING ───────────────────────────────────────────────────────
    elif state == APPROACHING:
        if dt >= DETECTION_TIMEOUT:
            rospy.logwarn('[pick_arm] Detection lost -> IDLE')
            chassis_stop()
            state = IDLE
            return

        pt    = last_arm_pt
        err_x = pt.x - TARGET_X   # positive: object is left  -> robot must strafe right
        err_y = pt.y - TARGET_Y   # positive: object is far   -> move forward
                                   # negative: object is close -> move backward

        aligned_x = abs(err_x) < TOL_X
        aligned_y = abs(err_y) < TOL_Y

        rospy.loginfo_throttle(0.5,
            f'[pick_arm] APPROACHING  '
            f'arm=({pt.x:.3f},{pt.y:.3f})  '
            f'err=({err_x:+.3f},{err_y:+.3f})  '
            f'aligned=({aligned_x},{aligned_y})')

        if aligned_x and aligned_y:
            rospy.loginfo(
                f'[pick_arm] ALIGNED -> CREEPING '
                f'({CREEP_SPEED}mm/s x {CREEP_TIME}s ~ {CREEP_SPEED*CREEP_TIME:.0f}mm)')
            chassis_cmd(CREEP_SPEED, 90, 0)
            state = CREEPING
        else:
            # Correct forward/backward error first, then lateral error
            if not aligned_y:
                # err_y < 0 -> arm_y < TARGET_Y -> object is far  -> forward (90deg)
                # err_y > 0 -> arm_y > TARGET_Y -> object is close -> backward (270deg)
                direction = 270 if err_y > 0 else 90
                chassis_cmd(FORWARD_SPEED, direction, 0)
            else:
                # err_x > 0 -> object is right (+arm_x) -> strafe right (0deg) -> arm_x decreases
                # err_x < 0 -> object is left  (-arm_x) -> strafe left (180deg) -> arm_x increases
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
            rospy.loginfo('[pick_arm] Creep done -> ALIGNED')
            state = ALIGNED

    # ── ALIGNED ───────────────────────────────────────────────────────────
    elif state == ALIGNED:
        if last_arm_pt is not None:
            pt = last_arm_pt
            px, py = pt.x, pt.y
            pz     = pt.z if pt.z > -0.020 else Z_GRASP
            rospy.loginfo(
                f'[pick_arm] Starting pick  '
                f'arm=({px:.3f}, {py:.3f}, {pz:.3f})')
            pick_and_place(px, py, pz)   # blocking direct call
        creep_start_t = None
        state         = IDLE
        rospy.loginfo('[pick_arm] Pick complete -> IDLE')


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

    # ── Initialize arm and move to HOME ──────────────────────────────────
    init_arm(servo_pub)
    init_joints()
    rospy.loginfo('[pick_arm] Moving to HOME...')
    move_to(*HOME, 'home')
    rospy.sleep(1.5)

    # ── Subscribers ───────────────────────────────────────────────────────
    rospy.Subscriber('/yolo/arm_point',           Point,           arm_point_cb)
    rospy.Subscriber('/target_item',              String,          target_item_cb)

    # ── 10 Hz control loop ────────────────────────────────────────────────
    rospy.Timer(rospy.Duration(0.1), control_loop)

    rospy.loginfo('[pick_arm] Node ready.')
    rospy.loginfo('  Subscribing : /yolo/arm_point  /target_item')
    rospy.loginfo('  Publishing  : /chassis_control/set_velocity')
    rospy.loginfo('               /servo_controllers/port_id_1/multi_id_pos_dur')
    rospy.loginfo(
        f'  Target      : x={TARGET_X:.3f} +/- {TOL_X:.3f}  '
        f'y={TARGET_Y:.3f} +/- {TOL_Y:.3f}  (m)')
    rospy.spin()