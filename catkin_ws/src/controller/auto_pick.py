#!/usr/bin/python3
# coding=utf8
"""
ArmPi Pro Auto Pick Node
─────────────────────────────────────────────────────────────────────────────
Executes an automated pick-and-place sequence upon receiving a 
target coordinate published to the /pick_target (geometry_msgs/Point) topic.

Coordinate System (in meters):
  x : Left/Right     (Positive = Robot's Left)
  y : Forward/Backward (Positive = Robot's Front)
  z : Height         (If pick_target.z is 0.0, the default Z_GRASP value is used)

Usage Examples:
  # Publish target pick position from another terminal:
  rostopic pub /pick_target geometry_msgs/Point "x: 0.0
y: 0.18
z: 0.0"

  # Or specify the drop/place position as well (place_target):
  rostopic pub /place_target geometry_msgs/Point "x: 0.1
y: 0.20
z: 0.10"
"""

import sys
import rospy
from geometry_msgs.msg import Point
from hiwonder_servo_msgs.msg import MultiRawIdPosDur, RawIdPosDur
from std_msgs.msg import Bool

sys.path.insert(0, '/home/ee478_team1/catkin_ws/src/armpi_pro_kinematics')
from kinematics import ik_transform

# ── Parameters ─────────────────────────────────────────────────────────
Z_APPROACH  = 0.12   # Approach height (airspace above the object, meters)
Z_GRASP     = -0.015   # Grasp height (actual picking level, meters) — Tune based on physical measurements
MOVE_SLEEP  = 1.2    # Dwell time after each motion (seconds) — Allows the arm to fully settle
GRIP_SLEEP  = 0.6    # Dwell time for gripper actuation (seconds)

SERVO_DURATION  = 800   # Servo travel time (milliseconds)
GRIPPER_OPEN    = 200   # Servo 1 pulse width (Open)
GRIPPER_CLOSE   = 500   # Servo 1 pulse width (Closed) — Adjust based on object geometry/size
SERVO2_DEFAULT  = 500   # Fixed pulse value for Joint 5

PITCH       = -90    # End-effector pitch angle (degrees)
PITCH_MIN   = -150   # Minimum allowable pitch — Used to lock the IK solution branch
PITCH_MAX   = -30    # Maximum allowable pitch — Used to prevent elbow-up configuration flip

# HOME Position (Returns here after completing a pick-and-place cycle)
HOME = (0.00, 0.15, 0.12)

# Default Drop/Place Position (Fallback if no /place_target message is received)
DEFAULT_DROP = (0.12, 0.15, 0.12)

# ── helper ────────────────────────────────────────────────────────
_servo_pub = None
_ik = None


def init(servo_pub):
    global _servo_pub, _ik
    _servo_pub = servo_pub
    _ik = ik_transform.ArmIK()


def send_servos(servo_dict, duration=SERVO_DURATION):
    msg = MultiRawIdPosDur()
    items = []
    for sid, pos in servo_dict.items():
        item = RawIdPosDur()
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
    Solve Inverse Kinematics (IK) and move the arm to the target position.
    
    If step_down=True, the arm descends incrementally from the current Z to the 
    target Z level to prevent unexpected flips or changes in the IK solution branch.
    """
    if step_down and current_z is not None and current_z > z:
        # 0.02m 간격으로 단계적으로 내려가기
        steps = int((current_z - z) / 0.02)
        for i in range(1, steps + 1):
            mid_z = current_z - i * 0.02
            target = _ik.setPitchRanges((x, y, mid_z), PITCH, PITCH_MIN, PITCH_MAX)
            if target:
                sd = target[1]
                send_servos({2: SERVO2_DEFAULT, 3: sd['servo3'],
                             4: sd['servo4'],  5: sd['servo5'], 6: sd['servo6']})
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
        rospy.logwarn(f'  No valid IK solution: {label}  ({x:.3f}, {y:.3f}, {z:.3f})')
        return False


# ── Pick & Place sequence ───────────────────────────────────────────────
def pick_and_place(px, py, pz, dx, dy, dz):
    """
    Pick up an object at coordinates (px, py, pz) and place it at (dx, dy, dz).
    Executes the full pick-and-place sequence including approach, grasp, 
    lift, transfer, and release phases.
    """
    rospy.loginfo(f'=== PICK  ({px:.3f}, {py:.3f}, {pz:.3f})')
    rospy.loginfo(f'=== PLACE ({dx:.3f}, {dy:.3f}, {dz:.3f})')

# ── PICK PHASE ───────────────────────────────────────────────────
    # 1. Open gripper
    set_gripper(GRIPPER_OPEN)
    rospy.sleep(GRIP_SLEEP)

    # 2. Move to approach height above the object
    if not move_to(px, py, Z_APPROACH, 'approach'):
        rospy.logerr('Pick aborted: approach IK failed')
        return False
    rospy.sleep(MOVE_SLEEP)

    # 3. Descend incrementally to grasp height (prevents IK branch configuration flip)
    if not move_to(px, py, pz, 'grasp', step_down=True, current_z=Z_APPROACH):
        rospy.logerr('Pick aborted: grasp IK failed')
        return False
    rospy.sleep(MOVE_SLEEP)

    # 4. Close gripper (Grasp the object)
    set_gripper(GRIPPER_CLOSE)
    rospy.sleep(GRIP_SLEEP)

    # 5. Lift the object up
    if not move_to(px, py, Z_APPROACH, 'lift'):
        rospy.logerr('Lift failed — gripper may still hold object')
    rospy.sleep(MOVE_SLEEP)

    # ── PLACE PHASE ──────────────────────────────────────────────────
    # 6. Move to approach height above the drop position
    if not move_to(dx, dy, max(dz + 0.05, Z_APPROACH), 'place_approach'):
        rospy.logerr('Place aborted: approach IK failed')
        return False
    rospy.sleep(MOVE_SLEEP)

    # 7. Lower the arm to the final drop height
    if not move_to(dx, dy, dz, 'place'):
        rospy.logerr('Place aborted: place IK failed')
        return False
    rospy.sleep(MOVE_SLEEP)

    # 8. Open gripper (Release the object)
    set_gripper(GRIPPER_OPEN)
    rospy.sleep(GRIP_SLEEP)

    # 9. Return to HOME position
    move_to(*HOME, 'home')
    rospy.sleep(MOVE_SLEEP)

    rospy.loginfo('=== Pick & Place DONE ===')
    return True


# ── ROS callback and state management ─────────────────────────────────────────────
_busy = False
_drop_pos = list(DEFAULT_DROP)   # can update with place_target


def pick_target_cb(msg):
    """
    Callback invoked upon receiving a message on /pick_target.
    Uses the default Z_GRASP value if msg.z is close to zero.
    """
    global _busy
    if _busy:
        rospy.logwarn('Pick already in progress, ignoring new target')
        return

    px, py = msg.x, msg.y
    # Use msg.z if it's within a valid range, otherwise fall back to default Z_GRASP
    pz = msg.z if msg.z > -0.020 else Z_GRASP
    dx, dy, dz = _drop_pos

    _busy = True
    try:
        pick_and_place(px, py, pz, dx, dy, dz)
    finally:
        _busy = False


def place_target_cb(msg):
    """
    Callback invoked upon receiving a message on /place_target to update the drop position.
    Uses the default drop height if msg.z is close to zero.
    """
    global _drop_pos
    _drop_pos = [
        msg.x,
        msg.y,
        # Fall back to DEFAULT_DROP height if msg.z is near zero or invalid
        msg.z if msg.z > 0.001 else DEFAULT_DROP[2]
    ]
    rospy.loginfo(f'Drop position updated: {_drop_pos}')

# ── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rospy.init_node('auto_pick', anonymous=True)

    servo_pub = rospy.Publisher(
        '/servo_controllers/port_id_1/multi_id_pos_dur',
        MultiRawIdPosDur, queue_size=1)
    rospy.sleep(0.5)

    init(servo_pub)

    # move to HOME location when initializing 
    rospy.loginfo('Auto pick node ready. Moving to HOME...')
    move_to(*HOME, 'home')
    rospy.sleep(1.5)

    rospy.Subscriber('/pick_target',  Point, pick_target_cb)
    rospy.Subscriber('/place_target', Point, place_target_cb)

    rospy.loginfo('Waiting for /pick_target  (geometry_msgs/Point x y z)')
    rospy.loginfo('Optional: /place_target to update drop position')
    rospy.spin()