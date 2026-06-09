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

import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
from hiwonder_servo_msgs.msg import MultiRawIdPosDur, RawIdPosDur
from chassis_control.msg import SetVelocity
from std_msgs.msg import Float64, String, Bool
from cv_bridge import CvBridge


_pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_pkg_path, 'armpi_pro_kinematics'))
sys.path.insert(0, '/home/ee478_team1/catkin_ws/src/armpi_pro_kinematics')
from kinematics import ik_transform


# ════════════════════════════════════════════════════════════════════════════
# ── Arm Parameters ───────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
Z_APPROACH     = 0.12       # Approach height (m)
Z_GRASP        = -0.017     # Grasp height (m) — lower than floor approach; tune in 5mm steps
MOVE_SLEEP     = 1.2        # Dwell time after each motion (s)
GRIP_SLEEP     = 0.6        # Dwell time after gripper actuation (s)

SERVO_DURATION = 800        # Servo travel time (ms)
GRIPPER_OPEN   = 200        # Servo 1 pulse width (open)
GRIPPER_CLOSE  = 550            # Servo 1 pulse width (closed) <- increased for tighter grip
SERVO2_DEFAULT = 500        # Fixed pulse value for joint 5

PITCH          = -90        # End-effector pitch angle (deg) — near-horizontal, scooping approach
PITCH_MIN      = -150       # IK search lower bound
PITCH_MAX      = -70        # IK search upper bound (= PITCH; fully horizontal)

HOME           = (0.00, 0.15, 0.10)    # Home position
GRASP_OFFSET_X = 0.000                 # Lateral offset correction (m) — tune if gripper misses left/right
GRASP_OFFSET_Y = 0.000                 # Forward offset correction (m) — tune if gripper misses front/back

# Post-grasp arm trajectory waypoints
LIFT_SLIGHT_Z       = 0.00        # First lift target after closing gripper (m)
WAYPOINT_LOW        = (0.000, 0.150, -0.000)   # Intermediate low  waypoint — clears the pick site
WAYPOINT_HIGH       = (0.000, 0.130,  0.160)   # Intermediate high waypoint — clears obstacles before place

# Chassis nudge at grasp height (executed after arm descends to Z_GRASP, before closing gripper)
# The robot pushes forward 1 cm at low speed so the gripper scoops under the object.
PICK_NUDGE_SPEED    = 30    # mm/s — slow push to avoid disturbing the object
PICK_NUDGE_DIST_MM  = 12    # mm   — forward nudge distance (1 cm)


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

FORWARD_SPEED     = 80      # Forward/backward full speed (mm/s)
STRAFE_SPEED      = 60      # Lateral strafe full speed (mm/s)
CREEP_SPEED       = 50      # Fine approach speed (mm/s)
CREEP_TIME        = 0.25    # Fine approach duration (s)

# Velocity deceleration profile
# Within DECEL_ZONE the chassis speed ramps linearly from full speed down to MIN_SPEED.
# This prevents overshoot caused by inertia when the alignment condition is first met.
DECEL_ZONE_Y      = 0.10    # Forward deceleration zone radius (m) — 5x TOL_Y
DECEL_ZONE_X      = 0.09    # Lateral deceleration zone radius (m) — 3x TOL_X
MIN_SPEED_FWD     = 30      # Minimum forward/backward speed (mm/s)
MIN_SPEED_STRAFE  = 25      # Minimum lateral strafe speed (mm/s)

VALID_Y_MIN       = -0.30   # Valid detection range lower bound
VALID_Y_MAX       =  0.60   # Valid detection range upper bound
DETECTION_TIMEOUT = 2.0     # Stop chassis if no detection for this duration (s)


# ════════════════════════════════════════════════════════════════════════════
# ── Grasp Verification Parameters ────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
VERIFY_POSE       = (0.000, 0.230, 0.020)   # Arm pose for depth check (m)
VERIFY_PITCH      = -180                     # End-effector pitch during verification
VERIFY_PITCH_MIN  = -185
VERIFY_PITCH_MAX  = -150
GRASP_DEPTH_THRES = 0.04    # Object detected if any pixel depth in patch < 4 cm
VERIFY_RADIUS     = 50      # Pixel sampling radius around expected gripper location
BACK_SPEED        = 60      # Chassis backup speed (mm/s)
BACK_TIME         = 0.83    # Backup duration (s)  ->  60mm/s x 0.83s ~ 50mm

# ── Linear calibration constants (mirror of yolo_detector.py) ────────────
#   arm_x = Ax*X_cam + Bx*Z_cam + Cx
#   arm_y = Ay*X_cam + By*Z_cam + Cy
CAL_Ax = 1.445;  CAL_Bx = -0.350;  CAL_Cx = 0.035
CAL_Ay = 0.015;  CAL_By = -0.316;  CAL_Cy = 0.299


# ════════════════════════════════════════════════════════════════════════════
# ── Object Position Kalman Filter ────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
class ObjectKalmanFilter:
    """
    2-D constant-position Kalman filter for arm-frame object coordinates.

    State  : x = [arm_x, arm_y]
    Process: constant-position model  (F = I)
    Measure: direct observation       (H = I)

    R  -- measurement noise covariance
          Inflated ~1000x from measured sensor noise (~1e-7) to absorb
          calibration residual error and YOLO bounding-box jitter.
          Corresponds to sigma ~= 10 mm per axis.

    Q  -- process noise covariance
          Allows the filter to track arm_point as the chassis approaches
          (arm_y decreases ~10-15 mm per YOLO frame at approach speed).
          Set equal to R -> steady-state Kalman gain K ~= 0.5,
          averaging 50 % new measurement and 50 % previous estimate.
    """

    R = np.diag([1e-4, 1e-4])   # measurement noise covariance  (sigma ~= 10 mm)
    Q = np.diag([1e-4, 1e-4])   # process noise covariance      (sigma ~= 10 mm)

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset filter state. Call when locking onto a new object (IDLE -> APPROACHING)."""
        self.x           = None                   # state vector [arm_x, arm_y]
        self.P           = np.diag([1.0, 1.0])   # large initial covariance -> trust first measurement fully
        self.initialized = False

    def update(self, arm_x, arm_y):
        """
        Ingest one YOLO measurement and return the filtered estimate.
        Returns (filtered_arm_x, filtered_arm_y).
        """
        z = np.array([arm_x, arm_y])

        if not self.initialized:
            self.x           = z.copy()
            self.initialized = True
            return arm_x, arm_y

        # Predict step (constant-position: state unchanged, covariance grows by Q)
        P_pred = self.P + self.Q

        # Update step
        S      = P_pred + self.R                  # innovation covariance
        K      = P_pred @ np.linalg.inv(S)        # Kalman gain (~0.5 at steady state)
        self.x = self.x + K @ (z - self.x)
        self.P = (np.eye(2) - K) @ P_pred

        return float(self.x[0]), float(self.x[1])


# ════════════════════════════════════════════════════════════════════════════
# ── Global State ─────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
state         = IDLE
last_detect_t = None
creep_start_t = None
last_arm_pt   = None        # Latest /yolo/arm_point message (Kalman-filtered)
_kf           = ObjectKalmanFilter()

# arm_pt snapshot taken at the APPROACHING -> CREEPING transition.
# This is the best-quality Y estimate: alignment just passed, chassis not yet
# moved by CREEP.  pick_and_place() subtracts the known creep distance from
# this value so the pick Y is independent of YOLO noise during/after creeping.
_locked_arm_pt = None

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

# Authority gate: the orchestrator grants chassis/pick control by activating the
# grab YOLO (/yolo_grab/activate True) and RECLAIMS it by deactivating (False).
# While inactive, pick_arm must NOT touch the chassis so the navigator can drive
# to the next store.
grab_active       = False
_grab_active_prev = False

# ── Camera intrinsics (updated from /camera/color/camera_info) ───────────
_cam_fx = 917.3;  _cam_fy = 915.3   # focal length (px) — default from prior log
_cam_cx = 642.8;  _cam_cy = 356.2   # principal point (px)
_cam_intrinsic_ready = False
_cv_bridge = CvBridge()


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


def camera_info_cb(msg):
    """Latch camera intrinsics from /camera/color/camera_info on first message."""
    global _cam_fx, _cam_fy, _cam_cx, _cam_cy, _cam_intrinsic_ready
    if not _cam_intrinsic_ready:
        _cam_fx = msg.K[0]
        _cam_fy = msg.K[4]
        _cam_cx = msg.K[2]
        _cam_cy = msg.K[5]
        _cam_intrinsic_ready = True
        rospy.loginfo(
            f'[pick_arm] Camera intrinsics: '
            f'fx={_cam_fx:.1f} fy={_cam_fy:.1f} cx={_cam_cx:.1f} cy={_cam_cy:.1f}')


def arm_to_pixel(arm_x, arm_y):
    """
    Inverse of the linear calibration transform (yolo_detector):
      arm frame (arm_x, arm_y) -> camera frame (X_cam, Z_cam) -> pixel (u, v)

    Y_cam is not included in the calibration, so v defaults to the principal point cy.
    Returns (u, v) or None if Z_cam <= 0.
    """
    det   = CAL_Ax * CAL_By - CAL_Bx * CAL_Ay
    X_cam = (CAL_By * (arm_x - CAL_Cx) - CAL_Bx * (arm_y - CAL_Cy)) / det
    Z_cam = (CAL_Ax * (arm_y - CAL_Cy) - CAL_Ay * (arm_x - CAL_Cx)) / det
    if Z_cam <= 0:
        return None
    u = int(X_cam * _cam_fx / Z_cam + _cam_cx)
    v = int(_cam_cy)
    return u, v


def verify_grasp_depth():
    """
    Depth-based grasp verification.
      1. Back chassis ~5 cm.
      2. Move arm to VERIFY_POSE with pitch = -180.
      3. Sample depth around expected end-effector pixel.
      4. Return True if min depth in patch < GRASP_DEPTH_THRES (4 cm).

    After this call the chassis is ~5 cm behind the pick position and the
    arm is at VERIFY_POSE — pick_and_place() must handle lift from there.
    """
    rospy.loginfo('[pick_arm] Grasp verify: backing up ~5 cm...')
    chassis_cmd(BACK_SPEED, 270, 0)   # 270 deg = backward
    rospy.sleep(BACK_TIME)
    chassis_stop()

    # Move arm to verification pose (pitch = -180, arm pointing up toward camera)
    rospy.loginfo('[pick_arm] Grasp verify: moving arm to verify pose...')
    vx, vy, vz = VERIFY_POSE
    target = _ik.setPitchRanges((vx, vy, vz), VERIFY_PITCH, VERIFY_PITCH_MIN, VERIFY_PITCH_MAX)
    if target:
        sd = target[1]
        send_servos({
            2: SERVO2_DEFAULT,
            3: sd['servo3'],
            4: sd['servo4'],
            5: sd['servo5'],
            6: sd['servo6'],
        })
    else:
        rospy.logwarn('[pick_arm] Grasp verify: no IK solution for verify pose — assuming PASS')
        return True
    rospy.sleep(1.2)   # wait for arm to settle

    # Compute expected pixel location of end effector
    uv = arm_to_pixel(vx, vy)
    if uv is None:
        rospy.logwarn('[pick_arm] Grasp verify: invalid pixel projection — assuming PASS')
        return True
    u, v = uv
    rospy.loginfo(f'[pick_arm] Grasp verify: sampling depth at pixel ({u}, {v}) r={VERIFY_RADIUS}')

    # Grab one aligned depth frame
    try:
        depth_msg = rospy.wait_for_message(
            '/camera/aligned_depth_to_color/image_raw', Image, timeout=2.0)
    except rospy.ROSException:
        rospy.logwarn('[pick_arm] Grasp verify: depth timeout — assuming PASS')
        return True

    depth = _cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
    h, w  = depth.shape

    # Sample patch around expected gripper pixel
    u0, u1 = max(0, u - VERIFY_RADIUS), min(w, u + VERIFY_RADIUS + 1)
    v0, v1 = max(0, v - VERIFY_RADIUS), min(h, v + VERIFY_RADIUS + 1)
    patch  = depth[v0:v1, u0:u1].astype(np.float32)
    valid  = patch[patch > 0]

    if len(valid) == 0:
        rospy.logwarn('[pick_arm] Grasp verify: no valid depth pixels in patch — assuming PASS')
        return True

    # Auto-detect scale: 1.0 for float32 depth (meters, e.g. simulation), 0.001 for uint16 (mm, real robot)
    scale = 1.0 if depth.dtype == np.float32 else 0.001
    min_depth = float(np.min(valid)) * scale
    rospy.loginfo(
        f'[pick_arm] Grasp verify: min_depth={min_depth*100:.1f} cm  '
        f'threshold={GRASP_DEPTH_THRES*100:.0f} cm')

    if min_depth < GRASP_DEPTH_THRES:
        rospy.loginfo('[pick_arm] Grasp verify: PASS — object detected in gripper')
        return True
    else:
        rospy.logwarn('[pick_arm] Grasp verify: FAIL — gripper appears empty')
        return False


def set_final_pose():
    """
    Move arm to final place pose using absolute joint angles (rad), then open gripper.
    Interpolates slowly over 4 seconds to avoid fast snapping.
    """
    import sensor_msgs.msg
    rospy.loginfo('[pick_arm] Moving to final place pose (slowly)...')
    target_j = [0.0, -0.4, 1.0, 1.0, 0.0]
    
    current_j = None
    try:
        js = rospy.wait_for_message('/joint_states', sensor_msgs.msg.JointState, timeout=1.0)
        current_j = [0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(1, 6):
            j_name = 'joint' + str(i)
            if j_name in js.name:
                idx = js.name.index(j_name)
                current_j[i-1] = js.position[idx]
    except Exception as e:
        rospy.logwarn('Could not get joint_states')
    
    if current_j is not None:
        steps = 40
        delay = 4.0 / steps
        for step in range(1, steps + 1):
            fraction = step / float(steps)
            for i in range(5):
                val = current_j[i] + fraction * (target_j[i] - current_j[i])
                _joint_pubs[i+1].publish(Float64(data=val))
            rospy.sleep(delay)
    else:
        for i in range(5):
            _joint_pubs[i+1].publish(Float64(data=target_j[i]))
        rospy.sleep(4.0)
    
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


def move_to(x, y, z, label='', step_down=False, current_z=None, duration=SERVO_DURATION):
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
                             4: sd['servo4'],   5: sd['servo5'], 6: sd['servo6']}, duration=duration)
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
        }, duration=duration)
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
    
    # Apply manual grasp offsets to correct pose estimation errors
    px += GRASP_OFFSET_X
    py += GRASP_OFFSET_Y
    
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

    # 3.5. Chassis nudge: push 1 cm forward at grasp height so gripper scoops under object.
    #      Arm stays at Z_GRASP during nudge — chassis brings gripper the final distance.
    nudge_dur = PICK_NUDGE_DIST_MM / PICK_NUDGE_SPEED   # e.g. 10mm / 30mm/s = 0.33 s
    rospy.loginfo(
        f'[pick_arm] Step 3.5: nudge chassis fwd {PICK_NUDGE_DIST_MM}mm '
        f'at {PICK_NUDGE_SPEED}mm/s ({nudge_dur:.2f}s)')
    chassis_cmd(PICK_NUDGE_SPEED, 90, 0)                # 90 deg = forward
    rospy.sleep(nudge_dur)
    chassis_stop()
    rospy.sleep(0.3)                                     # settle before gripping

    # 4. Close gripper (grasp object)
    set_gripper(GRIPPER_CLOSE)
    rospy.sleep(GRIP_SLEEP)

    # 5. Slight lift at pick position (grasp height -> approach height)
    rospy.loginfo('[pick_arm] Step 5: slight lift at pick position')
    if not move_to(px, py, LIFT_SLIGHT_Z, 'lift_slight', duration=2500):
        rospy.logerr('Slight lift failed — gripper may still be holding object')
    rospy.sleep(2.0)

    # 6. Move to low intermediate waypoint (x=0, y=0.110, z=-0.010)
    #    Pulls the arm inward and low to clear the pick site before raising.
    rospy.loginfo(f'[pick_arm] Step 6: move to low waypoint {WAYPOINT_LOW}')
    if not move_to(*WAYPOINT_LOW, 'waypoint_low'):
        rospy.logwarn('[pick_arm] Low waypoint IK failed — continuing')
    rospy.sleep(MOVE_SLEEP)

    # # 7. Raise to high intermediate waypoint (x=0, y=0.110, z=+0.160)
    # #    Lifts the object clear of obstacles before the place phase.
    # rospy.loginfo(f'[pick_arm] Step 7: raise to high waypoint {WAYPOINT_HIGH}')
    # if not move_to(*WAYPOINT_HIGH, 'waypoint_high'):
    #     rospy.logwarn('[pick_arm] High waypoint IK failed — continuing')
    # rospy.sleep(MOVE_SLEEP)

    # 8. Back up chassis before final placement
    rospy.loginfo('[pick_arm] Step 8: backing up chassis...')
    chassis_cmd(velocity=60, direction=270, angular=0)   # 270 deg = backward
    rospy.sleep(1.0)
    chassis_stop()

    # ── PLACE PHASE ───────────────────────────────────────────────────────
    # 9. Move to place pose via joint controllers and release object
    set_final_pose()

    # 10. Return to HOME position slowly
    rospy.loginfo('[pick_arm] Step 10: returning to HOME position...')
    move_to(*HOME, 'home_return', duration=3000)
    rospy.sleep(3.5)

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
    Receive /yolo/arm_point, apply range filter, run through Kalman filter,
    and store the filtered estimate as last_arm_pt.
    """
    global last_detect_t, last_arm_pt
    if not (VALID_Y_MIN < msg.y < VALID_Y_MAX):
        return

    fx, fy = _kf.update(msg.x, msg.y)

    filtered   = Point()
    filtered.x = fx
    filtered.y = fy
    filtered.z = msg.z      # z is a fixed grasp height — pass through unchanged

    last_arm_pt   = filtered
    last_detect_t = rospy.Time.now()

def target_item_cb(msg):
    global target_item
    target_item = msg.data
    rospy.loginfo(f'[pick_arm] Target item updated: {target_item}')


def grab_active_cb(msg):
    """Orchestrator's grant/reclaim of chassis-pick authority."""
    global grab_active
    grab_active = bool(msg.data)
    rospy.loginfo(f'[pick_arm] grab authority -> {grab_active}')


# ════════════════════════════════════════════════════════════════════════════
# ── Control Loop  10 Hz ──────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def control_loop(event):
    """
    rospy.Timer callback at 10 Hz.
    Handles state machine transitions and chassis commands.
    In ALIGNED state, calls pick_and_place() directly (blocking) then returns to IDLE.
    """
    global state, creep_start_t, target_visible, _locked_arm_pt, _grab_active_prev

    now = rospy.Time.now()

    # Update target_visible state
    if last_detect_t is None:
        target_visible = False
    else:
        dt = (now - last_detect_t).to_sec()
        target_visible = (dt < DETECTION_TIMEOUT)

    if target_visible_pub is not None:
        target_visible_pub.publish(Bool(target_visible))

    # ── Authority gate ────────────────────────────────────────────────────
    # Only command the chassis while the orchestrator has granted control
    # (grab YOLO active). The instant it reclaims control (grab off), stop ONCE
    # and then stay completely silent on the chassis so the navigator can drive
    # to the next store. Without this, pick_arm keeps holding the chassis and the
    # robot can't move on when the item isn't there.
    if not grab_active:
        if _grab_active_prev:
            chassis_stop()
            state = IDLE
            creep_start_t = None
            rospy.loginfo('[pick_arm] authority reclaimed -> releasing chassis (IDLE)')
        _grab_active_prev = False
        return
    _grab_active_prev = True

    # ── Detection timeout check ───────────────────────────────────────────
    if not target_visible:
        if state in [APPROACHING, CREEPING]:
            rospy.logwarn_throttle(3, '[pick_arm] No detection — stopping chassis')
            chassis_stop()
            state = IDLE
        if heartbeat_pub is not None and state == IDLE:
            if target_item not in ["None", ""]:
                heartbeat_pub.publish(String("could not detect"))
            else:
                heartbeat_pub.publish(String("idle"))
        return

    # ── IDLE ──────────────────────────────────────────────────────────────
    if state == IDLE:
        rospy.loginfo('[pick_arm] Object detected -> APPROACHING')
        _kf.reset()   # reset Kalman filter for new object lock-in
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
            f'arm(kf)=({pt.x:.3f},{pt.y:.3f})  '
            f'P=({_kf.P[0,0]:.2e},{_kf.P[1,1]:.2e})  '
            f'err=({err_x:+.3f},{err_y:+.3f})  '
            f'aligned=({aligned_x},{aligned_y})')

        if aligned_x and aligned_y:
            # Lock arm_pt now — highest alignment quality, chassis not yet moved
            _locked_arm_pt = last_arm_pt
            rospy.loginfo(
                f'[pick_arm] ALIGNED -> CREEPING '
                f'({CREEP_SPEED}mm/s x {CREEP_TIME}s ~ {CREEP_SPEED*CREEP_TIME:.0f}mm)  '
                f'locked y={_locked_arm_pt.y:.3f}')
            chassis_cmd(CREEP_SPEED, 90, 0)
            state = CREEPING
        else:
            # Correct lateral (X) error first, then forward/backward (Y) error.
            # Rationale: strafing to fix X keeps the object at the same distance,
            # so it stays within the camera FOV.  Moving forward to fix Y while X
            # is off-center risks the object drifting to the edge of the FOV and
            # being lost by the detector.
            #
            # Speed profile: linear ramp-down inside DECEL_ZONE so inertia does
            # not carry the chassis past the alignment window.
            #   speed = max(MIN_SPEED, FULL_SPEED * |err| / DECEL_ZONE)
            # Outside DECEL_ZONE the formula saturates at FULL_SPEED naturally.

            if not aligned_x:
                # Compute proportional strafe speed
                spd = max(MIN_SPEED_STRAFE,
                          int(STRAFE_SPEED * min(1.0, abs(err_x) / DECEL_ZONE_X)))
                # err_x > 0 -> object to the right (+arm_x) -> strafe right (0 deg)  -> arm_x decreases
                # err_x < 0 -> object to the left  (-arm_x) -> strafe left  (180 deg) -> arm_x increases
                direction = 0 if err_x > 0 else 180
                chassis_cmd(spd, direction, 0)
                rospy.loginfo_throttle(0.3,
                    f'[pick_arm] STRAFE  spd={spd} mm/s  err_x={err_x:+.3f}')
            else:
                # X aligned — now close the distance with proportional forward speed
                spd = max(MIN_SPEED_FWD,
                          int(FORWARD_SPEED * min(1.0, abs(err_y) / DECEL_ZONE_Y)))
                # err_y < 0 -> arm_y < TARGET_Y -> object is far   -> forward  (90 deg)
                # err_y > 0 -> arm_y > TARGET_Y -> object is close -> backward (270 deg)
                direction = 270 if err_y > 0 else 90
                chassis_cmd(spd, direction, 0)
                rospy.loginfo_throttle(0.3,
                    f'[pick_arm] FORWARD spd={spd} mm/s  err_y={err_y:+.3f}')

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
        if _locked_arm_pt is not None:
            # Y: use TARGET_Y directly.
            #    The alignment controller drives arm_y to TARGET_Y ± TOL_Y before
            #    CREEPING, so the object is at TARGET_Y when CREEP starts.
            #    Using locked_y underestimates by ~KF_lag (K=0.5 tracking lag)
            #    and subtracting creep_dist compounds the error — together causing
            #    ~20 mm undershoot.  TARGET_Y is deterministic and noise-free.
            #    If there is a residual offset after testing, tune GRASP_OFFSET_Y.
            px = last_arm_pt.x if last_arm_pt is not None else _locked_arm_pt.x
            py = TARGET_Y
            pz = Z_GRASP
            rospy.loginfo(
                f'[pick_arm] Starting pick  '
                f'locked_y={_locked_arm_pt.y:.3f}  target_y={TARGET_Y:.3f}  '
                f'pick=({px:.3f}, {py:.3f}, {pz:.3f})')
            pick_and_place(px, py, pz)   # blocking direct call
        creep_start_t  = None
        _locked_arm_pt = None
        state          = IDLE
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
    rospy.Subscriber('/yolo/arm_point',              Point,      arm_point_cb)
    rospy.Subscriber('/target_item',                 String,     target_item_cb)
    rospy.Subscriber('/yolo_grab/activate',          Bool,       grab_active_cb)
    rospy.Subscriber('/camera/color/camera_info',    CameraInfo, camera_info_cb)

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