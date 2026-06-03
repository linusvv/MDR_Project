#!/usr/bin/python3
# coding=utf8
"""
ArmPi Pro Keyboard Teleoperation v2
─────────────────────────────────────────────────────────────────────────────
- Chassis : Omni-directional movement via SetVelocity
            (Enforced at a low speed to prevent Raspberry Pi brownout)
- Arm     : Inverse Kinematics (IK)-based X/Y/Z Cartesian control via MultiRawIdPosDur
- Gripper : Direct servo pulse control (Servo ID: 1)

Key Mapping:
  Chassis Controls:
    W / S        : Move Forward / Backward
    A / D        : Strafe Left / Right
    Q / E        : Rotate Counter-Clockwise (Left) / Clockwise (Right)
    Space        : Emergency Stop / Halt Chassis

  Robot Arm Cartesian Controls:
    I / K        : Move Arm Forward / Backward (+/- Y-axis)
    J / L        : Move Arm Left / Right      (+/- X-axis)
    U / O        : Move Arm Up / Down         (+/- Z-axis)

  Gripper Controls:
    G / H        : Open / Close Gripper
    V            : Verify Grasp (print gripper state + pass/fail)

  System:
    Ctrl + C     : Terminate Program / Exit
"""

import sys
import tty
import termios
import rospy
from chassis_control.msg import SetVelocity
from hiwonder_servo_msgs.msg import MultiRawIdPosDur, RawIdPosDur
from hiwonder_servo_msgs.msg import JointState as ServoJointState

# IK import
import sys as _sys
_sys.path.insert(0, '/home/ee478_team1/catkin_ws/src/armpi_pro_kinematics')
try:
    from kinematics import ik_transform
    ik = ik_transform.ArmIK()
    USE_IK = True
except Exception as e:
    USE_IK = False
    print('[WARN] IK not available:', e)

# ── Parameters ────────────────────────────────────────────────────────
CHASSIS_SPEED   = 60     # Linear speed (mm/s)  <- Keep low to prevent Raspberry Pi brownout
CHASSIS_ANGULAR = 0.3    # Angular speed (rad/s) <- Keep low to maintain power stability

STEP_XY  = 0.01          # Incremental step size for arm movement (1cm per keypress)
PITCH    = -90           # End-effector pitch angle (degrees), fixed

# Initial Arm Configuration (meters) – ArmPi Pro default forward posture
ARM_X_INIT = 0.00        # Left/Right (Positive = Robot's Left)
ARM_Y_INIT = 0.21        # Forward/Backward (Positive = Robot's Front)
ARM_Z_INIT = 0.14        # Height (meters)

# Servo Actuation Values
SERVO_DURATION = 500     # Servo travel time (milliseconds)
GRIPPER_OPEN   = 200     # Servo 1 pulse width (Open)
GRIPPER_CLOSE  = 500     # Servo 1 pulse width (Closed)
SERVO2_DEFAULT = 520     # Fixed pulse value for Joint 5

# Robot Arm Workspace Bounds / Operational Limits (meters)
X_MIN, X_MAX = -2, 2
Y_MIN, Y_MAX =  -10, 10
Z_MIN, Z_MAX =  -10, 10

# Grasp verification threshold — same value as pick_arm_node.py
# Tune this until verify_grasp() reliably detects held objects
GRASP_ERROR_THRESHOLD = 0.1   # rad

HELP = """
╔══════════════════════════════════════════════╗
║      ArmPi Pro  Keyboard Teleop  v2          ║
╠══════════════════════════════════════════════╣
║  CHASSIS CONTROLS                            ║
║    W / S   : Move Forward / Backward         ║
║    A / D   : Strafe Left / Right             ║
║    Q / E   : Rotate CCW / CW                 ║
║    Space   : Emergency Stop / Halt           ║
╠══════════════════════════════════════════════╣
║  ARM CONTROLS (IK Cartesian)                 ║
║    I / K   : Move Forward / Backward (+/- Y) ║
║    J / L   : Move Left / Right      (+/- X)  ║
║    U / O   : Move Up / Down         (+/- Z)  ║
╠══════════════════════════════════════════════╣
║  GRIPPER                                     ║
║    G / H   : Open / Close Gripper            ║
║    V       : Verify Grasp (print state)      ║
╠══════════════════════════════════════════════╣
║    Ctrl+C  : Exit Program                    ║
╚══════════════════════════════════════════════╝
"""


class ArmPiTeleop:
    def __init__(self):
        rospy.init_node('armpi_teleop', anonymous=True)

        self.vel_pub = rospy.Publisher(
            '/chassis_control/set_velocity', SetVelocity, queue_size=1)
        self.servo_pub = rospy.Publisher(
            '/servo_controllers/port_id_1/multi_id_pos_dur',
            MultiRawIdPosDur, queue_size=1)

        # Current arm Cartesian position
        self.ax = ARM_X_INIT
        self.ay = ARM_Y_INIT
        self.az = ARM_Z_INIT

        # Latest gripper state from /r_joint_controller/state
        self._gripper_state = None
        rospy.Subscriber('/r_joint_controller/state', ServoJointState,
                         self._gripper_state_cb)

        rospy.sleep(0.5)

    # ── Gripper state callback ────────────────────────────────────────────
    def _gripper_state_cb(self, msg):
        self._gripper_state = msg

    # ── Chassis ──────────────────────────────────────────────────────────
    def chassis(self, velocity=0.0, direction=90.0, angular=0.0):
        msg = SetVelocity()
        msg.velocity  = float(velocity)
        msg.direction = float(direction)
        msg.angular   = float(angular)
        self.vel_pub.publish(msg)

    # ── Servo direct control ──────────────────────────────────────────────
    def send_servos(self, servo_dict):
        """servo_dict: {servo_id(int): pulse(0~1000)}"""
        msg = MultiRawIdPosDur()
        items = []
        for sid, pos in servo_dict.items():
            item = RawIdPosDur()
            item.id       = int(sid)
            item.position = int(max(0, min(1000, pos)))
            item.duration = SERVO_DURATION
            items.append(item)
        msg.id_pos_dur_list = items
        self.servo_pub.publish(msg)

    def set_gripper(self, pulse):
        self.send_servos({1: pulse})

    # ── IK arm movement ───────────────────────────────────────────────────
    def move_arm(self, x, y, z):
        x = max(X_MIN, min(X_MAX, x))
        y = max(Y_MIN, min(Y_MAX, y))
        z = max(Z_MIN, min(Z_MAX, z))

        if not USE_IK:
            print('\r  [IK unavailable]                    ', end='', flush=True)
            return

        target = ik.setPitchRanges((x, y, z), PITCH, -180, 0)
        if target:
            sd = target[1]
            self.send_servos({
                2: SERVO2_DEFAULT,
                3: sd['servo3'],
                4: sd['servo4'],
                5: sd['servo5'],
                6: sd['servo6'],
            })
            self.ax, self.ay, self.az = x, y, z
            print(f'\r  Arm  x={x:+.3f}  y={y:+.3f}  z={z:+.3f}m     ',
                  end='', flush=True)
        else:
            print(f'\r  [IK no solution]  x={x:+.3f}  y={y:+.3f}  z={z:+.3f}',
                  end='', flush=True)

    # ── Grasp verification ────────────────────────────────────────────────
    def verify_grasp(self):
        """
        Print gripper state and evaluate whether an object is being held.
        Uses same logic as pick_arm_node.verify_grasp().

        Workflow:
          1. Close gripper  (H key)
          2. Press V        -> prints goal / current / |error| and PASS/FAIL
          3. Tune GRASP_ERROR_THRESHOLD until result matches reality
        """
        # Raw terminal mode requires \r\n for newlines
        def rprint(s):
            print(f'\r{s}', flush=True)

        if self._gripper_state is None:
            rprint('  [GRASP CHECK] No state received from /r_joint_controller/state')
            rprint('  -> Check ROS network (ROS_IP / ROS_HOSTNAME)')
            return

        s     = self._gripper_state
        error = abs(s.error)

        rprint('  ┌─── Grasp Verification ───────────────────────┐')
        rprint(f'  │  goal_pos    : {s.goal_pos:+.4f} rad')
        rprint(f'  │  current_pos : {s.current_pos:+.4f} rad')
        rprint(f'  │  |error|     : {error:.4f} rad')
        rprint(f'  │  threshold   : {GRASP_ERROR_THRESHOLD} rad')

        if error > GRASP_ERROR_THRESHOLD:
            rprint(f'  │  Result      : PASS  ✓  (object detected)')
        else:
            rprint(f'  │  Result      : FAIL  ✗  (gripper fully closed)')

        rprint('  └──────────────────────────────────────────────┘')

    # ── Main loop ─────────────────────────────────────────────────────────
    def run(self):
        print(HELP)
        if USE_IK:
            print('  IK: ON  ->  moving to initial position...')
            self.move_arm(self.ax, self.ay, self.az)
        else:
            print('  IK: OFF  (check armpi_pro_kinematics build/source)')
        return
        # fd  = sys.stdin.fileno()
        # old = termios.tcgetattr(fd)
        # try:
        #     tty.setraw(fd)
        #     while not rospy.is_shutdown():
        #         key = sys.stdin.read(1)

        #         # Chassis
        #         if   key == 'w':  self.chassis(CHASSIS_SPEED,  90, 0)
        #         elif key == 's':  self.chassis(CHASSIS_SPEED, 270, 0)
        #         elif key == 'a':  self.chassis(CHASSIS_SPEED, 180, 0)
        #         elif key == 'd':  self.chassis(CHASSIS_SPEED,   0, 0)
        #         elif key == 'q':  self.chassis(0, 90,  CHASSIS_ANGULAR)
        #         elif key == 'e':  self.chassis(0, 90, -CHASSIS_ANGULAR)
        #         elif key == ' ':  self.chassis(0, 90, 0)

        #         # Arm IK
        #         elif key == 'i':  self.move_arm(self.ax, self.ay + STEP_XY, self.az)
        #         elif key == 'k':  self.move_arm(self.ax, self.ay - STEP_XY, self.az)
        #         elif key == 'j':  self.move_arm(self.ax - STEP_XY, self.ay, self.az)
        #         elif key == 'l':  self.move_arm(self.ax + STEP_XY, self.ay, self.az)
        #         elif key == 'u':  self.move_arm(self.ax, self.ay, self.az + STEP_XY)
        #         elif key == 'o':  self.move_arm(self.ax, self.ay, self.az - STEP_XY)

        #         # Gripper
        #         elif key == 'g':
        #             self.set_gripper(GRIPPER_OPEN)
        #             print('\r  Gripper: OPEN            ', flush=True)
        #         elif key == 'h':
        #             self.set_gripper(GRIPPER_CLOSE)
        #             print('\r  Gripper: CLOSE           ', flush=True)

        #         # Grasp verification
        #         elif key == 'v':
        #             self.verify_grasp()

        #         # Quit
        #         elif key == '\x03':
        #             break

        # finally:
        #     termios.tcsetattr(fd, termios.TCSADRAIN, old)
        #     self.chassis(0, 90, 0)
        #     print('\n  halt. Teleop shutdown.')


if __name__ == '__main__':
    ArmPiTeleop().run()
