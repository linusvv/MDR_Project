#!/usr/bin/python3
# coding=utf8
"""
Pick Arm Node  (auto_pick + pick_controller 통합)
─────────────────────────────────────────────────────────────────────────────
[pick_controller 역할]
  /yolo/arm_point (geometry_msgs/Point) 구독
  → 섀시(chassis)를 전진/스트레이프 하여 물체를 TARGET 위치로 정렬
  → 정렬 완료 시 pick_and_place() 를 별도 스레드로 직접 실행

[auto_pick 역할]
  IK 풀고 서보 명령을 순서대로 실행해 집기(pick) & 내려놓기(place) 수행

Coordinate System:
  arm_x : 좌(-)/우(+)   (로봇 기준 왼쪽이 +)
  arm_y : 전(+)/후(-)   (클수록 멀리)
  arm_z : 상(+)/하(-)

Chassis SetVelocity:
  velocity  : mm/s
  direction : 0~360°  (90°=전진, 270°=후진, 0°=우측, 180°=좌측)
  angular   : rad/s   (양수=반시계=좌회전)

Optional topic:
  /place_target (geometry_msgs/Point) : 내려놓을 위치 실시간 업데이트
"""

import os
import sys

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
Z_APPROACH     = 0.12       # 접근 높이 (m)
Z_GRASP        = -0.015     # 파지 높이 (m)  ← 실측에 맞게 튜닝
MOVE_SLEEP     = 1.2        # 각 동작 후 대기 (s)
GRIP_SLEEP     = 0.6        # 그리퍼 동작 후 대기 (s)

SERVO_DURATION = 800        # 서보 이동 시간 (ms)
GRIPPER_OPEN   = 200        # 서보1 펄스 (열기)
GRIPPER_CLOSE  = 500        # 서보1 펄스 (닫기) ← 물체 크기에 맞게 조정
SERVO2_DEFAULT = 500        # Joint5 고정값

PITCH          = -90        # 엔드이펙터 피치 (deg)
PITCH_MIN      = -150
PITCH_MAX      = -30

HOME           = (0.00, 0.15, 0.12)    # 홈 위치
DEFAULT_DROP   = (0.12, 0.15, 0.12)    # 기본 내려놓기 위치


# ════════════════════════════════════════════════════════════════════════════
# ── Controller Parameters (pick_controller) ──────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
IDLE        = 'IDLE'
APPROACHING = 'APPROACHING'
CREEPING    = 'CREEPING'
ALIGNED     = 'ALIGNED'

TARGET_Y          = 0.250   # 목표 전후 거리 (m) — 이 위치에서 pick 실행
TARGET_X          = 0.000   # 목표 좌우 위치 (m)
TOL_X             = 0.030   # 좌우 허용 오차 ±3 cm
TOL_Y             = 0.020   # 전후 허용 오차 ±2 cm

FORWARD_SPEED     = 80      # 전후 이동 속도 (mm/s)
STRAFE_SPEED      = 60      # 좌우 스트레이프 속도 (mm/s)
CREEP_SPEED       = 50      # 미세 접근 속도 (mm/s)
CREEP_TIME        = 0.6     # 미세 접근 시간 (s)  →  50mm/s × 0.6s ≈ 3 cm

VALID_Y_MIN       = -0.30   # 유효 감지 범위 하한
VALID_Y_MAX       =  0.60   # 유효 감지 범위 상한
DETECTION_TIMEOUT = 2.0     # 이 시간(s) 동안 감지 없으면 정지


# ════════════════════════════════════════════════════════════════════════════
# ── Global State ─────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
state         = IDLE
last_detect_t = None
creep_start_t = None
last_arm_pt   = None        # 최신 /yolo/arm_point

_drop_pos     = list(DEFAULT_DROP)

# ── Publisher handles ─────────────────────────────────────────────────────
_servo_pub = None
_ik        = None
vel_pub    = None


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
    IK를 풀고 서보 명령 전송.
    step_down=True 이면 현재 Z → 목표 Z 까지 0.02m 간격으로 단계 하강
    (IK 해 분기 변경 방지).
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
        rospy.logwarn(f'  IK 해 없음: {label}  ({x:.3f}, {y:.3f}, {z:.3f})')
        return False


def pick_and_place(px, py, pz, dx, dy, dz):
    """
    집기(px,py,pz) → 내려놓기(dx,dy,dz) 전체 시퀀스 실행.
    ALIGNED 상태에서 직접(블로킹) 호출됨.
    """
    rospy.loginfo(f'=== PICK  ({px:.3f}, {py:.3f}, {pz:.3f})')
    rospy.loginfo(f'=== PLACE ({dx:.3f}, {dy:.3f}, {dz:.3f})')

    # ── PICK PHASE ────────────────────────────────────────────────────────
    # 1. 그리퍼 열기
    set_gripper(GRIPPER_OPEN)
    rospy.sleep(GRIP_SLEEP)

    # 2. 접근 높이로 이동
    if not move_to(px, py, Z_APPROACH, 'approach'):
        rospy.logerr('Pick 중단: approach IK 실패')
        return False
    rospy.sleep(MOVE_SLEEP)

    # 3. 파지 높이로 단계 하강
    if not move_to(px, py, pz, 'grasp', step_down=True, current_z=Z_APPROACH):
        rospy.logerr('Pick 중단: grasp IK 실패')
        return False
    rospy.sleep(MOVE_SLEEP)

    # 4. 그리퍼 닫기 (파지)
    set_gripper(GRIPPER_CLOSE)
    rospy.sleep(GRIP_SLEEP)

    # 5. 들어올리기
    if not move_to(px, py, Z_APPROACH, 'lift'):
        rospy.logerr('Lift 실패 — 그리퍼가 물체를 잡고 있을 수 있음')
    rospy.sleep(MOVE_SLEEP)

    # ── PLACE PHASE ───────────────────────────────────────────────────────
    # 6. 내려놓기 위치 상공으로 이동
    if not move_to(dx, dy, max(dz + 0.05, Z_APPROACH), 'place_approach'):
        rospy.logerr('Place 중단: approach IK 실패')
        return False
    rospy.sleep(MOVE_SLEEP)

    # 7. 내려놓기 높이로 하강
    if not move_to(dx, dy, dz, 'place'):
        rospy.logerr('Place 중단: place IK 실패')
        return False
    rospy.sleep(MOVE_SLEEP)

    # 8. 그리퍼 열기 (해제)
    set_gripper(GRIPPER_OPEN)
    rospy.sleep(GRIP_SLEEP)

    # 9. HOME 복귀
    move_to(*HOME, 'home')
    rospy.sleep(MOVE_SLEEP)

    rospy.loginfo('=== Pick & Place 완료 ===')
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
    /yolo/arm_point 수신: 유효 범위 필터 후 글로벌 변수 갱신.
    """
    global last_detect_t, last_arm_pt
    if not (VALID_Y_MIN < msg.y < VALID_Y_MAX):
        return
    last_arm_pt   = msg
    last_detect_t = rospy.Time.now()


def place_target_cb(msg):
    """
    /place_target 수신: 내려놓기 위치 실시간 업데이트.
    """
    global _drop_pos
    _drop_pos = [
        msg.x,
        msg.y,
        msg.z if msg.z > 0.001 else DEFAULT_DROP[2],
    ]
    rospy.loginfo(f'[pick_arm] 내려놓기 위치 업데이트: {_drop_pos}')


# ════════════════════════════════════════════════════════════════════════════
# ── Control Loop  10 Hz (from pick_controller) ───────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def control_loop(event):
    """
    rospy.Timer 콜백 (10 Hz).
    상태 기계 전이 및 섀시 명령 발행.
    ALIGNED 상태에서 pick_and_place()를 블로킹 직접 호출 후 IDLE 복귀.
    """
    global state, creep_start_t

    now = rospy.Time.now()

    # ── 감지 타임아웃 확인 ────────────────────────────────────────────────
    if last_detect_t is None:
        if state == APPROACHING:
            rospy.logwarn_throttle(3, '[pick_arm] 감지 없음 — 섀시 정지')
            chassis_stop()
            state = IDLE
        return

    dt = (now - last_detect_t).to_sec()

    # ── IDLE ──────────────────────────────────────────────────────────────
    if state == IDLE:
        if dt < DETECTION_TIMEOUT:
            rospy.loginfo('[pick_arm] 물체 감지 → APPROACHING')
            state = APPROACHING

    # ── APPROACHING ───────────────────────────────────────────────────────
    elif state == APPROACHING:
        if dt >= DETECTION_TIMEOUT:
            rospy.logwarn('[pick_arm] 감지 손실 → IDLE')
            chassis_stop()
            state = IDLE
            return

        pt    = last_arm_pt
        err_x = pt.x - TARGET_X   # + : 물체가 왼쪽 → 로봇이 우측 이동 필요
        err_y = pt.y - TARGET_Y   # + : 물체가 멀리  → 전진 필요
                                   # - : 물체가 가까이 → 후진 필요

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
            # 전후 오차 우선 → 정렬 후 좌우 오차 보정
            if not aligned_y:
                # err_y < 0 → arm_y < TARGET_Y → 물체가 멀다 → 전진(90°)
                # err_y > 0 → arm_y > TARGET_Y → 물체가 가깝다 → 후진(270°)
                direction = 270 if err_y > 0 else 90
                chassis_cmd(FORWARD_SPEED, direction, 0)
            else:
                # err_x > 0 → 물체가 오른쪽(+arm_x) → 우측 이동(0°) → arm_x 감소
                # err_x < 0 → 물체가 왼쪽(-arm_x) → 좌측 이동(180°) → arm_x 증가
                direction = 0 if err_x > 0 else 180
                chassis_cmd(STRAFE_SPEED, direction, 0)

    # ── CREEPING ──────────────────────────────────────────────────────────
    elif state == CREEPING:
        if creep_start_t is None:
            creep_start_t = now
        elapsed = (now - creep_start_t).to_sec()
        if elapsed >= CREEP_TIME:
            chassis_stop()
            rospy.loginfo('[pick_arm] Creep 완료 → ALIGNED')
            state = ALIGNED

    # ── ALIGNED ───────────────────────────────────────────────────────────
    elif state == ALIGNED:
        if last_arm_pt is not None:
            pt = last_arm_pt
            px, py = pt.x, pt.y
            pz     = pt.z if pt.z > -0.020 else Z_GRASP
            dx, dy, dz = _drop_pos
            rospy.loginfo(
                f'[pick_arm] Pick 시작  '
                f'arm=({px:.3f}, {py:.3f}, {pz:.3f})')
            pick_and_place(px, py, pz, dx, dy, dz)   # 블로킹 직접 호출
        creep_start_t = None
        state         = IDLE
        rospy.loginfo('[pick_arm] Pick 완료 → IDLE')


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

    rospy.sleep(0.5)

    # ── 팔 초기화 + HOME 이동 ────────────────────────────────────────────
    init_arm(servo_pub)
    rospy.loginfo('[pick_arm] HOME 위치로 이동 중…')
    move_to(*HOME, 'home')
    rospy.sleep(1.5)

    # ── Subscribers ───────────────────────────────────────────────────────
    rospy.Subscriber('/yolo/arm_point', Point, arm_point_cb)
    rospy.Subscriber('/place_target',   Point, place_target_cb)

    # ── 10 Hz 제어 루프 ───────────────────────────────────────────────────
    rospy.Timer(rospy.Duration(0.1), control_loop)

    rospy.loginfo('[pick_arm] 노드 준비 완료.')
    rospy.loginfo('  구독: /yolo/arm_point  /place_target')
    rospy.loginfo('  발행: /chassis_control/set_velocity')
    rospy.loginfo('       /servo_controllers/port_id_1/multi_id_pos_dur')
    rospy.loginfo(
        f'  목표: x={TARGET_X:.3f} ± {TOL_X:.3f}  '
        f'y={TARGET_Y:.3f} ± {TOL_Y:.3f}  (m)')
    rospy.spin()