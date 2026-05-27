#!/usr/bin/env python3
import rospy
from hiwonder_servo_msgs.msg import ServoStateList

GRIPPER_OPEN = 200
GRIPPER_CLOSE = 500

def gripper_callback(msg):
    # Find servo id 1 (gripper) in the feedback list
    servo = next((s for s in msg.servo_states if s.id == 1), None)
    if servo is None:
        return

    actual_pulse = servo.position

    if actual_pulse <= (GRIPPER_OPEN + 20):
        rospy.loginfo(f"Pulse: {actual_pulse} -> [ OPEN ]")
    elif actual_pulse >= (GRIPPER_CLOSE - 20):
        rospy.logwarn(f"Pulse: {actual_pulse} -> [ EMPTY ] (Nothing grasped!)")
    else:
        rospy.loginfo(f"Pulse: {actual_pulse} -> [ CLOSED WITH OBJECT ] (Success)")

if __name__ == '__main__':
    rospy.init_node('gripper_test_node', anonymous=True)
    rospy.loginfo("observing gripper status...")
    
    # Abonniere das Topic mit den Servo-Feedbackdaten
    rospy.Subscriber('/servo_controllers/port_id_1/servo_states', ServoStateList, gripper_callback)
    
    rospy.spin() 