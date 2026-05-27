import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

model = YOLO("/home/ee478_team1/catkin_ws/src/controller/best.engine", task="detect")
bridge = CvBridge()

def callback(msg):
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    results = model(frame, conf=0.8, device=0, verbose=False)
    annotated = results[0].plot()
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls)]
        conf = float(box.conf)
        print(f"Erkannt: {cls_name} ({conf:.2f})")
    cv2.imshow("YOLOv11 Detection", annotated)
    cv2.waitKey(1)

rospy.init_node("yolo_detector")
rospy.Subscriber("/camera/color/image_raw", Image, callback)
print("Warte auf Kamerabilder...")
rospy.spin()
