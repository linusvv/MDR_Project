#!/usr/bin/env python3
import rospy
import os
import re
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class StoreCoordinatorNode:
    def __init__(self):
        rospy.init_node("store_coordinator_node")
        self.pub = rospy.Publisher("/store_markers", MarkerArray, queue_size=1, latch=True)
        self.stores = []
        self.load_stores()
        self.publish_markers()

    def load_stores(self):
        import rospkg
        try:
            pkg_path = rospkg.RosPack().get_path('hw4_exploration')
            mdr_path = os.path.dirname(os.path.dirname(os.path.dirname(pkg_path)))
            file_path = os.path.join(mdr_path, 'HW4', 'Store coordinates.txt')
        except Exception:
            file_path = "/home/ee478_team1/catkin_ws/src/MDR_Project/HW4/Store coordinates.txt"

        if not os.path.exists(file_path):
            rospy.logerr(f"Store coordinates file not found at {file_path}")
            return
            
        with open(file_path, "r") as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("#") or not line:
                continue
                
            # Parse (X, Y)
            match = re.search(r'\(([^,]+),\s*([^)]+)\)', line)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                self.stores.append((x, y))
                
        rospy.loginfo(f"Loaded {len(self.stores)} stores from configuration.")

    def publish_markers(self):
        marker_array = MarkerArray()
        
        for i, store in enumerate(self.stores):
            # Sphere marker for the store
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "stores"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = store[0]
            marker.pose.position.y = store[1]
            marker.pose.position.z = 0.5 # half meter up
            
            # Identity orientation
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0 # Cyan
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
            
            # Text marker
            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = rospy.Time.now()
            text.ns = "store_labels"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            
            text.pose.position.x = store[0]
            text.pose.position.y = store[1]
            text.pose.position.z = 0.8
            
            text.scale.z = 0.2
            
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            
            text.text = f"Store {i+1}"
            
            marker_array.markers.append(text)
            
        self.pub.publish(marker_array)

if __name__ == "__main__":
    try:
        node = StoreCoordinatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
