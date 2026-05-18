#!/usr/bin/env python3
import rospy
import math
from apriltag_ros.msg import AprilTagDetectionArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf.transformations

class TagVisualizer:
    def __init__(self):
        rospy.init_node("tag_visualizer")
        self.sub = rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.callback)
        self.pub = rospy.Publisher("/tag_markers", MarkerArray, queue_size=1)
        rospy.loginfo("Tag Visualizer Node Started. Publishing to /tag_markers.")

    def callback(self, msg):
        marker_array = MarkerArray()
        
        for i, detection in enumerate(msg.detections):
            tag_id = detection.id[0]
            size = detection.size[0] if len(detection.size) > 0 else 0.15 # Default size 0.15m if not provided
            
            pose = detection.pose.pose.pose
            x = pose.position.x
            y = pose.position.y
            z = pose.position.z
            
            distance = math.sqrt(x**2 + y**2 + z**2)
            
            # 1. Boundary (Line Strip)
            bound_marker = Marker()
            bound_marker.header = detection.pose.header
            bound_marker.ns = "tag_boundary"
            bound_marker.id = tag_id
            bound_marker.type = Marker.LINE_STRIP
            bound_marker.action = Marker.ADD
            bound_marker.scale.x = 0.01 # Line width
            bound_marker.color.r = 1.0
            bound_marker.color.g = 1.0
            bound_marker.color.b = 0.0
            bound_marker.color.a = 1.0
            
            # The tag frame is typically Z pointing OUT (normal), X right, Y down.
            # So the corners in the tag frame are (+-size/2, +-size/2, 0).
            # To transform these corners to the camera frame, we apply the pose transformation.
            q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            mat = tf.transformations.quaternion_matrix(q)
            mat[0][3] = x
            mat[1][3] = y
            mat[2][3] = z
            
            corners = [
                [size/2, size/2, 0, 1],
                [-size/2, size/2, 0, 1],
                [-size/2, -size/2, 0, 1],
                [size/2, -size/2, 0, 1],
                [size/2, size/2, 0, 1] # close the loop
            ]
            
            for c in corners:
                pt_cam = tf.transformations.unit_vector([
                    mat[0][0]*c[0] + mat[0][1]*c[1] + mat[0][2]*c[2] + mat[0][3],
                    mat[1][0]*c[0] + mat[1][1]*c[1] + mat[1][2]*c[2] + mat[1][3],
                    mat[2][0]*c[0] + mat[2][1]*c[1] + mat[2][2]*c[2] + mat[2][3]
                ])
                # Note: tf.transformations.unit_vector normalizes it. We DO NOT want to normalize.
                
                pt_cam_real = [
                    mat[0][0]*c[0] + mat[0][1]*c[1] + mat[0][2]*c[2] + mat[0][3],
                    mat[1][0]*c[0] + mat[1][1]*c[1] + mat[1][2]*c[2] + mat[1][3],
                    mat[2][0]*c[0] + mat[2][1]*c[1] + mat[2][2]*c[2] + mat[2][3]
                ]
                
                p = Point()
                p.x = pt_cam_real[0]
                p.y = pt_cam_real[1]
                p.z = pt_cam_real[2]
                bound_marker.points.append(p)
                
            marker_array.markers.append(bound_marker)
            
            # 2. Normal Vector (Arrow)
            arrow_marker = Marker()
            arrow_marker.header = detection.pose.header
            arrow_marker.ns = "tag_normal"
            arrow_marker.id = tag_id
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            
            # Start of arrow is tag center
            start_pt = Point(x, y, z)
            
            # End of arrow is 0.3m along Z axis (normal)
            normal_len = 0.3
            n_pt = [0, 0, normal_len, 1]
            end_x = mat[0][0]*n_pt[0] + mat[0][1]*n_pt[1] + mat[0][2]*n_pt[2] + mat[0][3]
            end_y = mat[1][0]*n_pt[0] + mat[1][1]*n_pt[1] + mat[1][2]*n_pt[2] + mat[1][3]
            end_z = mat[2][0]*n_pt[0] + mat[2][1]*n_pt[1] + mat[2][2]*n_pt[2] + mat[2][3]
            end_pt = Point(end_x, end_y, end_z)
            
            arrow_marker.points.append(start_pt)
            arrow_marker.points.append(end_pt)
            arrow_marker.scale.x = 0.02 # shaft diameter
            arrow_marker.scale.y = 0.04 # head diameter
            arrow_marker.scale.z = 0.0  # head length (0=default)
            arrow_marker.color.r = 0.0
            arrow_marker.color.g = 1.0
            arrow_marker.color.b = 0.0
            arrow_marker.color.a = 1.0
            marker_array.markers.append(arrow_marker)
            
            # 3. Distance Text (TEXT_VIEW_FACING)
            text_marker = Marker()
            text_marker.header = detection.pose.header
            text_marker.ns = "tag_distance"
            text_marker.id = tag_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y - (size/2 + 0.1) # Display slightly above the tag (y is down in camera frame)
            text_marker.pose.position.z = z
            text_marker.scale.z = 0.1 # Text height
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f"Dist: {distance:.2f}m"
            marker_array.markers.append(text_marker)
            
        # Delete markers if no tags are seen? 
        # For simplicity, we just publish the array. The markers will persist until updated.
        if len(msg.detections) > 0:
            self.pub.publish(marker_array)

if __name__ == "__main__":
    try:
        node = TagVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
