#!/usr/bin/env python3
"""
map_odom_publisher.py  –  TF heartbeat relay for the map → odom transform,
and dynamic alignment of RTAB-Map's origin to the AprilTag world frame.

ROOT CAUSE OF Costmap2DROS TRANSFORM TIMEOUT:
  RTAB-Map publishes rtabmap_map → odom only when it processes a new frame.
  When RTAB-Map stalls, the transform expires.
  
FIX:
  Cache the last good rtabmap_map → odom transform and re-broadcast it at 20 Hz.
  
ALIGNMENT FIX:
  Subscribe to /camera_link (published by apriltag_localization in the 'map' frame).
  Calculate the offset between the AprilTag 'map' and RTAB-Map's 'rtabmap_map'.
  Publish map → rtabmap_map at 20 Hz so the entire RTAB-Map is correctly shifted.
"""

import rospy
import tf2_ros
import tf.transformations as tf_trans
import numpy as np
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Odometry


class MapOdomPublisher:
    def __init__(self):
        rospy.init_node("map_odom_publisher")

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_br = tf2_ros.TransformBroadcaster()

        self.odom_pub = rospy.Publisher("/map_odom", Odometry, queue_size=10)
        
        # Subscribe to AprilTag localization
        self.pose_sub = rospy.Subscriber("/camera_link", PoseStamped, self.pose_cb)

        self.last_good_rtabmap_to_odom = None
        self.map_to_rtabmap_mat = np.eye(4)  # Default: no offset
        self.diag_divider = 0

    def pose_to_mat(self, pose):
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        mat = tf_trans.quaternion_matrix(q)
        mat[0, 3] = pose.position.x
        mat[1, 3] = pose.position.y
        mat[2, 3] = pose.position.z
        return mat

    def transform_to_mat(self, transform):
        q = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
        mat = tf_trans.quaternion_matrix(q)
        mat[0, 3] = transform.translation.x
        mat[1, 3] = transform.translation.y
        mat[2, 3] = transform.translation.z
        return mat

    def mat_to_transform(self, mat):
        t = TransformStamped().transform
        t.translation.x = mat[0, 3]
        t.translation.y = mat[1, 3]
        t.translation.z = mat[2, 3]
        q = tf_trans.quaternion_from_matrix(mat)
        t.rotation.x = q[0]
        t.rotation.y = q[1]
        t.rotation.z = q[2]
        t.rotation.w = q[3]
        return t

    def pose_cb(self, msg: PoseStamped):
        # msg is the pose of camera_link in the AprilTag 'map' frame
        try:
            # We need to find rtabmap_map -> camera_link
            t_rtabmap_to_cam = self.tf_buffer.lookup_transform("rtabmap_map", "camera_link", rospy.Time(0), rospy.Duration(0.1))
            
            map_T_cam = self.pose_to_mat(msg.pose)
            rtabmap_T_cam = self.transform_to_mat(t_rtabmap_to_cam.transform)
            
            # map -> rtabmap_map = (map -> cam) * (rtabmap_map -> cam)^-1
            self.map_to_rtabmap_mat = np.dot(map_T_cam, np.linalg.inv(rtabmap_T_cam))
            
            rospy.loginfo_once("[map_odom_publisher] Successfully aligned RTAB-Map origin to AprilTag frame!")
        except Exception as e:
            # Normal if rtabmap_map -> camera_link is not yet available
            pass

    def run(self):
        rate = rospy.Rate(20)
        rospy.loginfo("[map_odom_publisher] TF heartbeat relay and alignment started @ 20 Hz.")

        while not rospy.is_shutdown():
            # 1. Update rtabmap_map -> odom from RTAB-Map
            try:
                t = self.tf_buffer.lookup_transform("rtabmap_map", "odom", rospy.Time(0), rospy.Duration(0.0))
                if self.last_good_rtabmap_to_odom is None or t.header.stamp > self.last_good_rtabmap_to_odom.header.stamp:
                    self.last_good_rtabmap_to_odom = t
            except Exception:
                pass

            now = rospy.Time.now()

            # 2. Broadcast map -> rtabmap_map
            align_relay = TransformStamped()
            align_relay.header.stamp = now
            align_relay.header.frame_id = "map"
            align_relay.child_frame_id = "rtabmap_map"
            align_relay.transform = self.mat_to_transform(self.map_to_rtabmap_mat)
            self.tf_br.sendTransform(align_relay)

            # 3. Broadcast rtabmap_map -> odom
            if self.last_good_rtabmap_to_odom is not None:
                odom_relay = TransformStamped()
                odom_relay.header.stamp = now
                odom_relay.header.frame_id = "rtabmap_map"
                odom_relay.child_frame_id = "odom"
                odom_relay.transform = self.last_good_rtabmap_to_odom.transform
                self.tf_br.sendTransform(odom_relay)

                # 4. Diagnostic /map_odom
                self.diag_divider += 1
                if self.diag_divider >= 2:
                    self.diag_divider = 0
                    odom = Odometry()
                    odom.header.stamp = now
                    odom.header.frame_id = "map"
                    odom.child_frame_id = "base_footprint"
                    try:
                        full = self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0), rospy.Duration(0.0))
                        odom.pose.pose.position.x = full.transform.translation.x
                        odom.pose.pose.position.y = full.transform.translation.y
                        odom.pose.pose.position.z = full.transform.translation.z
                        odom.pose.pose.orientation = full.transform.rotation
                        self.odom_pub.publish(odom)
                    except Exception:
                        pass

            rate.sleep()

if __name__ == "__main__":
    node = MapOdomPublisher()
    node.run()
