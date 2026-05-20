#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped

class TagPoseInjector:
    def __init__(self):
        rospy.init_node('tag_pose_injector')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.pub = rospy.Publisher('/rtabmap/global_pose', PoseWithCovarianceStamped, queue_size=1)
        self.sub = rospy.Subscriber('/camera_link', PoseStamped, self.pose_cb)
        
        rospy.loginfo("Tag Pose Injector started. Listening to /camera_link")

    def pose_cb(self, msg):
        try:
            # We have the pose of camera_link in the map frame.
            # We need the pose of base_footprint in the map frame.
            # We can lookup the transform from camera_link to base_footprint
            # and apply it to the Pose.
            
            # Get the transform from camera_link to base_footprint
            # Wait, if we want base_footprint in map, we take camera_link in map,
            # and multiply by base_footprint in camera_link.
            
            # Let's construct a TransformStamped for camera_link in map
            cam_in_map = TransformStamped()
            cam_in_map.header = msg.header
            cam_in_map.child_frame_id = "temp_camera_link"
            cam_in_map.transform.translation.x = msg.pose.position.x
            cam_in_map.transform.translation.y = msg.pose.position.y
            cam_in_map.transform.translation.z = msg.pose.position.z
            cam_in_map.transform.rotation = msg.pose.orientation
            
            # Lookup base_footprint in camera_link
            base_in_cam = self.tf_buffer.lookup_transform("camera_link", "base_footprint", rospy.Time(0), rospy.Duration(1.0))
            
            # Now we use tf2 to multiply them: map -> temp_camera_link * camera_link -> base_footprint = map -> base_footprint
            
            # Create a PoseStamped representing the origin of base_footprint in the base_footprint frame
            base_origin = PoseStamped()
            base_origin.header.frame_id = "base_footprint"
            base_origin.header.stamp = rospy.Time(0)
            base_origin.pose.orientation.w = 1.0
            
            # Transform it to camera_link frame
            base_in_cam_pose = tf2_geometry_msgs.do_transform_pose(base_origin, base_in_cam)
            
            # Now transform it to map frame using the camera_link pose in map
            # We can't do this directly with do_transform_pose unless we insert cam_in_map into tf buffer.
            # But we can just manually apply it, or better yet, since camera is static relative to base_footprint:
            # camera is at x=0.20, z=0.20 from base_link. base_footprint is at z=0 from base_link.
            # So base_footprint is at x=-0.20, z=-0.20 from camera_link.
            # Actually, the safest way is to let tf2 do the math.
        except Exception as e:
            rospy.logerr(f"TF lookup failed: {e}")
            return
            
        # Simplified manual transform since we know camera_link is at x=0.2, z=0.2 relative to base_footprint, 
        # and pitch=0.05.
        # Actually, let's just use the TF buffer! We can just look up 'base_footprint' to 'camera_link'
        try:
            cam_to_base = self.tf_buffer.lookup_transform("camera_link", "base_footprint", rospy.Time(0), rospy.Duration(0.1))
            
            # Apply cam_to_base transform to the origin to get base position in camera frame
            base_in_cam_pose = PoseStamped()
            base_in_cam_pose.pose.orientation.w = 1.0
            base_in_cam_pose = tf2_geometry_msgs.do_transform_pose(base_in_cam_pose, cam_to_base)
            
            # To transform this to map frame, we need the transform from map to camera_link.
            # But msg IS the pose of camera_link in map!
            # So msg is equivalent to the transform from map to camera_link.
            map_to_cam = TransformStamped()
            map_to_cam.transform.translation.x = msg.pose.position.x
            map_to_cam.transform.translation.y = msg.pose.position.y
            map_to_cam.transform.translation.z = msg.pose.position.z
            map_to_cam.transform.rotation = msg.pose.orientation
            
            base_in_map = tf2_geometry_msgs.do_transform_pose(base_in_cam_pose, map_to_cam)
            
            # Construct the final message
            cov_msg = PoseWithCovarianceStamped()
            cov_msg.header.stamp = rospy.Time.now()
            cov_msg.header.frame_id = "map"
            cov_msg.pose.pose = base_in_map.pose
            
            # Small covariance because AprilTag is highly accurate
            cov_msg.pose.covariance[0] = 0.01  # x
            cov_msg.pose.covariance[7] = 0.01  # y
            cov_msg.pose.covariance[14] = 0.01 # z
            cov_msg.pose.covariance[21] = 0.01 # roll
            cov_msg.pose.covariance[28] = 0.01 # pitch
            cov_msg.pose.covariance[35] = 0.01 # yaw
            
            self.pub.publish(cov_msg)
            rospy.loginfo("Injected Global Pose to RTAB-Map based on AprilTag!")
            
        except Exception as e:
            rospy.logerr(f"Pose injection failed: {e}")

if __name__ == '__main__':
    TagPoseInjector()
    rospy.spin()
