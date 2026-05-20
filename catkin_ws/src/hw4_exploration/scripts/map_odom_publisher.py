#!/usr/bin/env python3
import rospy
import tf2_ros
from nav_msgs.msg import Odometry

def run():
    rospy.init_node("map_odom_publisher")
    pub = rospy.Publisher("/map_odom", Odometry, queue_size=10)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        try:
            trans = tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0))
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "map"
            odom.child_frame_id = "base_footprint"
            odom.pose.pose.position.x = trans.transform.translation.x
            odom.pose.pose.position.y = trans.transform.translation.y
            odom.pose.pose.position.z = trans.transform.translation.z
            odom.pose.pose.orientation = trans.transform.rotation
            pub.publish(odom)
        except Exception as e:
            pass
        rate.sleep()

if __name__ == "__main__":
    run()
