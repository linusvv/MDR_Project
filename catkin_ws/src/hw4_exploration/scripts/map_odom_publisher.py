#!/usr/bin/env python3
"""
map_odom_publisher.py  –  TF heartbeat relay for the map → odom transform.

ROOT CAUSE OF Costmap2DROS TRANSFORM TIMEOUT:
  RTAB-Map publishes map → odom only when it processes a new frame (can be
  as slow as 1-2 Hz under load, or stops entirely if RTAB-Map freezes/crashes).
  The TF2 buffer keeps transforms for ~10 seconds by default.  When RTAB-Map
  stalls for > 10 s the transform expires → Costmap2DROS can no longer resolve
  the robot pose in the 'map' frame → constant timeout warnings.

FIX:
  Cache the last good map → odom transform and re-broadcast it at 20 Hz with
  rospy.Time.now() as the stamp.  This keeps the transform permanently fresh
  in every subscriber's TF buffer — even during a 30+ second RTAB-Map stall.
  Once RTAB-Map recovers it publishes a newer transform that naturally
  supersedes the relay's cached copy.

Also publishes /map_odom (nav_msgs/Odometry) at 10 Hz for diagnostic display.
"""

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry


def run():
    rospy.init_node("map_odom_publisher")

    # ------------------------------------------------------------------ #
    # TF infrastructure
    # ------------------------------------------------------------------ #
    # Large buffer so we keep a long history of RTAB-Map transforms
    tf_buffer   = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    tf_br       = tf2_ros.TransformBroadcaster()

    # Diagnostic odometry publisher
    odom_pub = rospy.Publisher("/map_odom", Odometry, queue_size=10)

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    last_good: TransformStamped = None   # last successfully looked-up transform
    diag_divider = 0                     # throttle /map_odom to 10 Hz inside 20 Hz loop

    rate = rospy.Rate(20)   # 20 Hz heartbeat is enough to keep TF buffer fresh
    rospy.loginfo("[map_odom_publisher] TF heartbeat relay started (map → odom @ 20 Hz).")

    while not rospy.is_shutdown():
        # ---- 1. Try to get a fresher transform from RTAB-Map ---------- #
        try:
            t = tf_buffer.lookup_transform("map", "odom", rospy.Time(0),
                                           rospy.Duration(0.0))  # non-blocking
            # Accept if newer than what we have cached
            if last_good is None or t.header.stamp > last_good.header.stamp:
                last_good = t
        except Exception:
            pass   # RTAB-Map not ready yet or temporarily frozen – use cache

        # ---- 2. Re-broadcast with current timestamp ------------------- #
        if last_good is not None:
            relay = TransformStamped()
            relay.header.stamp     = rospy.Time.now()   # ← key: always fresh
            relay.header.frame_id  = "map"
            relay.child_frame_id   = "odom"
            relay.transform        = last_good.transform
            tf_br.sendTransform(relay)

            # ---- 3. Publish /map_odom diagnostic topic at 10 Hz ------- #
            diag_divider += 1
            if diag_divider >= 2:
                diag_divider = 0
                odom = Odometry()
                odom.header.stamp     = relay.header.stamp
                odom.header.frame_id  = "map"
                odom.child_frame_id   = "base_footprint"
                try:
                    # Full chain: map → base_footprint (for position display)
                    full = tf_buffer.lookup_transform("map", "base_footprint",
                                                      rospy.Time(0),
                                                      rospy.Duration(0.0))
                    odom.pose.pose.position.x    = full.transform.translation.x
                    odom.pose.pose.position.y    = full.transform.translation.y
                    odom.pose.pose.position.z    = full.transform.translation.z
                    odom.pose.pose.orientation   = full.transform.rotation
                except Exception:
                    pass
                odom_pub.publish(odom)

        rate.sleep()


if __name__ == "__main__":
    run()
