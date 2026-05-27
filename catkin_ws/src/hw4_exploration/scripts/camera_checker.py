#!/usr/bin/env python3
import rospy
import subprocess
import time
import sys

def check_and_start():
    rospy.init_node('camera_checker', anonymous=True)
    topic_name = '/camera/color/image_raw'
    rospy.loginfo(f"[Camera Checker] Checking if 3D camera is publishing on {topic_name}...")
    
    camera_running = False
    start_time = time.time()
    
    # Wait up to 5 seconds to see if the camera topic exists in the active topic list
    while time.time() - start_time < 5.0 and not rospy.is_shutdown():
        topics = rospy.get_published_topics()
        if any(t[0] == topic_name for t in topics):
            camera_running = True
            break
        time.sleep(0.5)
        
    if camera_running:
        rospy.loginfo("[Camera Checker] 3D camera is already running. No action needed.")
        # Keep node alive to prevent launch file from showing it finished/exited
        rospy.spin()
    else:
        rospy.loginfo("[Camera Checker] 3D camera not detected. Launching realsense2_camera rs_aligned_depth.launch...")
        # Start realsense camera with enable_pointcloud:=true to get the point cloud for costmap generator
        cmd = ["roslaunch", "realsense2_camera", "rs_aligned_depth.launch", "enable_pointcloud:=true"]
        proc = subprocess.Popen(cmd)
        
        # Shutdown cleanly by killing the subprocess
        def shutdown_hook():
            rospy.loginfo("[Camera Checker] Shutting down, terminating realsense camera process...")
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        
        rospy.on_shutdown(shutdown_hook)
        
        while not rospy.is_shutdown():
            if proc.poll() is not None:
                rospy.logwarn("[Camera Checker] Realsense process terminated unexpectedly.")
                break
            time.sleep(1.0)

if __name__ == '__main__':
    try:
        check_and_start()
    except rospy.ROSInterruptException:
        pass
