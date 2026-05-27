#!/usr/bin/env python3
import rospy
import subprocess
import time
import sys

def check_and_start():
    rospy.init_node('camera_checker', anonymous=True)
    
    # Force restart camera to apply new resolution settings
    rospy.loginfo("[Camera Checker] Terminating existing realsense nodes to apply high resolution parameters...")
    subprocess.call(["rosnode", "kill", "/camera/realsense2_camera"])
    subprocess.call(["rosnode", "kill", "/camera/realsense2_camera_manager"])
    time.sleep(2.5)
    
    rospy.loginfo("[Camera Checker] Launching realsense2_camera rs_aligned_depth.launch...")
    # Start realsense camera with enable_pointcloud:=true to get the point cloud for costmap generator
    cmd = [
        "roslaunch", "realsense2_camera", "rs_aligned_depth.launch",
        "enable_pointcloud:=true"
    ]
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
