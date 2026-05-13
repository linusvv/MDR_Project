#!/usr/bin/env python3
import rospy
import time
import json
import cv2
import numpy as np
import base64
from std_msgs.msg import String
from hw4_exploration.srv import AdjustBase, AdjustBaseRequest
from gpt_llm_client.srv import LLMVisionQuery, LLMVisionQueryRequest

def run_tests():
    rospy.init_node('verification_test_node')
    
    rospy.loginfo("=== Starting Phase 2b Verification ===")
    
    # 1. Test Semantic Map Publisher
    rospy.loginfo("--- Test 1: Semantic Observation Publisher ---")
    obs_pub = rospy.Publisher("/semantic_observations", String, queue_size=1)
    time.sleep(1) # wait for connection
    test_obs = {"has_signboard": True, "category": "TestStore", "direction": "Left"}
    obs_msg = String()
    obs_msg.data = json.dumps(test_obs)
    obs_pub.publish(obs_msg)
    rospy.loginfo("Published dummy observation to /semantic_observations. Check if semantic_map_node logged it.")
    time.sleep(1)
    
    # 2. Test Adjust Base Service
    rospy.loginfo("--- Test 2: Adjust Base Service ---")
    rospy.loginfo("Waiting for /adjust_base_for_grasping service...")
    try:
        rospy.wait_for_service("/adjust_base_for_grasping", timeout=5.0)
        adjust_srv = rospy.ServiceProxy("/adjust_base_for_grasping", AdjustBase)
        req = AdjustBaseRequest()
        req.delta_x = 0.05
        req.delta_y = 0.0
        req.delta_theta = 0.0
        rospy.loginfo("Calling adjust_base_for_grasping (Move forward 5cm)...")
        res = adjust_srv(req)
        rospy.loginfo(f"Service returned: success={res.success}, message={res.message}")
    except rospy.ROSException:
        rospy.logerr("Service /adjust_base_for_grasping is NOT available. Is adjust_base_node running?")
    except Exception as e:
        rospy.logerr(f"Error calling adjust_base service: {e}")

    # 3. Test LLM Vision Client
    rospy.loginfo("--- Test 3: LLM Vision Query Service ---")
    rospy.loginfo("Waiting for /llm_vision_query service...")
    try:
        rospy.wait_for_service("/llm_vision_query", timeout=5.0)
        vision_srv = rospy.ServiceProxy("/llm_vision_query", LLMVisionQuery)
        
        # Create a tiny dummy image (white square)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode('.jpg', img)
        b64_img = base64.b64encode(buffer).decode('utf-8')
        
        req = LLMVisionQueryRequest()
        req.prompt = "This is a blank test image. Just reply with: {\"has_signboard\": false}"
        req.base64_image = b64_img
        
        rospy.loginfo("Calling Vision API with dummy image...")
        res = vision_srv(req)
        rospy.loginfo(f"Vision API responded: {res.response}")
    except rospy.ROSException:
        rospy.logerr("Service /llm_vision_query is NOT available. Is simple_llm_client running?")
    except Exception as e:
        rospy.logerr(f"Error calling vision service: {e}")

    rospy.loginfo("=== Verification Tests Completed ===")

if __name__ == '__main__':
    run_tests()
