#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import base64
import json
import os
import threading
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from gpt_llm_client.srv import LLMVisionQuery, LLMVisionQueryRequest
from hw4_exploration.srv import AnalyzeSign, AnalyzeSignResponse, DetectShopfront, DetectShopfrontResponse

class VisionPerceptionNode:
    def __init__(self):
        rospy.init_node("vision_perception_node")
        
        self.bridge = CvBridge()
        self.image_lock = threading.Lock()
        self.latest_cv_image = None
        
        # Load storefront templates for local OpenCV classification fallback
        self.templates = {}
        self.template_dir = "/home/linusv/project_5/HW4/Stores"
        self.load_templates()
        
        # Subscribe to RealSense RGB camera topic
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_cb)
        
        # Do not wait for LLM service at startup to prevent deadlock.
        # It will be connected on-demand when Remote AI mode is enabled.
        self.llm_vision_srv = None
        
        # Publisher for semantic map builder
        self.semantic_pub = rospy.Publisher("/semantic_observations", String, queue_size=1)
        
        # Advertise Services
        self.analyze_srv = rospy.Service("/analyze_sign", AnalyzeSign, self.handle_analyze_sign)
        self.detect_srv = rospy.Service("/detect_shopfront", DetectShopfront, self.handle_detect_shopfront)
        
        rospy.loginfo("Vision Perception Services advertised and ready.")

    def image_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.image_lock:
                self.latest_cv_image = cv_img
        except Exception as e:
            rospy.logwarn(f"CvBridge conversion error: {e}")

    def load_templates(self):
        rospy.loginfo("Local ORB feature matching is disabled.")

    def get_latest_frame(self):
        with self.image_lock:
            if self.latest_cv_image is None:
                return None
            return self.latest_cv_image.copy()

    def handle_analyze_sign(self, req):
        rospy.loginfo(f"[/analyze_sign] Requested for target: {req.target_shop}")
        res = AnalyzeSignResponse()
        res.direction = "UNKNOWN"
        
        frame = self.get_latest_frame()
        if frame is None:
            rospy.logwarn("No camera frame available to analyze signboard.")
            return res
            
        # Encode image to base64
        _, buffer = cv2.imencode(".jpg", frame)
        b64_img = base64.b64encode(buffer).decode("utf-8")
        
        prompt = (
            f"You are a robot vision system looking at a signboard at an intersection in a maze. "
            f"The robot is trying to find the target store: '{req.target_shop}'. "
            f"The signboard displays the names or categories of stores (Cafe, Fast-food restaurant, Pharmacy, Convenience Store) along with direction arrows. "
            f"Analyze the signboard image. Determine which arrow corresponds to the target shop '{req.target_shop}' or its category. "
            f"Return ONLY a raw JSON object with a single key 'direction' which must be one of: "
            f"'LEFT', 'RIGHT', 'STRAIGHT', or 'UNKNOWN'."
        )
        # Check if local AI mode is requested
        use_local = rospy.get_param("/use_local_ai", True)
        if use_local:
            # Local AI Mode Heuristic: systematically alternate turns at intersections
            if not hasattr(self, "last_turn"):
                self.last_turn = "LEFT"
            
            next_turn = "LEFT" if self.last_turn == "RIGHT" else ("RIGHT" if self.last_turn == "STRAIGHT" else "STRAIGHT")
            self.last_turn = next_turn
            res.direction = next_turn
            rospy.loginfo(f"[/analyze_sign] Local AI Mode: systematically choosing {next_turn}")
            
            # Publish to semantic map
            obs = {
                "has_signboard": True,
                "category": req.target_shop,
                "direction": next_turn.capitalize()
            }
            obs_msg = String()
            obs_msg.data = json.dumps(obs)
            self.semantic_pub.publish(obs_msg)
            return res
            
        try:
            if self.llm_vision_srv is None:
                rospy.loginfo("Connecting to /llm_vision_query service on demand...")
                try:
                    rospy.wait_for_service("/llm_vision_query", timeout=3.0)
                    self.llm_vision_srv = rospy.ServiceProxy("/llm_vision_query", LLMVisionQuery)
                except rospy.ROSException as e:
                    rospy.logerr(f"LLM Vision service not available: {e}")
                    # Fallback to systematic local mode
                    if not hasattr(self, "last_turn"):
                        self.last_turn = "LEFT"
                    next_turn = "LEFT" if self.last_turn == "RIGHT" else ("RIGHT" if self.last_turn == "STRAIGHT" else "STRAIGHT")
                    self.last_turn = next_turn
                    res.direction = next_turn
                    return res

            llm_req = LLMVisionQueryRequest()
            llm_req.prompt = prompt
            llm_req.base64_image = b64_img
            
            llm_res = self.llm_vision_srv(llm_req)
            rospy.loginfo(f"[/analyze_sign] LLM raw response: {llm_res.response}")
            
            # Parse JSON safely
            # Clean possible markdown formatting from response
            cleaned_resp = llm_res.response.strip()
            if cleaned_resp.startswith("```json"):
                cleaned_resp = cleaned_resp[7:]
            if cleaned_resp.endswith("```"):
                cleaned_resp = cleaned_resp[:-3]
            cleaned_resp = cleaned_resp.strip()
            
            data = json.loads(cleaned_resp)
            direction = data.get("direction", "UNKNOWN").upper().strip()
            if direction in ["LEFT", "RIGHT", "STRAIGHT", "UNKNOWN"]:
                res.direction = direction
                rospy.loginfo(f"[/analyze_sign] Parsed direction: {direction}")
                
                # Publish to semantic map
                obs = {
                    "has_signboard": True,
                    "category": req.target_shop,
                    "direction": direction.capitalize()
                }
                obs_msg = String()
                obs_msg.data = json.dumps(obs)
                self.semantic_pub.publish(obs_msg)
                
        except Exception as e:
            rospy.logerr(f"Error calling LLM Vision for signboard: {e}")
            
        return res

    def classify_by_color_histogram(self, frame):
        try:
            # Crop to center region to avoid wall and background noise
            h, w = frame.shape[:2]
            center = frame[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
            hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
            
            # Color range definitions (Hue range is 0-180 in OpenCV)
            ranges = {
                "green": cv2.inRange(hsv, (35, 40, 40), (85, 255, 255)),
                "yellow": cv2.inRange(hsv, (20, 50, 50), (32, 255, 255)),
                "orange": cv2.inRange(hsv, (10, 50, 50), (20, 255, 255)),
                "red1": cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)),
                "red2": cv2.inRange(hsv, (170, 50, 50), (180, 255, 255)),
                "blue": cv2.inRange(hsv, (95, 50, 50), (130, 255, 255)),
                "cyan": cv2.inRange(hsv, (80, 50, 50), (95, 255, 255)),
                "white": cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
            }
            
            counts = {k: cv2.countNonZero(v) for k, v in ranges.items()}
            counts["red"] = counts["red1"] + counts["red2"]
            
            total_pixels = center.shape[0] * center.shape[1]
            
            # Print debugging pixel percentages
            rospy.loginfo(f"[HSV Seg] px% -> green:{counts['green']/total_pixels*100:.1f}%, yellow:{counts['yellow']/total_pixels*100:.1f}%, orange:{counts['orange']/total_pixels*100:.1f}%, red:{counts['red']/total_pixels*100:.1f}%, blue:{counts['blue']/total_pixels*100:.1f}%, cyan:{counts['cyan']/total_pixels*100:.1f}%, white:{counts['white']/total_pixels*100:.1f}%")
            
            # Minimum area threshold (8% of center region occupied by dominant color)
            thresh = 0.08
            
            # 1. Green Store
            if counts["green"] / total_pixels > thresh:
                return "GREEN STORE"
            # 2. Yellow Burger
            if counts["yellow"] / total_pixels > thresh:
                return "YELLOW BURGER"
            # 3. Orange Cafe
            if counts["orange"] / total_pixels > thresh:
                return "ORANGE CAFE"
            # 4. Red options (Red Burger or Red Pharmacy)
            if counts["red"] / total_pixels > thresh:
                # Red Burger has a yellow sub-component in its graphic, Red Pharmacy is mostly solid red
                if counts["yellow"] > counts["red"] * 0.15:
                    return "RED BURGER"
                else:
                    return "RED PHARMACY"
            # 5. Blue/Cyan options (Blue Cafe or Blue Store)
            if counts["blue"] / total_pixels > thresh or counts["cyan"] / total_pixels > thresh:
                # Blue Cafe is cyan/light blue, Blue Store is dark blue
                if counts["cyan"] > counts["blue"] * 0.4:
                    return "BLUE CAFE"
                else:
                    return "BLUE STORE"
            # 6. White Cafe
            if counts["white"] / total_pixels > 0.20:
                return "WHITE CAFE"
                
        except Exception as e:
            rospy.logwarn(f"Color classification error: {e}")
        return None

    def handle_detect_shopfront(self, req):
        rospy.loginfo(f"[/detect_shopfront] Requested for target: {req.target_shop}")
        res = DetectShopfrontResponse()
        res.is_found = False
        
        frame = self.get_latest_frame()
        if frame is None:
            rospy.logwarn("No camera frame available to detect storefront.")
            return res
            
        # Try local OpenCV color classification first
        detected_store = self.classify_by_color_histogram(frame)
        
        # If local classification is uncertain, call the VLM API
        if detected_store is None:
            use_local = rospy.get_param("/use_local_ai", True)
            if use_local:
                rospy.loginfo("Local AI Mode active. No VLM call. Defaulting to UNKNOWN")
                detected_store = "UNKNOWN"
            else:
                rospy.loginfo("Calling LLM Vision service...")
                if self.llm_vision_srv is None:
                    rospy.loginfo("Connecting to /llm_vision_query service on demand...")
                    try:
                        rospy.wait_for_service("/llm_vision_query", timeout=3.0)
                        self.llm_vision_srv = rospy.ServiceProxy("/llm_vision_query", LLMVisionQuery)
                    except rospy.ROSException as e:
                        rospy.logerr(f"LLM Vision service not available: {e}")
                        detected_store = "UNKNOWN"
                
                if self.llm_vision_srv is not None:
                    # Encode image to base64
                    _, buffer = cv2.imencode(".jpg", frame)
                    b64_img = base64.b64encode(buffer).decode("utf-8")
                    
                    prompt = (
                        f"You are a robot vision system looking at a storefront in a maze. "
                        f"We are trying to find the target store: '{req.target_shop}'. "
                        f"The available storefront options in this maze are exactly:\n"
                        f"- BLUE CAFE\n- BLUE STORE\n- GREEN STORE\n- ORANGE CAFE\n"
                        f"- RED BURGER\n- RED PHARMACY\n- WHITE CAFE\n- YELLOW BURGER\n\n"
                        f"Determine which of the 8 storefronts is in the camera view. "
                        f"Return ONLY a raw JSON object with two keys:\n"
                        f"1. 'storefront' (string, must be the exact matched storefront name, or 'UNKNOWN')\n"
                        f"2. 'is_found' (boolean, true if it matches '{req.target_shop}' or its category, false otherwise)."
                    )
                    
                    try:
                        llm_req = LLMVisionQueryRequest()
                        llm_req.prompt = prompt
                        llm_req.base64_image = b64_img
                        
                        llm_res = self.llm_vision_srv(llm_req)
                        rospy.loginfo(f"[/detect_shopfront] LLM raw response: {llm_res.response}")
                        
                        # Parse JSON safely
                        cleaned_resp = llm_res.response.strip()
                        if cleaned_resp.startswith("```json"):
                            cleaned_resp = cleaned_resp[7:]
                        if cleaned_resp.endswith("```"):
                            cleaned_resp = cleaned_resp[:-3]
                        cleaned_resp = cleaned_resp.strip()
                        
                        data = json.loads(cleaned_resp)
                        detected_store = data.get("storefront", "UNKNOWN").upper().strip()
                        
                    except Exception as e:
                        rospy.logerr(f"Error calling LLM Vision for shopfront: {e}")
                        detected_store = "UNKNOWN"
                else:
                    detected_store = "UNKNOWN"
        
        # Map storefront names to correct Categories
        category_map = {
            "BLUE CAFE": "Café",
            "ORANGE CAFE": "Café",
            "WHITE CAFE": "Café",
            "RED BURGER": "Fast-food restaurant",
            "YELLOW BURGER": "Fast-food restaurant",
            "RED PHARMACY": "Pharmacy",
            "BLUE STORE": "Convenience store",
            "GREEN STORE": "Convenience store"
        }
        
        category = category_map.get(detected_store, "Unknown")
        rospy.loginfo(f"[/detect_shopfront] Identified storefront: {detected_store} (Category: {category})")
        
        # Check if the detected storefront matches the requested target shop
        # Support matching either exact storefront name (e.g. BLUE CAFE) or its Category (e.g. Café)
        if detected_store != "UNKNOWN":
            res.is_found = (
                req.target_shop.upper() in [detected_store, category.upper()] or
                detected_store in req.target_shop.upper() or
                category.upper() in req.target_shop.upper()
            )
            
            # Publish observation to the semantic map builder node
            obs = {
                "has_signboard": False,
                "storefront": detected_store,
                "category": category,
                "direction": ""
            }
            obs_msg = String()
            obs_msg.data = json.dumps(obs)
            self.semantic_pub.publish(obs_msg)
            rospy.loginfo(f"[/detect_shopfront] Published semantic observation: {obs}")
            
        return res

if __name__ == "__main__":
    try:
        node = VisionPerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
