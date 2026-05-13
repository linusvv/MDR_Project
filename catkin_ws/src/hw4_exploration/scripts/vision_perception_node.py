#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from gpt_llm_client.srv import LLMVisionQuery, LLMVisionQueryRequest
import base64
import json

class VisionPerceptionNode:
    def __init__(self):
        rospy.init_node("vision_perception_node")
        
        self.bridge = CvBridge()
        
        # Wait for the Vision LLM service to be available
        rospy.loginfo("Waiting for llm_vision_query service...")
        rospy.wait_for_service("llm_vision_query")
        self.vision_client = rospy.ServiceProxy("llm_vision_query", LLMVisionQuery)
        rospy.loginfo("Connected to llm_vision_query service.")
        
        # Publishers and Subscribers
        self.observation_pub = rospy.Publisher("/semantic_observations", String, queue_size=10)
        
        # Throttle parameters to avoid spamming the API
        self.last_process_time = rospy.Time.now()
        self.process_interval = rospy.Duration(3.0) # Process an image every 3 seconds
        
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)
        
        # The prompt engineered to extract specific assignment info
        self.prompt = """
        You are looking through the front camera of a robot exploring a maze.
        Identify if there is a signboard in this image.
        If there is a signboard, identify:
        1. The store category (Convenience store, Cafe, Fast-food restaurant, Pharmacy).
        2. The direction the arrow on the signboard is pointing (Left, Right, Forward, Unknown).
        
        Respond ONLY with a valid JSON object in the following format. Do not add markdown blocks or any other text.
        {"has_signboard": true/false, "category": "Store Type", "direction": "Arrow Direction"}
        """
        
    def image_callback(self, msg):
        current_time = rospy.Time.now()
        if (current_time - self.last_process_time) < self.process_interval:
            return
            
        self.last_process_time = current_time
        
        try:
            # Convert ROS Image to OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Resize image to save tokens and latency (OpenAI recommends max 2048x2048, but 512x512 is plenty for signboards)
            cv_image_resized = cv2.resize(cv_image, (512, 512))
            
            # Encode image to JPEG then to Base64
            _, buffer = cv2.imencode('.jpg', cv_image_resized)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # Create the request
            req = LLMVisionQueryRequest()
            req.prompt = self.prompt
            req.base64_image = base64_image
            
            # Call the service
            rospy.loginfo("Sending image to Vision API...")
            res = self.vision_client(req)
            
            # Parse and publish the response
            try:
                parsed_json = json.loads(res.response)
                if parsed_json.get("has_signboard"):
                    rospy.loginfo(f"Detected Signboard: {parsed_json}")
                    
                    # Publish the valid observation
                    obs_msg = String()
                    obs_msg.data = json.dumps(parsed_json)
                    self.observation_pub.publish(obs_msg)
                else:
                    rospy.logdebug("No signboard detected.")
            except json.JSONDecodeError:
                rospy.logerr(f"Failed to parse LLM response as JSON. Raw response: {res.response}")
                
        except Exception as e:
            rospy.logerr(f"Error in vision processing: {e}")

if __name__ == "__main__":
    try:
        node = VisionPerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
