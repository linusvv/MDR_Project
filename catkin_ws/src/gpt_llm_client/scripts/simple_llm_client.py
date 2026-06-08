#!/usr/bin/env python3
import rospy
import os
from openai import OpenAI
from gpt_llm_client.srv import LLMQuery, LLMQueryResponse, LLMVisionQuery, LLMVisionQueryResponse


class StatelessLLMClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not self.api_key:
            import rospkg
            try:
                pkg_path = rospkg.RosPack().get_path('gpt_llm_client')
                mdr_path = os.path.dirname(os.path.dirname(os.path.dirname(pkg_path)))
                key_file = os.path.join(mdr_path, 'HW4', 'ChatGPT_API_KEY.txt')
            except Exception:
                key_file = "/home/ee478_team1/catkin_ws/src/MDR_Project/HW4/ChatGPT_API_KEY.txt"

            if os.path.exists(key_file):
                try:
                    with open(key_file, "r") as f:
                        content = f.read().strip()
                    import re
                    match = re.search(r'\b(sk-[a-zA-Z0-9_-]+)\b', content)
                    if match:
                        self.api_key = match.group(1)
                        rospy.loginfo(f"Successfully extracted and loaded OPENAI_API_KEY from {key_file}")
                    else:
                        rospy.logwarn(f"{key_file} found but could not extract a valid sk-... key.")
                except Exception as e:
                    rospy.logerr(f"Error reading key file: {e}")
            else:
                rospy.logwarn(f"OPENAI_API_KEY not set and key file not found at {key_file}. Waiting for user setup via HRI.")

        # Build initial client (may be recreated when a dynamic key is provided via ROS param)
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

        ################################### DO NOT USE BIGGER MODEL ###################################
        self.model = rospy.get_param("~model", "gpt-4.1-nano") ### Please use "gpt-4o-mini" or "gpt-4.1-nano" only. DO NOT USE BIGGER MODEL
        ################################### DO NOT USE BIGGER MODEL ###################################

        self.service = rospy.Service("llm_query", LLMQuery, self.handle)
        self.vision_service = rospy.Service("llm_vision_query", LLMVisionQuery, self.handle_vision)
        rospy.loginfo("Stateless LLM client ready")

    def load_dynamic_key(self):
        """Reload API key from ROS param (set by web UI) and recreate client if needed."""
        dynamic_key = rospy.get_param("/openai_api_key", "").strip()
        key = dynamic_key if dynamic_key else self.api_key
        if key:
            # Recreate client if the key changed or client was never created
            if self.client is None or dynamic_key:
                self.client = OpenAI(api_key=key)
            return True
        return False

    def handle(self, req):
        if not self.load_dynamic_key():
            rospy.logwarn("LLM query failed: No OpenAI API key loaded.")
            return LLMQueryResponse("ERROR: OpenAI API Key is missing. Please paste and set your API key in the Delivery panel of the Web Control Dashboard.")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a robot assistant."},
                    {"role": "user", "content": req.prompt}
                ]
            )
            answer = resp.choices[0].message.content
            return LLMQueryResponse(answer)

        except Exception as e:
            rospy.logerr(f"OpenAI error: {e}")
            return LLMQueryResponse(f"ERROR: {e}")

    def handle_vision(self, req):
        if not self.load_dynamic_key():
            rospy.logwarn("LLM Vision query failed: No OpenAI API key loaded.")
            return LLMVisionQueryResponse("ERROR: OpenAI API Key is missing. Please paste and set your API key in the Delivery panel of the Web Control Dashboard.")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a robot assistant capable of analyzing images."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": req.prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{req.base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            answer = resp.choices[0].message.content
            return LLMVisionQueryResponse(answer)

        except Exception as e:
            rospy.logerr(f"OpenAI Vision error: {e}")
            return LLMVisionQueryResponse(f"ERROR: {e}")


if __name__ == "__main__":
    rospy.init_node("stateless_llm_client")
    node = StatelessLLMClient()
    rospy.spin()
