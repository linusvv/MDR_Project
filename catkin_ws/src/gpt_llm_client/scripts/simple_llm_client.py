#!/usr/bin/env python3
import rospy
import os
import openai
from gpt_llm_client.srv import LLMQuery, LLMQueryResponse, LLMVisionQuery, LLMVisionQueryResponse


class StatelessLLMClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            key_file = "/home/linusv/project_5/HW4/ChatGPT_API_KEY.txt"
            if os.path.exists(key_file):
                try:
                    with open(key_file, "r") as f:
                        content = f.read().strip()
                    import re
                    match = re.search(r'\b(sk-[a-zA-Z0-9_-]+)\b', content)
                    if match:
                        api_key = match.group(1)
                        rospy.loginfo("Successfully extracted and loaded OPENAI_API_KEY from HW4/ChatGPT_API_KEY.txt")
                    else:
                        api_key = ""
                        rospy.logwarn("ChatGPT_API_KEY.txt found but could not extract a valid sk-... key.")
                except Exception as e:
                    api_key = ""
                    rospy.logerr(f"Error reading key file: {e}")
            else:
                rospy.logwarn("OPENAI_API_KEY not set and ChatGPT_API_KEY.txt not found. Waiting for user setup via HRI.")
                api_key = ""

        openai.api_key = api_key

        ################################### DO NOT USE BIGGER MODEL ###################################
        self.model = rospy.get_param("~model", "gpt-4.1-nano") ### Please use "gpt-4o-mini" or "gpt-4.1-nano" only. DO NOT USE BIGGER MODEL
        ################################### DO NOT USE BIGGER MODEL ###################################

        self.service = rospy.Service("llm_query", LLMQuery, self.handle)
        self.vision_service = rospy.Service("llm_vision_query", LLMVisionQuery, self.handle_vision)
        rospy.loginfo("Stateless LLM client ready")

    def load_dynamic_key(self):
        # Dynamically load session key from ROS parameters set by HRI Web UI
        dynamic_key = rospy.get_param("/openai_api_key", "").strip()
        if dynamic_key:
            openai.api_key = dynamic_key
            return True
        return bool(openai.api_key)

    def handle(self, req):
        if not self.load_dynamic_key():
            rospy.logwarn("LLM query failed: No OpenAI API key loaded.")
            return LLMQueryResponse("ERROR: OpenAI API Key is missing. Please paste and set your API key in the Delivery panel of the Web Control Dashboard.")
        try:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a robot assistant."},
                    {"role": "user", "content": req.prompt}
                ]
            )

            answer = resp["choices"][0]["message"]["content"]
            return LLMQueryResponse(answer)

        except Exception as e:
            rospy.logerr(f"OpenAI error: {e}")
            return LLMQueryResponse(f"ERROR: {e}")

    def handle_vision(self, req):
        if not self.load_dynamic_key():
            rospy.logwarn("LLM Vision query failed: No OpenAI API key loaded.")
            return LLMVisionQueryResponse("ERROR: OpenAI API Key is missing. Please paste and set your API key in the Delivery panel of the Web Control Dashboard.")
        try:
            resp = openai.ChatCompletion.create(
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

            answer = resp["choices"][0]["message"]["content"]
            return LLMVisionQueryResponse(answer)

        except Exception as e:
            rospy.logerr(f"OpenAI Vision error: {e}")
            return LLMVisionQueryResponse(f"ERROR: {e}")


if __name__ == "__main__":
    rospy.init_node("stateless_llm_client")
    node = StatelessLLMClient()
    rospy.spin()
