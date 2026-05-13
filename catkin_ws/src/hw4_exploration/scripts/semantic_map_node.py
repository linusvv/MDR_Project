#!/usr/bin/env python3
import rospy
import json
import tf2_ros
from std_msgs.msg import String

class SemanticMapNode:
    def __init__(self):
        rospy.init_node("semantic_map_node")
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.semantic_map = {
            "stores": {},
            "signboards": []
        }
        
        self.obs_sub = rospy.Subscriber("/semantic_observations", String, self.obs_callback)
        rospy.loginfo("Semantic Map Node Ready.")

    def obs_callback(self, msg):
        try:
            obs = json.loads(msg.data)
            
            # Get current robot pose from TF
            try:
                trans = self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0), rospy.Duration(1.0))
                x = trans.transform.translation.x
                y = trans.transform.translation.y
                
                obs["location"] = {"x": x, "y": y}
                
                # Store the observation
                self.semantic_map["signboards"].append(obs)
                rospy.loginfo(f"Added to semantic map: {obs}")
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF Error in semantic map: {e}")
                
        except json.JSONDecodeError:
            pass

if __name__ == "__main__":
    try:
        node = SemanticMapNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
