#include <ros/ros.h>
#include <ros/package.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <apriltag_ros/AprilTagDetectionArray.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_msgs/TFMessage.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <string>
#include <yaml-cpp/yaml.h>

#define DEG2RAD(deg) deg/180.0*M_PI

class TAGS
{
public:
    std::vector<std::string> _names;
    std::vector<double> _sizes;
    std::vector<tf2::Transform> _transforms;
};

class APRILTAG_LOCALIZATION
{
private:
    ros::NodeHandle* _nh;
    ros::Rate _rate;

    bool tag_config_loaded = false;
    bool camera_transform_found = false;
    bool tf_broadcaster_initialized = false;

    std::string package_path;
    std::string tag_config_name;
    double threshold_dist = 1.5;

    bool loadTagConfig(std::string file_name);
    bool initTFBroadcaster();

    void tf_static_cb(tf2_msgs::TFMessage msg);

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    tf2::Transform orientation_correction;
    
protected:
    ros::Publisher apriltag_localization_pub;
    ros::Publisher global_tag_pose_pub;

    ros::Subscriber tag_detections_sub;
    ros::Subscriber tf_static_sub;

    tf2_ros::StaticTransformBroadcaster static_tf_broadcaster_;
    std::vector<geometry_msgs::TransformStamped> tag_transforms;
    tf2_ros::TransformBroadcaster true_rt_tf_broadcaster;

    std::string world_frame_name, camera_frame_name, image_frame_name;
    geometry_msgs::TransformStamped cam2optical_geo;
    tf2::Transform cam2optical_tf;
    bool broadcast_world2cam_tf = false;

    apriltag_ros::AprilTagDetectionArray tag_detection;
    geometry_msgs::PoseStamped apriltag_localization;

    TAGS tag_rts;

public:
    APRILTAG_LOCALIZATION(ros::NodeHandle* nh, ros::Rate rate);
    
    bool getTrueRT();
};