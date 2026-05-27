#include <apriltag_localization/apriltag_localization.h>

APRILTAG_LOCALIZATION::APRILTAG_LOCALIZATION(ros::NodeHandle* nh, ros::Rate rate): _nh(nh), _rate(rate),
    tf_buffer_(), tf_listener_(tf_buffer_)
{
    package_path = ros::package::getPath("apriltag_localization");

    _nh->getParam(ros::this_node::getName()+ "/tag_config_name", tag_config_name);
    _nh->getParam(ros::this_node::getName()+ "/threshold_dist", threshold_dist);
    _nh->getParam(ros::this_node::getName()+ "/world_frame_name", world_frame_name);
    _nh->getParam(ros::this_node::getName()+ "/camera_frame_name", camera_frame_name);
    _nh->getParam(ros::this_node::getName()+ "/image_frame_name", image_frame_name);
    _nh->getParam(ros::this_node::getName()+ "/broadcast_world2cam_tf", broadcast_world2cam_tf);

    apriltag_localization_pub = _nh->advertise<geometry_msgs::PoseStamped>("apriltag_localization_pose", 1);
    global_tag_pose_pub = _nh->advertise<geometry_msgs::PoseStamped>(camera_frame_name, 1);

    // World's front left up coordinate system to tag's right up depth coordinates
    tf2::Quaternion x90(DEG2RAD(0.0), DEG2RAD(90.0), DEG2RAD(0.0));
    tf2::Quaternion y_90(DEG2RAD(-90.0), DEG2RAD(00.0), DEG2RAD(0.0));
    tf2::Quaternion orientation = x90*y_90;

    orientation_correction.setRotation(orientation);
    orientation_correction.setOrigin(tf2::Vector3(0.0, 0.0, 0.0));

    if(loadTagConfig(tag_config_name))
        tag_config_loaded = true;

    // Initialize TF broadcaster
    if(initTFBroadcaster())
        tf_broadcaster_initialized = true;

    try
    {
        ROS_INFO("Waiting for the TF from camera_link frame to camera_optical frame...");
        cam2optical_geo = tf_buffer_.lookupTransform(camera_frame_name, image_frame_name, ros::Time(0), ros::Duration(10));
        camera_transform_found = true;
    }
    catch(tf2::TransformException &ex)
    {
        ROS_ERROR("%s",ex.what());
        ros::Duration(1.0).sleep();
    }

    tag_detection.detections.clear();
}

bool APRILTAG_LOCALIZATION::initTFBroadcaster()
{
    std::vector<geometry_msgs::TransformStamped> static_transforms;

    static_transforms.reserve(tag_rts._transforms.size());

    for (size_t i = 0; i < tag_rts._transforms.size(); i++)
    {
        geometry_msgs::TransformStamped msg;
        msg.header.stamp    = ros::Time::now();
        msg.header.frame_id = world_frame_name;
        msg.child_frame_id  = "W2T" + tag_rts._names.at(i);

        msg.transform = tf2::toMsg(tag_rts._transforms.at(i));

        static_transforms.push_back(msg);
    }

    static_tf_broadcaster_.sendTransform(static_transforms);

    ROS_INFO_STREAM("Static TF for " 
                    << static_transforms.size() 
                    << " tag transforms published once.");

    return true;
}
bool APRILTAG_LOCALIZATION::loadTagConfig(std::string file_name)
{
    std::string file_path = package_path + "/config/" + file_name + ".yaml";

    ROS_INFO_STREAM("Loading tag configuration..." << file_path);

    YAML::Node node;
    try{
        ROS_INFO_STREAM("Loading config: " << file_path);
        node = YAML::LoadFile(file_path);
    }
    catch(std::exception &e)
    {
        ROS_ERROR("Failed to load tag config.");
        return false;

    }

    ROS_INFO_STREAM(node["TAG_TRUE_RT"]["TAGS"].size() << " configs are found");
    
    for(size_t tag = 0; tag < node["TAG_TRUE_RT"]["TAGS"].size(); tag++)
    {
        try
        {
            std::string tag_rt_name = node["TAG_TRUE_RT"]["TAGS"][tag][0].as<std::string>();
            double tag_rt_size = node["TAG_TRUE_RT"]["TAGS"][tag][1].as<double>();

            tf2::Transform tag_rt;
            geometry_msgs::Point location;
            location.x = node["TAG_TRUE_RT"]["TAGS"][tag][2].as<double>();
            location.y = node["TAG_TRUE_RT"]["TAGS"][tag][3].as<double>();
            location.z = node["TAG_TRUE_RT"]["TAGS"][tag][4].as<double>();

            double heading = node["TAG_TRUE_RT"]["TAGS"][tag][5].as<double>();
            tf2::Quaternion heading_quat;
            heading_quat.setRotation(tf2::Vector3(0.0, 0.0, 1.0), DEG2RAD(heading));

            tag_rt.setOrigin(tf2::Vector3(location.x, location.y, location.z));
            tag_rt.setRotation(heading_quat);
            tag_rt = tag_rt*orientation_correction;

            tag_rts._names.push_back(tag_rt_name);
            tag_rts._sizes.push_back(tag_rt_size);
            tag_rts._transforms.push_back(tag_rt);

            ROS_INFO_STREAM("Tag info about " << tag_rt_name << " is loaded successfully.");
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            ROS_ERROR_STREAM("Failed to load info about tag config index: " << tag+1);
            ROS_ERROR_STREAM("Please check formating or typo in file: " << file_name);
        }
    }

    ROS_INFO_STREAM("All " << node["TAG_TRUE_RT"]["TAGS"].size() <<  " tags are successfully loaded!");

    return true;
}

bool APRILTAG_LOCALIZATION::getTrueRT()
{
    if(!tag_config_loaded)
    {
        ROS_ERROR("Configuration is not initialized...");
        return false;
    }

    if(!camera_transform_found)
    {
        ROS_ERROR("Camera transform does not initialized...");
        ROS_WARN("Please check launch arguments defining frames");
        return false;
    }

    std::vector<double>             dist_to_tag;
    std::vector<tf2::Transform>     optical2tag_list;
    std::vector<std::string>        name_list;

    // Selecting minimum distance tag to compute ground truth pose
    //? With only localization Tag environment
    for (size_t i = 0; i < tag_rts._names.size(); ++i)
    {
        const std::string& tag_name  = tag_rts._names[i];
        const std::string  tag_frame = tag_name;

        try
        {
            // geometry_msgs::TransformStamped optical2tag_geo =
            //     tf_buffer_.lookupTransform(image_frame_name,
            //                                 tag_frame,
            //                                 ros::Time(0)); // lookup timeout

            geometry_msgs::TransformStamped optical2tag_geo =
                tf_buffer_.lookupTransform(image_frame_name, tag_frame, ros::Time(0));
            
            ros::Time now = ros::Time::now();
            if ((now - optical2tag_geo.header.stamp).toSec() > 0.5)
            {
                ROS_WARN("Tag %s transform is old. Skip.", tag_frame.c_str());
                continue;
            }

            const double x = optical2tag_geo.transform.translation.x;
            const double y = optical2tag_geo.transform.translation.y;
            const double z = optical2tag_geo.transform.translation.z;

            const double dist = std::sqrt(x*x + y*y + z*z);

            tf2::Transform optical2tag_tf;
            tf2::fromMsg(optical2tag_geo.transform, optical2tag_tf);

            dist_to_tag.push_back(dist);
            optical2tag_list.push_back(optical2tag_tf);
            name_list.push_back(tag_name);
        }
        catch (tf2::TransformException& ex)
        {
            // ROS_DEBUG("%s", ex.what());
            continue;
        }
    }

    // If no tag detected
    if (dist_to_tag.empty())
    {
        ROS_INFO("No tag bundle frames visible in TF. Skip localization.");
        return false;
    }
    
    auto min_it   = std::min_element(dist_to_tag.begin(), dist_to_tag.end());
    int  min_idx  = std::distance(dist_to_tag.begin(), min_it);
    double min_dist = *min_it;

    const std::string& selected_name = name_list[min_idx];
    if (min_dist > threshold_dist)
    {
        ROS_INFO_STREAM("Closest tag bundle '" << selected_name
                        << "' is too far. Dist: " << min_dist
                        << ", threshold: " << threshold_dist);
        ROS_INFO("Drop current localization...");
        return false;
    }
    else
    {
        ROS_INFO_STREAM("Tag bundle '" << selected_name
                        << "' selected at dist " << min_dist);
    }

    // Find config index
    auto it = std::find(tag_rts._names.begin(), tag_rts._names.end(), selected_name);
    if (it == tag_rts._names.end())
    {
        ROS_ERROR_STREAM("Selected tag bundle name '" << selected_name << "' not found in configuration.");
        return false;
    }
    size_t config_idx = std::distance(tag_rts._names.begin(), it);
    ROS_INFO_STREAM("Tag bundle config found at index " << config_idx);

    // map -> tag transformation
    tf2::Transform global2tag_tf = tag_rts._transforms.at(config_idx);

    geometry_msgs::PoseStamped global2tag_geo;
    tf2::toMsg(global2tag_tf, global2tag_geo.pose);
    global2tag_geo.header.frame_id = world_frame_name;
    global2tag_geo.header.stamp    = ros::Time::now();
    apriltag_localization_pub.publish(global2tag_geo);

    // optical -> tag
    tf2::Transform optical2tag_tf = optical2tag_list[min_idx];

    // cam mount -> optical transformation
    tf2::fromMsg(cam2optical_geo.transform, cam2optical_tf);

    // map -> camera mount transformation
    tf2::Transform global2_cam_tf = global2tag_tf * optical2tag_tf.inverse() * cam2optical_tf.inverse();

    // map -> camera pose publish
    geometry_msgs::PoseStamped global2cam_pose;
    tf2::toMsg(global2_cam_tf, global2cam_pose.pose);
    global2cam_pose.header.stamp    = ros::Time::now();
    global2cam_pose.header.frame_id = world_frame_name;
    global_tag_pose_pub.publish(global2cam_pose);

    // map -> camera TF broadcast
    if (broadcast_world2cam_tf)
    {
        geometry_msgs::TransformStamped global2_cam_geo;
        global2_cam_geo.header.frame_id = world_frame_name;
        global2_cam_geo.header.stamp    = ros::Time::now();
        global2_cam_geo.child_frame_id  = camera_frame_name;
        global2_cam_geo.transform.translation.x = global2cam_pose.pose.position.x;
        global2_cam_geo.transform.translation.y = global2cam_pose.pose.position.y;
        global2_cam_geo.transform.translation.z = global2cam_pose.pose.position.z;
        global2_cam_geo.transform.rotation      = global2cam_pose.pose.orientation;
        true_rt_tf_broadcaster.sendTransform(global2_cam_geo);
    }

    ROS_INFO_STREAM("Global pose of camera (bundle-based): " << global2cam_pose.pose);

    return true;
}