#ifndef __WALL_FOLLOWER_NODE_HPP__
#define __WALL_FOLLOWER_NODE_HPP__

#include <iostream>
#include <cmath>
#include <vector>

// ROS
#include "ros/ros.h"
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>

class WallFollower
{
  public:
    WallFollower(ros::NodeHandle& nh);
    ~WallFollower();

    // Callbacks
    void CallbackOccupancyGrid(const nav_msgs::OccupancyGrid& msg);
    void CallbackOdometry(const nav_msgs::Odometry& msg);

    // Core Loop
    void FollowWall();

    // Parameters
    double DESIRED_DISTANCE_FROM_WALL = 0.6; // [m] increased to prevent physical hitting
    double MAX_TRANSLATION_VEL = 0.3; // [m/s]
    double MAX_ROTATION_VEL = 1.0; // [rad/s]
    double K_p_dist = 1.5;
    double K_p_yaw = 2.0;

    double smooth_dist_ = -1.0;
    double smooth_angle_ = 0.0;
    double smooth_front_dist_ = -1.0;

    enum State {
        SEARCHING,
        APPROACHING,
        ALIGNING,
        FOLLOWING
    };

  private:
    ros::NodeHandle nh_;

    // Subscribers
    ros::Subscriber sub_occupancy_grid_;
    ros::Subscriber sub_odom_;

    // Publisher
    ros::Publisher pub_cmd_vel_;

    // State
    State current_state_ = SEARCHING;
    nav_msgs::OccupancyGrid local_map_;
    nav_msgs::Odometry ego_odom_;
    bool has_map_ = false;
    bool has_odom_ = false;

    // Helper functions
    bool findClosestWall(double& min_dist, double& angle_to_wall, double& front_dist);
    double normalizePiToPi(double angle);
};

#endif // __WALL_FOLLOWER_NODE_HPP__
