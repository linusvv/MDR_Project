/*
    Copyright (c) 2015, Damian Barczynski <daan.net@wp.eu>
    Following tool is licensed under the terms and conditions of the ISC license.
    For more information visit https://opensource.org/licenses/ISC.
*/
#ifndef __CONTROL_SPACE_NODE_HPP__
#define __CONTROL_SPACE_NODE_HPP__

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>

// pcl
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "tf/transform_datatypes.h"
#include <tf2_msgs/TFMessage.h>
#include <tf/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
// ros
#include "ros/ros.h"
#include <nav_msgs/GetMap.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Int16.h>
#include <std_msgs/Bool.h>
#include <std_msgs/UInt8MultiArray.h>
#include <visualization_msgs/Marker.h>
#include <nav_msgs/OccupancyGrid.h>
#include <ackermann_msgs/AckermannDrive.h>

// Utils
double normalizePiToPi(float angle)
{
  return std::fmod(angle + M_PI, 2 * M_PI) - M_PI; // (angle + pi) % (2 * pi) - pi;
}

class Node
{
    public:
        // The default constructor for 3D array initialization
        Node(): Node(0, 0, 0, 0, 0, 0, 0, 0, -1, false, 0.0, 0.0) {}
        // Constructor for a node with the given arguments
        Node(double x, double y, double z, double yaw, double delta,
             double cost_control, double cost_colli, double cost_total,
             int idx, bool collision, double v = 0.0, double w = 0.0) {
            this->x = x;
            this->y = y;
            this->z = z;
            this->yaw = yaw;
            this->delta = delta;
            this->cost_control = cost_control;
            this->cost_colli = cost_colli;
            this->cost_total = cost_total;
            this->idx = idx;
            this->collision = collision;
            this->v = v;
            this->w = w;
        }

        // the x position
        double x;
        // the y position
        double y;
        // the z position
        double z;
        // the heading yaw
        double yaw;
        // the steering angle delta
        double delta;
        // the minimum distance to goal
        double minDistGoal;
        // [cost] the steering control cost
        double cost_control;
        // [cost] the traversability cost
        double cost_colli;
        // [cost] the total cost
        double cost_total;
        // the index on path
        int idx;
        // flag for collision
        bool collision;
        double v;
        double w;
};

class MotionPlanner
{
  public:
    MotionPlanner(ros::NodeHandle& nh);        
    ~MotionPlanner();
    // ROS node
    ros::NodeHandle nh_;
    // Callback
    void CallbackOccupancyGrid(const nav_msgs::OccupancyGrid& msg);
    void CallbackGoalPoint(const nav_msgs::Path& msg);
    void CallbackEgoOdom(const nav_msgs::Odometry& msg);

    // Publihsher
    void PublishSelectedMotion(std::vector<Node> motionMinCost);
    void PublishMotionPrimitives(std::vector<std::vector<Node>> motionPrimitives);
    void PublishCommand(std::vector<Node> motionMinCost);
    void StopRobot();
    // Algorithms
    void Plan();
    std::vector<std::vector<Node>> GenerateMotionPrimitives(nav_msgs::OccupancyGrid localMap);
    std::vector<Node> RolloutMotion(Node startNode, double maxProgress, double v, double w, nav_msgs::OccupancyGrid localMap);
    std::vector<Node> SelectMotion(std::vector<std::vector<Node>> motionPrimitives);
    bool CheckCollision(Node goalNode, nav_msgs::OccupancyGrid localMap);
    int GetMaxOccupancy(Node goalNode, nav_msgs::OccupancyGrid localMap);
    bool CheckRunCondition();
    Node GlobalToLocalCoordinate(Node globalNode, nav_msgs::Odometry egoOdom);
    geometry_msgs::PoseStamped GlobalToLocalCoordinate(geometry_msgs::PoseStamped poseGlobal, nav_msgs::Odometry egoOdom);


    // Utils
    Node LocalToPlannerCorrdinate(Node nodeLocal);
    void PublishData(std::vector<Node> motionMinCost, std::vector<std::vector<Node>> motionPrimitives);
    
    // TODO: define necessary parameters below
    // Parameters (Map)
    double mapMinX = -5.0;  // Match heightmap_to_costmap exactly
    double mapMaxX = 15.0;
    double mapMinY = -10.0;
    double mapMaxY = 10.0;
    double mapResol = 0.1; // [m / grid]
    int OCCUPANCY_THRES = 90;

    double origin_x = 0.0;
    double origin_y = 0.0;
    std::string frame_id = "base_link";
    
    // Parameters
    double FOV = 85.2 * (M_PI / 180.0); // [rad] FOV of point cloud (realsense)
    double MAX_SENSOR_RANGE = 10.0; // [m] maximum sensor range (realsense)
    double WHEELBASE = 0.05; // [m] Reduced virtual wheelbase to allow tighter turning arcs
    double DIST_RESOL = 0.1; // [m] distance resolution for control space sampling
    double TIME_RESOL = 0.2; // [sec] Doubled time resolution to halve velocity to 0.5 m/s
    double MOTION_VEL = DIST_RESOL / TIME_RESOL; // [m/s] velocity between each motion (for rollout)
    double DELTA_RESOL = 4.0 * (M_PI / 180.0); // Doubled resolution to increase search speed & stability
    double MAX_DELTA = 75.0 * (M_PI / 180.0); // [rad] increased maximum steering angle for U-turns
    double MAX_PROGRESS = 2.0; // Reduced for even more reactive planning in extremely tight corridors

    double ARRIVAL_THRES = 0.3; // [m] further reduced for tighter targets

    // - cost weights
    double W_COST_DIRECTION      = 0.5; // Reduced further to favor safety over heading
    double W_COST_TRAVERSABILITY = 800.0; // Increased to prioritize wall avoidance above all else
    
    // - collision checking
    double INFLATION_SIZE = 3; // 3x3 footprint (30cm) to account for robot physical width
    double LOOKAHEAD_DIST = 2.5; // [m] increased lookahead for smoother tracking

    // Motion primitives
    std::vector<std::vector<Node>> motionCandidates;
    
  private:
    // Input
    ros::Subscriber subOccupancyGrid;
    ros::Subscriber subEgoOdom;
    ros::Subscriber subGoalPoint;
    
    // Output
    ros::Publisher pubSelectedMotion;
    ros::Publisher pubMotionPrimitives;
    ros::Publisher pubCommand;
    ros::Publisher pubAckermannCommand;
    ros::Publisher pubTruncTarget;
    
    Node goalNode;

    // I/O Data
    nav_msgs::OccupancyGrid localMap;
    nav_msgs::Odometry egoOdom;
    nav_msgs::Path globalPath;

    int last_closest_idx = 0;

    geometry_msgs::PoseStamped goalPose; // target goal point 
    Node localNode; // LOS target goal point 
    Node truncLocalNode; // truncated LOS target goal point 
    
    double goal_x = 0.0;
    double goal_y = 0.0;
    double goal_yaw = 0.0;

    double trunc_local_x = 0.0; // truncated pose within local map range
    double trunc_local_y = 0.0; // truncated pose within local map range
    double trunc_local_yaw = 0.0; // truncated pose within local map range

    double ego_x = 0.0;
    double ego_y = 0.0;
    double ego_yaw = 0.0;

    // Signal checker
    bool bGetMap = false;
    bool bGetGoal = false;
    bool bGetLocalNode = false;
    bool bGetEgoOdom = false;

};


#endif // __CONTROL_SPACE_NODE_HPP__