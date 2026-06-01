#include "ros/ros.h"
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <tf2_ros/transform_listener.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <teb_local_planner/teb_local_planner_ros.h>
#include <memory>
#include <cmath>

class TebPlannerNode {
public:
    TebPlannerNode(ros::NodeHandle& nh) : nh_(nh), tf_buffer_(ros::Duration(60.0)), tf_listener_(tf_buffer_), plan_divider_(0) {
        // Subscribers & Publishers
        subGoalPoint = nh_.subscribe("/graph_planner/path/global_path", 1, &TebPlannerNode::CallbackGoalPoint, this);
        pubCommand = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);  // No latch: other planners must not receive stale TEB velocities

        // Initialize Costmap and TEB Planner objects gracefully
        try {
            costmap_ros_.reset(new costmap_2d::Costmap2DROS("local_costmap", tf_buffer_));
            teb_planner_.reset(new teb_local_planner::TebLocalPlannerROS());
            teb_planner_->initialize("TebLocalPlannerROS", &tf_buffer_, costmap_ros_.get());
            ROS_INFO("TEB Local Planner initialized successfully.");
        } catch (const std::exception& e) {
            ROS_WARN("Failed to initialize TEB Local Planner: %s", e.what());
        }
    }

    void CallbackGoalPoint(const nav_msgs::Path& msg) {
        if (!teb_planner_) return;
        if (msg.poses.empty()) {
            path_received_ = false;
            last_cmd_vel_ = geometry_msgs::Twist();
            pubCommand.publish(last_cmd_vel_);
            return;
        }
        
        global_path_ = msg;
        path_received_ = true;
        last_closest_idx_ = 0; // reset on new path
        plan_divider_ = 3;     // Trigger plan calculation on next loop iteration
    }

    void Plan() {
        if (!teb_planner_ || !costmap_ros_ || !path_received_) {
            last_cmd_vel_ = geometry_msgs::Twist();
            return;
        }

        // TEB requires the robot's current pose from the costmap
        geometry_msgs::PoseStamped robot_pose;
        if (!costmap_ros_->getRobotPose(robot_pose)) {
            ROS_WARN_THROTTLE(1.0, "Could not get robot pose from costmap. Will not command velocity.");
            last_cmd_vel_ = geometry_msgs::Twist();
            return;
        }

        // Prune the path to prevent the robot from going back to previously passed waypoints or circling
        int closest_idx = last_closest_idx_;
        double min_dist = 999999.0;
        
        int search_start = std::max(0, last_closest_idx_);
        int search_end = std::min((int)global_path_.poses.size(), last_closest_idx_ + 100); // local forward window
        for (int i = search_start; i < search_end; ++i) {
            double dx = global_path_.poses[i].pose.position.x - robot_pose.pose.position.x;
            double dy = global_path_.poses[i].pose.position.y - robot_pose.pose.position.y;
            double dist = sqrt(dx*dx + dy*dy);
            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = i;
            }
        }
        last_closest_idx_ = closest_idx;

        // Feed only the upcoming portion of the global path into TEB
        std::vector<geometry_msgs::PoseStamped> transformed_plan;
        for (size_t i = closest_idx; i < global_path_.poses.size(); ++i) {
            geometry_msgs::PoseStamped p = global_path_.poses[i];
            if (p.header.frame_id.empty()) {
                p.header.frame_id = global_path_.header.frame_id.empty() ? "odom" : global_path_.header.frame_id;
            }
            transformed_plan.push_back(p);
        }

        if (!teb_planner_->setPlan(transformed_plan)) {
            ROS_WARN_THROTTLE(1.0, "Failed to set plan for TEB local planner");
        }

        geometry_msgs::Twist cmd_vel;

        // Arrival rule evaluation
        if (teb_planner_->isGoalReached()) {
            last_cmd_vel_ = geometry_msgs::Twist();
            pubCommand.publish(last_cmd_vel_); // Publish immediate stop command
            path_received_ = false;            // Deactivate repeater
            ROS_INFO_THROTTLE(1.0, "Goal Reached!");
            return;
        }

        // Optimization & Velocity generation step
        if (teb_planner_->computeVelocityCommands(cmd_vel)) {
            // Get obstacle distance for info only
            double min_obstacle_dist = getMinObstacleDistance(robot_pose.pose.position.x, robot_pose.pose.position.y);
            last_cmd_vel_ = cmd_vel;
            ROS_INFO_THROTTLE(1.0, "TEB OK: cmd_vel [vx: %.2f, vy: %.2f, w: %.2f] (obstacle_dist: %.2f m)", 
                              cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.angular.z, min_obstacle_dist);
        } else {
            ROS_WARN_THROTTLE(1.0, "TEB failed to compute velocity commands.");
            ROS_WARN_THROTTLE(1.0, "Robot is currently stuck at X: %.2f, Y: %.2f", robot_pose.pose.position.x, robot_pose.pose.position.y);
            last_cmd_vel_ = geometry_msgs::Twist(); // Stop on failure
        }
    }

    void SpinOnce() {
        std::string planner_type;
        nh_.param<std::string>("/local_planner_type", planner_type, "control_space");
        if (planner_type != "teb") {
            // Another planner (e.g. auto_pick / control_space) is active.
            // If we were previously repeating, flush a stop command and disarm the repeater
            // so the TEB repeater never overwrites another controller's cmd_vel.
            if (path_received_) {
                path_received_ = false;
                last_cmd_vel_ = geometry_msgs::Twist();
                pubCommand.publish(last_cmd_vel_); // explicit zero-velocity stop
                ROS_INFO_ONCE("[TEB] Non-TEB planner active – repeater disarmed.");
            }
            return;
        }

        // Only repeat/publish if we have an active path/goal we are approaching
        if (!path_received_) return;

        // Check emergency stop condition
        bool is_paused = false;
        nh_.param<bool>("/exploration_paused", is_paused, false);
        std::string state = "IDLE";
        nh_.param<std::string>("/exploration_state", state, "IDLE");
        if (is_paused || state == "IDLE" || state == "STOP" || state == "RECOVERY") {
            last_cmd_vel_ = geometry_msgs::Twist();
            pubCommand.publish(last_cmd_vel_);
            path_received_ = false; // Deactivate repeater
            return;
        }

        // Run planning optimization at 25 Hz (every 2nd iteration of the 50 Hz loop)
        plan_divider_++;
        if (plan_divider_ >= 2) {
            plan_divider_ = 0;
            Plan();
        }

        // Continually publish cmd_vel at 20 Hz (only while path_received_ is true)
        if (path_received_) {
            pubCommand.publish(last_cmd_vel_);
        }
    }

private:
    ros::NodeHandle nh_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::shared_ptr<costmap_2d::Costmap2DROS> costmap_ros_;
    std::shared_ptr<teb_local_planner::TebLocalPlannerROS> teb_planner_;

    ros::Subscriber subGoalPoint;
    ros::Publisher pubCommand;

    nav_msgs::Path global_path_;
    bool path_received_ = false;
    int last_closest_idx_ = 0;
    geometry_msgs::Twist last_cmd_vel_;
    int plan_divider_;

    // Calculate minimum distance to obstacles using costmap
    double getMinObstacleDistance(double robot_x, double robot_y) {
        if (!costmap_ros_) return 1.5; // Safe default
        
        costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
        double min_dist = 1.5;
        
        // Check a circular area around the robot
        unsigned int robot_mx, robot_my;
        if (!costmap->worldToMap(robot_x, robot_y, robot_mx, robot_my)) {
            return 1.5;
        }
        
        // Sample obstacles in expanding radius
        double resolution = costmap->getResolution();
        int search_radius = (int)(1.5 / resolution); // Search 1.5m radius
        
        for (int dx = -search_radius; dx <= search_radius; ++dx) {
            for (int dy = -search_radius; dy <= search_radius; ++dy) {
                unsigned int mx = robot_mx + dx;
                unsigned int my = robot_my + dy;
                
                if (mx < 0 || mx >= costmap->getSizeInCellsX() || 
                    my < 0 || my >= costmap->getSizeInCellsY()) {
                    continue;
                }
                
                unsigned char cost = costmap->getCost(mx, my);
                // Cost > 50 is generally considered an obstacle (varies by costmap config)
                if (cost > 50) {
                    double world_x, world_y;
                    costmap->mapToWorld(mx, my, world_x, world_y);
                    double dist = std::sqrt((world_x - robot_x) * (world_x - robot_x) + 
                                           (world_y - robot_y) * (world_y - robot_y));
                    if (dist < min_dist) {
                        min_dist = dist;
                    }
                }
            }
        }
        
        return min_dist;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "teb_planner_node");
    ros::NodeHandle nh;
    
    TebPlannerNode node(nh);
    ros::Rate rate(50); // 50 Hz continuous repeater rate
    
    while (ros::ok()) {
        ros::spinOnce();
        node.SpinOnce();
        rate.sleep();
    }
    return 0;
}
