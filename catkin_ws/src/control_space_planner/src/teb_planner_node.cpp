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
    TebPlannerNode(ros::NodeHandle& nh) : nh_(nh), tf_listener_(tf_buffer_), consecutive_failures_(0), recovery_state_(0) {
        // Initialize Costmap and TEB Planner objects gracefully first
        try {
            costmap_ros_.reset(new costmap_2d::Costmap2DROS("local_costmap", tf_buffer_));
            teb_planner_.reset(new teb_local_planner::TebLocalPlannerROS());
            teb_planner_->initialize("TebLocalPlannerROS", &tf_buffer_, costmap_ros_.get());
            ROS_INFO("TEB Local Planner initialized successfully.");
        } catch (const std::exception& e) {
            ROS_WARN("Failed to initialize TEB Local Planner: %s", e.what());
        }

        // Subscribers & Publishers initialized at the very end to prevent callbacks firing on uninitialized members
        subGoalPoint = nh_.subscribe("/graph_planner/path/global_path", 1, &TebPlannerNode::CallbackGoalPoint, this);
        pubCommand = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1, false); // DO NOT LATCH cmd_vel
        pubLocalPlan = nh_.advertise<nav_msgs::Path>("/teb_local_plan", 1);
    }

    void CallbackGoalPoint(const nav_msgs::Path& msg) {
        if (msg.poses.empty() || !teb_planner_) return;
        
        // Only reset the closest-index when the path actually changes
        // (different number of poses or different endpoint)
        bool is_new_path = false;
        if (global_path_.poses.size() != msg.poses.size()) {
            is_new_path = true;
        } else if (!global_path_.poses.empty()) {
            auto& old_end = global_path_.poses.back().pose.position;
            auto& new_end = msg.poses.back().pose.position;
            auto& old_start = global_path_.poses.front().pose.position;
            auto& new_start = msg.poses.front().pose.position;
            double d_end = std::hypot(old_end.x - new_end.x, old_end.y - new_end.y);
            double d_start = std::hypot(old_start.x - new_start.x, old_start.y - new_start.y);
            if (d_end > 0.1 || d_start > 0.1) {
                is_new_path = true;
            }
        } else {
            is_new_path = true;
        }
        
        global_path_ = msg;
        path_received_ = true;
        
        if (is_new_path) {
            int start_idx = 0;
            try {
                auto t_map = tf_buffer_.lookupTransform("map", "base_footprint", ros::Time(0), ros::Duration(0.1));
                double min_dist = 999999.0;
                for (size_t i = 0; i < global_path_.poses.size(); ++i) {
                    double dx = global_path_.poses[i].pose.position.x - t_map.transform.translation.x;
                    double dy = global_path_.poses[i].pose.position.y - t_map.transform.translation.y;
                    double dist = std::sqrt(dx*dx + dy*dy);
                    if (dist < min_dist) {
                        min_dist = dist;
                        start_idx = i;
                    }
                }
            } catch (tf2::TransformException& ex) {
                // Ignore and start from 0 if transform fails
            }
            last_closest_idx_ = start_idx;
        }
    }

    void Plan() {
        std::string planner_type;
        nh_.param<std::string>("/local_planner_type", planner_type, "control_space");
        if (planner_type != "teb") return;

        // Check emergency stop condition
        bool is_paused = false;
        nh_.param<bool>("/exploration_paused", is_paused, false);
        std::string state = "IDLE";
        nh_.param<std::string>("/exploration_state", state, "IDLE");
        if (is_paused || state == "IDLE" || state == "STOP" || state == "RECOVERY") {
            if (was_running_) {
                geometry_msgs::Twist cmd_vel;
                cmd_vel.linear.x = 0.0;
                cmd_vel.linear.y = 0.0;
                cmd_vel.angular.z = 0.0;
                pubCommand.publish(cmd_vel);
                was_running_ = false;
            }
            recovery_state_ = 0;
            consecutive_failures_ = 0;
            return;
        }
        
        was_running_ = true;

        if (!teb_planner_ || !costmap_ros_ || !path_received_) return;

        // Direct TF lookup instead of costmap_ros_->getRobotPose().
        // Costmap2DROS internally stores the timestamp of its last successful pose
        // and uses it for subsequent sensor-data TF lookups. Once that timestamp
        // gets stale (e.g. VO stall during recovery), getRobotPose() returns false
        // forever. A direct Time(0) lookup always gets the freshest available data.
        geometry_msgs::PoseStamped robot_pose_map;
        geometry_msgs::PoseStamped robot_pose_odom;
        try {
            auto t_map = tf_buffer_.lookupTransform("map", "base_footprint", ros::Time(0), ros::Duration(0.2));
            auto t_odom = tf_buffer_.lookupTransform("odom", "base_footprint", ros::Time(0), ros::Duration(0.2));
            
            double pose_age = (ros::Time::now() - t_map.header.stamp).toSec();
            if (pose_age > 1.5) {
                if (recovery_state_ == 0) {
                    geometry_msgs::Twist stop;
                    pubCommand.publish(stop);
                    ROS_WARN_THROTTLE(1.0, "TF stale (%.2fs old) — stopping robot to prevent blind driving.", pose_age);
                    return;
                } else {
                    ROS_WARN_THROTTLE(1.0, "TF stale (%.2fs old) during recovery — continuing blind recovery spin.", pose_age);
                }
            }

            robot_pose_map.header = t_map.header;
            robot_pose_map.pose.position.x    = t_map.transform.translation.x;
            robot_pose_map.pose.position.y    = t_map.transform.translation.y;
            robot_pose_map.pose.position.z    = t_map.transform.translation.z;
            robot_pose_map.pose.orientation   = t_map.transform.rotation;
            
            robot_pose_odom.header = t_odom.header;
            robot_pose_odom.pose.position.x    = t_odom.transform.translation.x;
            robot_pose_odom.pose.position.y    = t_odom.transform.translation.y;
            robot_pose_odom.pose.position.z    = t_odom.transform.translation.z;
            robot_pose_odom.pose.orientation   = t_odom.transform.rotation;
        } catch (tf2::TransformException& ex) {
            geometry_msgs::Twist stop;
            pubCommand.publish(stop);
            ROS_WARN_THROTTLE(1.0, "TF stale — stopping robot. (%s)", ex.what());
            return;
        }

        double rx_odom = robot_pose_odom.pose.position.x;
        double ry_odom = robot_pose_odom.pose.position.y;
        double min_obstacle_dist = getMinObstacleDistance(rx_odom, ry_odom);

        geometry_msgs::Twist cmd_vel;

        // 1. Obstacle too close crash-avoidance override / Emergency Brake
        // If an obstacle is closer than 0.35m (5cm from physical footprint edge of 0.30m), trigger automatic emergency stop
        if (min_obstacle_dist < 0.20) {
            ROS_WARN_THROTTLE(0.5, "EMERGENCY BRAKE: Obstacle too close (%.2fm). Stopping robot!", min_obstacle_dist);
            cmd_vel.linear.x = 0.0;
            cmd_vel.linear.y = 0.0;
            cmd_vel.angular.z = 0.0;
            pubCommand.publish(cmd_vel);

            // Trigger global emergency stop
            nh_.setParam("/exploration_paused", true);
            nh_.setParam("/exploration_state", "STOP");

            consecutive_failures_ = 0;
            recovery_state_ = 0;
            return;
        }

        // 2. Active recovery state execution
        if (recovery_state_ == 1) { // Backup recovery phase
            double dt = (ros::Time::now() - recovery_start_time_).toSec();
            if (dt < 1.5) {
                ROS_WARN_THROTTLE(0.5, "Executing Recovery: Backing up (%.1fs / 1.5s)...", dt);
                cmd_vel.linear.x = -0.03;
                cmd_vel.linear.y = 0.0;
                cmd_vel.angular.z = 0.0;
                pubCommand.publish(cmd_vel);
                return;
            } else {
                // Transition to spin phase
                recovery_state_ = 2;
                recovery_start_time_ = ros::Time::now();
            }
        }
        
        if (recovery_state_ == 2) { // Spin recovery phase
            double dt = (ros::Time::now() - recovery_start_time_).toSec();
            if (dt < 2.0) {
                ROS_WARN_THROTTLE(0.5, "Executing Recovery: Spinning in place (%.1fs / 2.0s)...", dt);
                cmd_vel.linear.x = 0.0;
                cmd_vel.linear.y = 0.0;
                // 0.15 rad/s (half of original 0.3): slow enough that RTAB-Map VO
                // can maintain feature tracking through the rotation without stalling.
                cmd_vel.angular.z = 0.15;
                pubCommand.publish(cmd_vel);
                return;
            } else {
                // Recovery cycle complete, try planning again
                recovery_state_ = 0;
                consecutive_failures_ = 0;
                ROS_INFO("Recovery cycle complete. Retrying TEB planner...");
            }
        }

        // Prune the path to prevent the robot from going back to previously passed waypoints or circling
        int closest_idx = last_closest_idx_;
        double min_dist = 999999.0;
        
        int search_start = std::max(0, last_closest_idx_);
        int search_end = std::min((int)global_path_.poses.size(), last_closest_idx_ + 100); // local forward window
        for (int i = search_start; i < search_end; ++i) {
            double dx = global_path_.poses[i].pose.position.x - robot_pose_map.pose.position.x;
            double dy = global_path_.poses[i].pose.position.y - robot_pose_map.pose.position.y;
            double dist = sqrt(dx*dx + dy*dy);
            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = i;
            }
        }
        last_closest_idx_ = closest_idx;

        // Feed only the upcoming portion of the global path into TEB
        std::vector<geometry_msgs::PoseStamped> transformed_plan;
        if (global_path_.poses.size() <= 2) {
            // Do not prune short plans to prevent oscillation between start and goal indices
            for (size_t i = 0; i < global_path_.poses.size(); ++i) {
                geometry_msgs::PoseStamped p = global_path_.poses[i];
                p.header.frame_id = "map";
                p.header.stamp = ros::Time(0); // Use latest available transforms
                transformed_plan.push_back(p);
            }
        } else {
            for (size_t i = closest_idx; i < global_path_.poses.size(); ++i) {
                geometry_msgs::PoseStamped p = global_path_.poses[i];
                p.header.frame_id = "map";
                p.header.stamp = ros::Time(0); // Use latest available transforms
                transformed_plan.push_back(p);
            }
        }

        if (!teb_planner_->setPlan(transformed_plan)) {
            ROS_WARN_THROTTLE(1.0, "Failed to set plan for TEB local planner");
        }

        // Arrival rule evaluation
        if (teb_planner_->isGoalReached()) {
            cmd_vel.linear.x = 0;
            cmd_vel.linear.y = 0;
            cmd_vel.angular.z = 0;
            pubCommand.publish(cmd_vel);
            ROS_INFO_THROTTLE(1.0, "Goal Reached!");
            return;
        }

        // Optimization & Velocity generation step
        if (teb_planner_->computeVelocityCommands(cmd_vel)) {
            consecutive_failures_ = 0; // Reset failure counter on success
            recovery_state_ = 0;
            
            // Adjust turning speed factor based on obstacle distance
            double angular_factor = 0.4 + (std::min(min_obstacle_dist, 1.2) / 1.2) * 0.6; // 0.4 to 1.0
            cmd_vel.angular.z *= angular_factor;
            
            pubCommand.publish(cmd_vel);
            
            // Publish the pruned plan for visualization at ~5 Hz
            plan_publish_counter_++;
            if (plan_publish_counter_ >= 4) {
                plan_publish_counter_ = 0;
                nav_msgs::Path local_plan;
                local_plan.header.stamp = ros::Time::now();
                local_plan.header.frame_id = "map";
                local_plan.poses = transformed_plan;
                pubLocalPlan.publish(local_plan);
            }
            
            ROS_INFO_THROTTLE(1.0, "TEB OK: cmd_vel [vx: %.2f, vy: %.2f, w: %.2f] (obstacle_dist: %.2f m, idx: %d/%zu)", 
                              cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.angular.z, min_obstacle_dist, last_closest_idx_, global_path_.poses.size());
        } else {
            consecutive_failures_++;
            ROS_WARN_THROTTLE(1.0, "TEB planning failed (consecutive: %d/10)", consecutive_failures_);
            
            if (consecutive_failures_ >= 10) {
                // Trigger recovery behavior
                recovery_state_ = 1;
                recovery_start_time_ = ros::Time::now();
                ROS_WARN("Robot is STUCK. Initiating backup & spin recovery strategy...");
            } else {
                // Stop the robot temporarily while waiting for retries
                cmd_vel.linear.x = 0.0;
                cmd_vel.linear.y = 0.0;
                cmd_vel.angular.z = 0.0;
                pubCommand.publish(cmd_vel);
            }
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
    ros::Publisher pubLocalPlan;

    nav_msgs::Path global_path_;
    bool path_received_ = false;
    bool was_running_ = false;
    int plan_publish_counter_ = 0;
    int last_closest_idx_ = 0;

    int consecutive_failures_ = 0;
    int recovery_state_ = 0; // 0: None, 1: Backup, 2: Spin
    ros::Time recovery_start_time_;

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
        
        // Sample obstacles in a tight radius. TEB's own obstacle layer already
        // handles wider clearance; this check is only for the crash-avoidance
        // override (< 0.32 m). Reducing from 1.5m (961 cells) to 0.5m (121 cells)
        // cuts CPU cost ~8× at 20 Hz, preventing TF timeouts that triggered
        // spurious recovery behaviors.
        double resolution = costmap->getResolution();
        int search_radius = (int)(0.5 / resolution); // Search 0.5m radius
        
        for (int dx = -search_radius; dx <= search_radius; ++dx) {
            for (int dy = -search_radius; dy <= search_radius; ++dy) {
                unsigned int mx = robot_mx + dx;
                unsigned int my = robot_my + dy;
                
                if (mx < 0 || mx >= costmap->getSizeInCellsX() || 
                    my < 0 || my >= costmap->getSizeInCellsY()) {
                    continue;
                }
                
                unsigned char cost = costmap->getCost(mx, my);
                // Cost 254 is LETHAL_OBSTACLE (actual physical object).
                // Cost 253 is INSCRIBED_INFLATED_OBSTACLE.
                // We must ignore costs < 254 to avoid detecting the inscribed inflation gradient as a solid wall!
                if (cost == 254) {
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
    ros::NodeHandle private_nh("~");
    
    double planner_frequency = 20.0;
    private_nh.param<double>("planner_frequency", planner_frequency, 20.0);
    ROS_INFO("TEB Planner loop rate set to %.1f Hz", planner_frequency);
    
    TebPlannerNode node(nh);
    ros::Rate rate(planner_frequency);
    
    while (ros::ok()) {
        ros::spinOnce();
        node.Plan();
        rate.sleep();
    }
    return 0;
}
