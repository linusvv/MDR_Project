/*
 * lidar_costmap_node.cpp
 * Simple ROS node to generate a 2D costmap from LaserScan (LIDAR) data.
 *
 * Subscribes: /scan (sensor_msgs/LaserScan)
 * Publishes:  /map/lidar_costmap (nav_msgs/OccupancyGrid)
 */

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Pose.h>
#include <vector>
#include <cmath>

class LidarCostmapNode {
public:
    LidarCostmapNode() : nh_("~") {
        // Parameters
        nh_.param("resolution", resolution_, 0.05); // meters/cell
        nh_.param("size_x", size_x_, 200); // cells
        nh_.param("size_y", size_y_, 200); // cells
        nh_.param("origin_x", origin_x_, -5.0); // meters
        nh_.param("origin_y", origin_y_, -5.0); // meters
        nh_.param("robot_radius", robot_radius_, 0.2); // meters
        nh_.param("obstacle_value", obstacle_value_, 100);
        nh_.param("free_value", free_value_, 0);

        costmap_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/map/lidar_costmap", 1, true);
        scan_sub_ = nh_.subscribe("/scan", 1, &LidarCostmapNode::scanCallback, this);

        // Prepare static costmap meta-data
        costmap_.header.frame_id = "base_link";
        costmap_.info.resolution = resolution_;
        costmap_.info.width = size_x_;
        costmap_.info.height = size_y_;
        costmap_.info.origin.position.x = origin_x_;
        costmap_.info.origin.position.y = origin_y_;
        costmap_.info.origin.position.z = 0.0;
        costmap_.info.origin.orientation.w = 1.0;
        costmap_.data.resize(size_x_ * size_y_, -1); // -1: unknown
    }

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan) {
        // Clear costmap
        std::fill(costmap_.data.begin(), costmap_.data.end(), free_value_);

        // Mark obstacles from scan
        for (size_t i = 0; i < scan->ranges.size(); ++i) {
            float r = scan->ranges[i];
            if (std::isnan(r) || std::isinf(r) || r < scan->range_min || r > scan->range_max)
                continue;
            float angle = scan->angle_min + i * scan->angle_increment;
            float x = r * std::cos(angle);
            float y = r * std::sin(angle);
            int mx = static_cast<int>((x - origin_x_) / resolution_);
            int my = static_cast<int>((y - origin_y_) / resolution_);
            if (mx >= 0 && mx < size_x_ && my >= 0 && my < size_y_)
                costmap_.data[my * size_x_ + mx] = obstacle_value_;
        }
        costmap_.header.stamp = ros::Time::now();
        costmap_pub_.publish(costmap_);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber scan_sub_;
    ros::Publisher costmap_pub_;
    nav_msgs::OccupancyGrid costmap_;
    // Parameters
    double resolution_, origin_x_, origin_y_, robot_radius_;
    int size_x_, size_y_, obstacle_value_, free_value_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_costmap_node");
    LidarCostmapNode node;
    ros::spin();
    return 0;
}
