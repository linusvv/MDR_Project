/*  Copyright (C) Amir Darwesh
 * 
 *  License: Modified BSD Software License 
 */

#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common_headers.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/MapMetaData.h>
#include <geometry_msgs/Pose.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#define MAP_IDX(sx, i, j) ((sx) * (j) + (i))

class HeightmapToCostMap
{
public:
    HeightmapToCostMap();
    void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
    void generate_costmap();

    bool DO_INFLATION = false; // Turned off to drastically reduce CPU usage; Costmap2DROS will handle inflation.
    float RESOLUTION_ = 0.1; // [m / cell]
    float MAP_MIN_X =  -5; // map min x position
    float MAP_MAX_X =  15; // map max x position
    float MAP_MIN_Y = -10; // map min y position
    float MAP_MAX_Y =  10; // map max y position

    float MIN_OBSTACLE_HEIGHT = 0.02; // [m] ignore near-zero floor reflections
    float MAX_OBSTACLE_HEIGHT = 0.10; // [m] robot body clearance — ignore anything above this (floating signs, table tops)

    float INFLATION_RADIUS = 0.35; // [m] Drastically reduced so the robot is not terrified of walls
    float INFLATION_RES    = RESOLUTION_; // [m] resolution of inflation
    int INFLATION_BINS     = (INFLATION_RADIUS * 2) / INFLATION_RES + 1;

private:
    ros::NodeHandle nh_;
    std::string cloud_topic_; //default input
    std::string map_topic_;
    ros::Subscriber sub_;
    ros::Publisher cost_map_pub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Variables
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz;

    bool bGetPoint = false;
    ros::Time last_process_time_;
    bool first_frame_ = true;

    // Grid sizes
    int width_;
    int height_;

    // Persistent memory
    std::vector<int> ttl_grid_;
    std::vector<int> hit_count_grid_;
};

HeightmapToCostMap::HeightmapToCostMap() : cloud_topic_("/points/velodyne_obstacles"), map_topic_("/map/local_map/obstacle"), tf_listener_(tf_buffer_)
{
    sub_ = nh_.subscribe(cloud_topic_, 1, &HeightmapToCostMap::cloud_cb, this);
    cost_map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(map_topic_, 10);

    float w = MAP_MAX_X - MAP_MIN_X + 0.5f;
    width_ = int(w / RESOLUTION_ + 0.5f);
    float h = MAP_MAX_Y - MAP_MIN_Y + 0.5f;
    height_ = int(h / RESOLUTION_ + 0.5f);

    ttl_grid_.assign(width_ * height_, 0);
    hit_count_grid_.assign(width_ * height_, 0);

    ROS_INFO("[HeightmapToCostMap] Loaded! Map size: %d x %d with memory filter enabled.", width_, height_);
}

void HeightmapToCostMap::cloud_cb(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud_xyz_);
    cloud_xyz = cloud_xyz_;
    bGetPoint = true;
}

void HeightmapToCostMap::generate_costmap()
{
    if (bGetPoint)
    {
        ros::Time current_time = ros::Time::now();
        std::string target_frame = cloud_xyz->header.frame_id;
        
        // --- 1. SPATIAL MEMORY SHIFT (Odometry compensation) ---
        if (!first_frame_) {
            try {
                // Find how the robot moved by looking up where the OLD frame is in the CURRENT frame.
                // Inverse-mapping: target=OLD, source=CURRENT. 
                // This lets us perfectly query where a physical world point in the CURRENT grid was previously mapped.
                geometry_msgs::TransformStamped t_inv = tf_buffer_.lookupTransform(
                    target_frame, last_process_time_, 
                    target_frame, current_time, 
                    "odom", ros::Duration(0.1));
                
                std::vector<int> new_ttl_grid(width_ * height_, 0);
                std::vector<int> new_hit_grid(width_ * height_, 0);

                for (int y = 0; y < height_; ++y) {
                    for (int x = 0; x < width_; ++x) {
                        double px_new = MAP_MIN_X + (x + 0.5) * RESOLUTION_;
                        double py_new = MAP_MIN_Y + (y + 0.5) * RESOLUTION_;

                        geometry_msgs::Point pt_new;
                        pt_new.x = px_new; pt_new.y = py_new; pt_new.z = 0.0;
                        geometry_msgs::Point pt_old;
                        tf2::doTransform(pt_new, pt_old, t_inv);

                        int x_old = int((pt_old.x - MAP_MIN_X) / RESOLUTION_);
                        int y_old = int((pt_old.y - MAP_MIN_Y) / RESOLUTION_);

                        if (x_old >= 0 && x_old < width_ && y_old >= 0 && y_old < height_) {
                            int idx_new = MAP_IDX(width_, x, y);
                            int idx_old = MAP_IDX(width_, x_old, y_old);
                            new_ttl_grid[idx_new] = ttl_grid_[idx_old];
                            new_hit_grid[idx_new] = hit_count_grid_[idx_old];
                        }
                    }
                }
                ttl_grid_ = new_ttl_grid;
                hit_count_grid_ = new_hit_grid;
            } catch (tf2::TransformException &ex) {
                // If TF fails, we just don't shift. Small penalty during stalls.
            }
        }
        first_frame_ = false;
        last_process_time_ = current_time;

        // --- 2. POPULATE CURRENT HITS ---
        std::vector<bool> hits_this_frame(width_ * height_, false);
        for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cloud_xyz->begin(); it != cloud_xyz->end(); it++)
        {
            if ((!(isnan(it->x) | isnan(it->y))) && 
                (it->x >= MAP_MIN_X && it->x < MAP_MAX_X) && 
                (it->y >= MAP_MIN_Y && it->y < MAP_MAX_Y) &&
                (it->z >= MIN_OBSTACLE_HEIGHT && it->z < MAX_OBSTACLE_HEIGHT))
            {
                // CRITICAL: Filter out the robot footprint (0.32m radius) to prevent self-collision detections!
                if (sqrt(it->x * it->x + it->y * it->y) <= 0.32) continue;

                int x = int((it->x - MAP_MIN_X) / RESOLUTION_);
                int y = int((it->y - MAP_MIN_Y) / RESOLUTION_);
                
                if (x < width_ && x >= 0 && y < height_ && y >= 0)
                {
                    hits_this_frame[MAP_IDX(width_, x, y)] = true;
                }
            }
        }

        // --- 3. TEMPORAL FILTER & TTL DECAY ---
        nav_msgs::MapMetaData mapMeta;
        mapMeta.resolution = RESOLUTION_;
        mapMeta.width = width_;
        mapMeta.height = height_;

        geometry_msgs::Pose oPose;
        oPose.position.x = MAP_MIN_X - RESOLUTION_/2;
        oPose.position.y = MAP_MIN_Y - RESOLUTION_/2;
        mapMeta.origin = oPose;

        nav_msgs::OccupancyGrid oMap;
        oMap.info = mapMeta;
        oMap.data.assign(width_ * height_, 0); 
        oMap.header.frame_id = target_frame; 
        oMap.header.stamp = current_time;

        int point_count = 0;
        for (int i = 0; i < width_ * height_; ++i) {
            if (hits_this_frame[i]) {
                hit_count_grid_[i]++;
                // Require 2 consecutive frames to reject false positives!
                if (hit_count_grid_[i] >= 2) {
                    ttl_grid_[i] = 5; // 0.5 seconds of memory at 10Hz
                }
            } else {
                hit_count_grid_[i] = 0;
            }

            if (ttl_grid_[i] > 0) {
                ttl_grid_[i]--;
                oMap.data[i] = 100;
                point_count++;
            }
        }

        // --- 4. INFLATION ---
        if (point_count > 0 && DO_INFLATION)
        {
            std::vector<int> obs_indices;
            for(int i=0; i < oMap.data.size(); ++i) {
                if(oMap.data[i] == 100) obs_indices.push_back(i);
            }

            int rad_cells = static_cast<int>(INFLATION_RADIUS / RESOLUTION_);
            for(int idx : obs_indices) {
                int ox = idx % width_;
                int oy = idx / width_;

                for(int dy = -rad_cells; dy <= rad_cells; ++dy) {
                    for(int dx = -rad_cells; dx <= rad_cells; ++dx) {
                        int tx = ox + dx;
                        int ty = oy + dy;
                        if(tx >= 0 && tx < width_ && ty >= 0 && ty < height_) {
                            double d = sqrt(dx*dx + dy*dy) * RESOLUTION_;
                            if(d <= INFLATION_RADIUS) {
                                int8_t val = static_cast<int8_t>(100.0 * (1.0 - (d / INFLATION_RADIUS)));
                                int tidx = MAP_IDX(width_, tx, ty);
                                if(val > oMap.data[tidx]) oMap.data[tidx] = val;
                            }
                        }
                    }
                }
            }
        }

        cost_map_pub_.publish(oMap);
        bGetPoint = false;
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "heightmap_to_costmap");

    HeightmapToCostMap hcm; 
    ros::Rate rate(10);
    while (ros::ok())
    {
        ros::spinOnce(); 
        hcm.generate_costmap();
        rate.sleep();
    }
}
