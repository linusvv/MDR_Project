/*  Copyright (C) Amir Darwesh
 * 
 *  License: Modified BSD Software License 
 */

#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAP_IDX(sx, i, j) ((sx) * (j) + (i))

class HeightmapToCostMap
{
public:
    HeightmapToCostMap();
    void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
    void rtabmap_cb(const nav_msgs::OccupancyGridConstPtr &msg);
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

    float RTABMAP_OVERLAY_RADIUS = 4.0; // [m]
    float CAMERA_FOV_ANGLE = 45.0; // [deg]

private:
    ros::NodeHandle nh_;
    std::string cloud_topic_; //default input
    std::string map_topic_;
    ros::Subscriber sub_;
    ros::Subscriber rtabmap_sub_;
    ros::Publisher cost_map_pub_;
    ros::Publisher viz_map_pub_;   // Diagnostic: camera=75, RTABMap=50, overlap=100

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Variables
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz;
    nav_msgs::OccupancyGridConstPtr rtabmap_msg_;

    bool bGetPoint = false;
    ros::Time last_process_time_;
    bool first_frame_ = true;
    geometry_msgs::TransformStamped last_map_tf_;
    int decay_frame_counter_ = 0;   // Decay hit_count every 2nd frame (slows ghost removal)

    // Grid sizes
    int width_;
    int height_;

    // Persistent memory
    std::vector<int> hit_count_grid_;
};

HeightmapToCostMap::HeightmapToCostMap() : cloud_topic_("/points/velodyne_obstacles"), map_topic_("/map/local_map/obstacle"), tf_listener_(tf_buffer_)
{
    sub_ = nh_.subscribe(cloud_topic_, 1, &HeightmapToCostMap::cloud_cb, this);
    rtabmap_sub_ = nh_.subscribe("/rtabmap/grid_map", 1, &HeightmapToCostMap::rtabmap_cb, this);
    cost_map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(map_topic_, 10);
    viz_map_pub_  = nh_.advertise<nav_msgs::OccupancyGrid>("/map/local_map/obstacle_viz", 10);

    float w = MAP_MAX_X - MAP_MIN_X + 0.5f;
    width_ = int(w / RESOLUTION_ + 0.5f);
    float h = MAP_MAX_Y - MAP_MIN_Y + 0.5f;
    height_ = int(h / RESOLUTION_ + 0.5f);

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

void HeightmapToCostMap::rtabmap_cb(const nav_msgs::OccupancyGridConstPtr &msg)
{
    rtabmap_msg_ = msg;
}

void HeightmapToCostMap::generate_costmap()
{
    if (bGetPoint)
    {
        ros::Time current_time = ros::Time::now();
        std::string target_frame = cloud_xyz->header.frame_id;
        
        // --- 1. SPATIAL MEMORY SHIFT (Odometry compensation) ---
        geometry_msgs::TransformStamped current_map_tf;
        try {
            // CRITICAL: Use 'map' frame instead of 'odom'. 
            // If the robot is physically moved, 'odom' (wheel encoders) doesn't change, but 'map' (localization) does!
            current_map_tf = tf_buffer_.lookupTransform("map", target_frame, ros::Time(0));
        } catch (tf2::TransformException &ex) {
            ROS_WARN_THROTTLE(1.0, "[Costmap Memory] TF Error, cannot shift memory: %s", ex.what());
            bGetPoint = false;
            return;
        }

        if (!first_frame_) {
            // Calculate relative transform mathematically to avoid TF history timeout issues.
            // T_inv = T_last_inv * T_current
            tf2::Transform T_current;
            tf2::fromMsg(current_map_tf.transform, T_current);

            tf2::Transform T_last;
            tf2::fromMsg(last_map_tf_.transform, T_last);

            tf2::Transform T_inv = T_last.inverse() * T_current;
            geometry_msgs::TransformStamped t_inv_msg;
            t_inv_msg.transform = tf2::toMsg(T_inv);

            std::vector<int> new_hit_grid(width_ * height_, 0);

            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    double px_new = MAP_MIN_X + (x + 0.5) * RESOLUTION_;
                    double py_new = MAP_MIN_Y + (y + 0.5) * RESOLUTION_;

                    geometry_msgs::Point pt_new;
                    pt_new.x = px_new; pt_new.y = py_new; pt_new.z = 0.0;
                    geometry_msgs::Point pt_old;
                    tf2::doTransform(pt_new, pt_old, t_inv_msg);

                    int x_old = int((pt_old.x - MAP_MIN_X) / RESOLUTION_);
                    int y_old = int((pt_old.y - MAP_MIN_Y) / RESOLUTION_);

                    if (x_old >= 0 && x_old < width_ && y_old >= 0 && y_old < height_) {
                        int idx_new = MAP_IDX(width_, x, y);
                        int idx_old = MAP_IDX(width_, x_old, y_old);
                        new_hit_grid[idx_new] = hit_count_grid_[idx_old];
                    }
                }
            }
            hit_count_grid_ = new_hit_grid;
        }
        first_frame_ = false;
        last_process_time_ = current_time;
        last_map_tf_ = current_map_tf;

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

        // --- 3. TEMPORAL FILTER: persistent memory with slow decay ---
        const int MAX_HIT_COUNT = 24;  // ~3 s at 8 Hz before an obstacle fully fades
        const int MIN_SHOW_COUNT = 2;  // 2 consecutive hits required to confirm
        decay_frame_counter_++;
        bool do_decay = (decay_frame_counter_ % 2 == 0); // Decay every 2nd frame (~4 Hz)

        nav_msgs::MapMetaData mapMeta;
        mapMeta.resolution = RESOLUTION_;
        mapMeta.width = width_;
        mapMeta.height = height_;
        geometry_msgs::Pose oPose;
        oPose.position.x = MAP_MIN_X - RESOLUTION_/2;
        oPose.position.y = MAP_MIN_Y - RESOLUTION_/2;
        mapMeta.origin = oPose;

        nav_msgs::OccupancyGrid oMap;
        nav_msgs::OccupancyGrid vizMap;  // Diagnostic map: camera=75, RTABMap=50
        oMap.info = mapMeta;
        oMap.data.assign(width_ * height_, 0);
        oMap.header.frame_id = target_frame;
        oMap.header.stamp = current_time;
        vizMap.info = mapMeta;
        vizMap.data.assign(width_ * height_, 0);
        vizMap.header.frame_id = target_frame;
        vizMap.header.stamp = current_time;

        int point_count = 0;
        for (int i = 0; i < width_ * height_; ++i) {
            if (hits_this_frame[i]) {
                if (hit_count_grid_[i] < MAX_HIT_COUNT) hit_count_grid_[i]++;
            } else {
                if (do_decay && hit_count_grid_[i] > 0) hit_count_grid_[i]--;
            }

            // Show confirmed memory OR immediate single-frame hit for safety
            bool show = (hit_count_grid_[i] >= MIN_SHOW_COUNT) || hits_this_frame[i];
            if (show) {
                oMap.data[i] = 100;
                vizMap.data[i] = 75;  // Camera / depth-sensor obstacle
                point_count++;
            }
        }

        // --- 3.5 OVERLAY RTAB-MAP IN BLIND SPOTS (outside FOV +-45 deg) ---
        if (rtabmap_msg_ && !rtabmap_msg_->data.empty()) {
            geometry_msgs::TransformStamped rtab_to_base_tf;
            geometry_msgs::TransformStamped base_to_rtab_tf;
            bool tf_ok = false;
            try {
                rtab_to_base_tf = tf_buffer_.lookupTransform(target_frame, rtabmap_msg_->header.frame_id, ros::Time(0));
                base_to_rtab_tf = tf_buffer_.lookupTransform(rtabmap_msg_->header.frame_id, target_frame, ros::Time(0));
                tf_ok = true;
            } catch (tf2::TransformException &ex) {
                ROS_WARN_THROTTLE(2.0, "[Costmap Memory] TF Error looking up RTAB-Map to %s: %s", target_frame.c_str(), ex.what());
            }

            if (tf_ok) {
                double res = rtabmap_msg_->info.resolution;
                double origin_x = rtabmap_msg_->info.origin.position.x;
                double origin_y = rtabmap_msg_->info.origin.position.y;
                unsigned int rtab_w = rtabmap_msg_->info.width;
                unsigned int rtab_h = rtabmap_msg_->info.height;

                // Robot coordinates in the RTABmap frame
                double robot_x_rtab = base_to_rtab_tf.transform.translation.x;
                double robot_y_rtab = base_to_rtab_tf.transform.translation.y;

                double rtab_radius = RTABMAP_OVERLAY_RADIUS;
                double fov_rad = CAMERA_FOV_ANGLE * (M_PI / 180.0);

                // Bounding box in metric coordinates
                double min_x = robot_x_rtab - rtab_radius;
                double max_x = robot_x_rtab + rtab_radius;
                double min_y = robot_y_rtab - rtab_radius;
                double max_y = robot_y_rtab + rtab_radius;

                // Convert bounding box to RTAB map grid indices
                int start_x = std::max(0, (int)floor((min_x - origin_x) / res));
                int end_x = std::min((int)rtab_w - 1, (int)ceil((max_x - origin_x) / res));
                int start_y = std::max(0, (int)floor((min_y - origin_y) / res));
                int end_y = std::min((int)rtab_h - 1, (int)ceil((max_y - origin_y) / res));

                for (int gy = start_y; gy <= end_y; ++gy) {
                    for (int gx = start_x; gx <= end_x; ++gx) {
                        int rtab_idx = gy * rtab_w + gx;
                        int8_t cell_val = rtabmap_msg_->data[rtab_idx];
                        if (cell_val > 50) {
                            // Center coordinates in RTABmap frame
                            geometry_msgs::Point pt_rtab;
                            pt_rtab.x = origin_x + (gx + 0.5) * res;
                            pt_rtab.y = origin_y + (gy + 0.5) * res;
                            pt_rtab.z = 0.0;

                            // Transform to robot base frame
                            geometry_msgs::Point pt_base;
                            tf2::doTransform(pt_rtab, pt_base, rtab_to_base_tf);

                            // Check distance
                            double dist_sq = pt_base.x * pt_base.x + pt_base.y * pt_base.y;
                            if (dist_sq <= rtab_radius * rtab_radius) {
                                // Check angle relative to robot heading (x-axis in base_footprint)
                                double angle = atan2(pt_base.y, pt_base.x);
                                if (std::abs(angle) > fov_rad) { // Outside camera FOV
                                    // Map pt_base to our local costmap grid
                                    int cx = int((pt_base.x - MAP_MIN_X) / RESOLUTION_);
                                    int cy = int((pt_base.y - MAP_MIN_Y) / RESOLUTION_);
                                    if (cx >= 0 && cx < width_ && cy >= 0 && cy < height_) {
                                        int tidx = MAP_IDX(width_, cx, cy);
                                        if (oMap.data[tidx] != 100) {
                                            oMap.data[tidx] = 100;
                                            point_count++;
                                        }
                                        // Viz: RTABMap cell = 50, camera wins (75 stays if already set)
                                        if (vizMap.data[tidx] == 0) {
                                            vizMap.data[tidx] = 50;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
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
        viz_map_pub_.publish(vizMap);
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
