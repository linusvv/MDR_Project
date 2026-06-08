/*  Copyright (C) Amir Darwesh
 * 
 *  License: Modified BSD Software License 
 */


#include <string>
#include <iostream>
#include <algorithm>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common_headers.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/MapMetaData.h>
#include <geometry_msgs/Pose.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <cmath>

const double PI = 3.14159265358979323846;

#define MAP_IDX(sx, i, j) ((sx) * (j) + (i))

class HeightmapToCostMap
{
public:
    HeightmapToCostMap();
    void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
    void rtabmap_cb(const nav_msgs::OccupancyGridConstPtr &map_msg);
    void generate_costmap();

    bool DO_INFLATION = false; // Turned off to drastically reduce CPU usage; Costmap2DROS will handle inflation.
    float RESOLUTION_ = 0.1; // [m / cell]
    float MAP_MIN_X =  -5; // map min x position
    float MAP_MAX_X =  15; // map max x position
    float MAP_MIN_Y = -10; // map min y position
    float MAP_MAX_Y =  10; // map max y position

    float MIN_OBSTACLE_HEIGHT = 0.02; // [m] ignore near-zero floor reflections
    float MAX_OBSTACLE_HEIGHT = 0.20; // [m] robot body clearance — ignore anything above this (floating signs, table tops)

    float INFLATION_RADIUS = 0.6; // [m] Increased to 1.5x (0.4m -> 0.6m) for a larger dangerous zone
    float INFLATION_RES    = RESOLUTION_; // [m] resolution of inflation
    int INFLATION_BINS     = (INFLATION_RADIUS * 2) / INFLATION_RES + 1;

private:
    ros::NodeHandle nh_;
    std::string cloud_topic_; //default input
    std::string map_topic_;
    ros::Subscriber sub_;
    ros::Subscriber rtabmap_sub_;
    ros::Publisher cost_map_pub_;

    // Variables
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz;
    nav_msgs::OccupancyGrid rtabmap_grid_;
    tf::TransformListener tf_listener_;

    bool bGetPoint = false;
    bool bGetRtabmap = false;
    std::vector<int8_t> grid_data;
};

HeightmapToCostMap::HeightmapToCostMap() : cloud_topic_("/points/velodyne_obstacles"), map_topic_("/map/local_map/obstacle")
{
    sub_ = nh_.subscribe(cloud_topic_, 1, &HeightmapToCostMap::cloud_cb, this);
    rtabmap_sub_ = nh_.subscribe("/rtabmap/grid_map", 1, &HeightmapToCostMap::rtabmap_cb, this);

    cost_map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(map_topic_, 10);

    //print some info about the node
    ROS_INFO("[HeightmapToCostMap] Loaded!");
}

void HeightmapToCostMap::rtabmap_cb(const nav_msgs::OccupancyGridConstPtr &map_msg)
{
    rtabmap_grid_ = *map_msg;
    bGetRtabmap = true;
}

void HeightmapToCostMap::cloud_cb(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
{
    // Update point cloud data
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud_xyz_); // conver to pcl object
    cloud_xyz = cloud_xyz_;
    bGetPoint = true;
}

void HeightmapToCostMap::generate_costmap()
{
        if (bGetPoint)
        {
            int width_ = int(MAP_MAX_X - MAP_MIN_X + 0.5f);
            width_ = int(width_ / RESOLUTION_ + 0.5f);

            int height_ = int(MAP_MAX_Y - MAP_MIN_Y + 0.5f);
            height_ = int(height_ / RESOLUTION_ + 0.5f);

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
            oMap.header.frame_id = cloud_xyz->header.frame_id; 
            oMap.header.stamp = ros::Time::now();

            int point_count = 0;
            // Pass 1: Mark direct hits
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
                        oMap.data[MAP_IDX(width_, x, y)] = 100;
                        point_count++;
                    }
                }
            }

            // Pass 1.5: Mark RTABMap obstacles outside the camera's FOV
            if (bGetRtabmap && !rtabmap_grid_.data.empty())
            {
                std::string target_frame = cloud_xyz->header.frame_id;
                std::string source_frame = rtabmap_grid_.header.frame_id;
                tf::StampedTransform transform;
                bool can_transform = false;
                try
                {
                    if (tf_listener_.waitForTransform(target_frame, source_frame, ros::Time(0), ros::Duration(0.1)))
                    {
                        tf_listener_.lookupTransform(target_frame, source_frame, ros::Time(0), transform);
                        can_transform = true;
                    }
                }
                catch (tf::TransformException &ex)
                {
                    ROS_WARN_THROTTLE(5.0, "[HeightmapToCostMap] TF error transforming RTABMap to %s: %s", target_frame.c_str(), ex.what());
                }

                if (can_transform)
                {
                    double res_map = rtabmap_grid_.info.resolution;
                    double origin_map_x = rtabmap_grid_.info.origin.position.x;
                    double origin_map_y = rtabmap_grid_.info.origin.position.y;
                    int width_map = rtabmap_grid_.info.width;
                    int height_map = rtabmap_grid_.info.height;

                    // Project the 1.5m radius region around the robot to map frame to find search bounding box
                    tf::Point robot_pos = transform.inverse() * tf::Point(0, 0, 0);
                    double min_map_x = robot_pos.x() - 1.50;
                    double max_map_x = robot_pos.x() + 1.50;
                    double min_map_y = robot_pos.y() - 1.50;
                    double max_map_y = robot_pos.y() + 1.50;

                    int min_col = std::max(0, int((min_map_x - origin_map_x) / res_map));
                    int max_col = std::min(width_map - 1, int((max_map_x - origin_map_x) / res_map + 1));
                    int min_row = std::max(0, int((min_map_y - origin_map_y) / res_map));
                    int max_row = std::min(height_map - 1, int((max_map_y - origin_map_y) / res_map + 1));

                    for (int r = min_row; r <= max_row; ++r)
                    {
                        for (int c = min_col; c <= max_col; ++c)
                        {
                            int8_t val = rtabmap_grid_.data[r * width_map + c];
                            if (val > 50)
                            {
                                double mx = origin_map_x + (c + 0.5) * res_map;
                                double my = origin_map_y + (r + 0.5) * res_map;

                                tf::Point pt_map(mx, my, 0);
                                tf::Point pt_local = transform * pt_map;

                                if (pt_local.x() >= MAP_MIN_X && pt_local.x() < MAP_MAX_X &&
                                    pt_local.y() >= MAP_MIN_Y && pt_local.y() < MAP_MAX_Y)
                                {
                                    double rtab_dist = sqrt(pt_local.x() * pt_local.x() + pt_local.y() * pt_local.y());
                                    if (rtab_dist <= 0.32 || rtab_dist > 1.50)
                                        continue;

                                    // Check if point is inside camera FOV (using 30 degrees half-angle to close the gap)
                                    bool inside_fov = false;
                                    if (pt_local.x() > 0.0)
                                    {
                                        double angle = atan2(pt_local.y(), pt_local.x());
                                        if (std::abs(angle) <= (30.0 * PI / 180.0))
                                        {
                                            inside_fov = true;
                                        }
                                    }

                                    if (!inside_fov)
                                    {
                                        int lx = int((pt_local.x() - MAP_MIN_X) / RESOLUTION_);
                                        int ly = int((pt_local.y() - MAP_MIN_Y) / RESOLUTION_);
                                        if (lx < width_ && lx >= 0 && ly < height_ && ly >= 0)
                                        {
                                            int idx = MAP_IDX(width_, lx, ly);
                                            if (oMap.data[idx] != 100)
                                            {
                                                oMap.data[idx] = 95;
                                                point_count++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (point_count > 0 && DO_INFLATION)
            {
                // Pass 2: Continuous Gradient Inflation
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
            if (point_count == 0) ROS_INFO_THROTTLE(2, "[HeightmapToCostMap] Published empty map: 0 points survived filter (MIN: %f, MAX: %f)", MIN_OBSTACLE_HEIGHT, MAX_OBSTACLE_HEIGHT);
            
            bGetPoint = false; // WAIT FOR NEW POINTCLOUD BEFORE REPROCESSING
        }
        else
        {
            ROS_INFO_ONCE("No point cloud yet!!!");
        }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "heightmap_to_costmap");

    HeightmapToCostMap hcm; //this loads up the node
    ros::Rate rate(10);
    while (ros::ok())
    {
        ros::spinOnce(); //where she stops nobody knows
        hcm.generate_costmap();
        rate.sleep();
    }
}
