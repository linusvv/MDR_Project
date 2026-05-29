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

#define MAP_IDX(sx, i, j) ((sx) * (j) + (i))

class HeightmapToCostMap
{
public:
    HeightmapToCostMap();
    void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &cloud_msg);
    void generate_costmap();

    bool DO_INFLATION = true; // true
    float RESOLUTION_ = 0.1; // [m / cell]
    float MAP_MIN_X =  -5; // map min x position
    float MAP_MAX_X =  15; // map max x position
    float MAP_MIN_Y = -10; // map min y position
    float MAP_MAX_Y =  10; // map max y position

    float MIN_OBSTACLE_HEIGHT = 0.04; // [m] ignore ground points
    float MAX_OBSTACLE_HEIGHT = 0.30; // [m] Limit to 30cm to allow passing under floating signboards

    float INFLATION_RADIUS = 0.6; // [m] Increased to 1.5x (0.4m -> 0.6m) for a larger dangerous zone
    float INFLATION_RES    = RESOLUTION_; // [m] resolution of inflation
    int INFLATION_BINS     = (INFLATION_RADIUS * 2) / INFLATION_RES + 1;

private:
    ros::NodeHandle nh_;
    std::string cloud_topic_; //default input
    std::string map_topic_;
    ros::Subscriber sub_;
    ros::Publisher cost_map_pub_;

    // Varialbes
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz;

    bool bGetPoint = false;
    std::vector<int8_t> grid_data;
};

HeightmapToCostMap::HeightmapToCostMap() : cloud_topic_("/points/velodyne_obstacles"), map_topic_("/map/local_map/obstacle")
{
    sub_ = nh_.subscribe(cloud_topic_, 1, &HeightmapToCostMap::cloud_cb, this);

    cost_map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(map_topic_, 10);

    //print some info about the node
    ROS_INFO("[HeightmapToCostMap] Loaded!");
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
                    int x = int((it->x - MAP_MIN_X) / RESOLUTION_);
                    int y = int((it->y - MAP_MIN_Y) / RESOLUTION_);
                    
                    if (x < width_ && x >= 0 && y < height_ && y >= 0)
                    {
                        oMap.data[MAP_IDX(width_, x, y)] = 100;
                        point_count++;
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
            ROS_INFO_THROTTLE(5, "No point cloud yet!!!");
        }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "heightmap_to_costmap");

    HeightmapToCostMap hcm; //this loads up the node
    ros::Rate rate(50);
    while (ros::ok())
    {
        ros::spinOnce(); //where she stops nobody knows
        hcm.generate_costmap();
        rate.sleep();
    }
}
