#include "ros/ros.h"
#include <nav_msgs/Path.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_2d/cost_values.h>

#include <memory>
#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>

// =============================================================================
//  A* local planner for ArmPi Pro (mecanum / holonomic) — self-contained
//
//  System logic (confirmed):
//    (1) AprilTag (+ RTAB-Map) localization  -> publishes map->base_footprint TF
//    (2) local_costmap fed by RTAB-Map/sensors -> WALLS + OTHER ROBOTS
//    (3) this node: A* on the local costmap every cycle -> drive to goal
//
//  What is hardcoded here (fixed reference data from store_xy.yaml / tags.yaml):
//    - STORE goal coordinates  (used as the navigation goal)
//    - SIGNBOARD/AprilTag map poses (reference only; localization is a
//      separate node that produces the TF this planner consumes)
//
//  What is NOT hardcoded (on purpose):
//    - Walls and dynamic robots. Those come from the local costmap that
//      RTAB-Map SLAM populates. Hardcoding a second wall source would fight
//      the SLAM map and cause exactly the "mess up" we want to avoid.
//
//  Goal source priority:
//    1. rosparam /target_store (store name OR category) -> hardcoded table  [MAIN]
//    2. /graph_planner/path/global_path                  -> last pose       [FALLBACK]
// =============================================================================

// ---- Hardcoded map reference data ------------------------------------------
struct Store { const char* name; const char* category; double x; double y; };
static const std::vector<Store> STORES = {
    {"store_burger_A", "burger",            -1.25, -1.15},
    {"store_burger_B", "burger",             1.25, -0.85},
    {"store_cafe_A",   "cafe",              -0.75, -0.85},
    {"store_cafe_B",   "cafe",               1.00, -1.65},
    {"store_phar_A",   "pharmacy",           1.00,  1.50},
    {"store_phar_B",   "pharmacy",           0.75, -1.20},
    {"store_store_A",  "convenience_store", -1.00,  1.50},
    {"store_store_B",  "convenience_store", -1.00, -1.65},
};

// SIGNBOARD AprilTag poses [name, x(m), y(m), psi(deg)] from tags.yaml.
// Reference only: the AprilTag localization node uses these to correct the
// robot's map pose. The A* planner just consumes the resulting TF.
struct Tag { const char* name; double x; double y; double psi_deg; };
static const std::vector<Tag> TAGS = {
    {"SIGNBOARD01", -2.00, -2.135, 270.0}, {"SIGNBOARD04", -1.88, -2.250, 180.0},
    {"SIGNBOARD02", -2.00,  0.635,  90.0}, {"SIGNBOARD03", -2.00,  0.865, 270.0},
    {"SIGNBOARD05", -1.88,  0.750, 180.0}, {"SIGNBOARD06", -0.12, -2.250,   0.0},
    {"SIGNBOARD08",  0.00, -2.135, 270.0}, {"SIGNBOARD10",  0.12, -2.250, 180.0},
    {"SIGNBOARD07", -0.12,  0.750,   0.0}, {"SIGNBOARD09",  0.00,  0.635,  90.0},
    {"SIGNBOARD11",  0.12,  0.750, 180.0}, {"SIGNBOARD12",  1.88, -2.250,   0.0},
    {"SIGNBOARD14",  2.00, -2.135, 270.0}, {"SIGNBOARD13",  1.88,  0.750,   0.0},
    {"SIGNBOARD15",  2.00,  0.635,  90.0}, {"SIGNBOARD16",  2.00,  0.865, 270.0},
};

class AStarPlannerNode {
public:
    AStarPlannerNode(ros::NodeHandle& nh) : nh_(nh), tf_listener_(tf_buffer_) {
        ros::NodeHandle pnh("~");
        pnh.param("max_vel_x",        max_vel_x_,        0.20);
        pnh.param("max_vel_y",        max_vel_y_,        0.15);
        pnh.param("max_vel_theta",    max_vel_theta_,    0.60);
        pnh.param("lookahead_dist",   lookahead_dist_,   0.35);
        pnh.param("goal_tolerance",   goal_tolerance_,   0.12);
        pnh.param("inflation_weight", inflation_weight_, 3.0);
        pnh.param("lethal_cost",      lethal_cost_,      253);
        pnh.param("crash_dist",       crash_dist_,       0.32);
        pnh.param("carrot_max_dist",  carrot_max_dist_,  2.0);

        try {
            costmap_ros_.reset(new costmap_2d::Costmap2DROS("local_costmap", tf_buffer_));
            costmap_ros_->start();
            ROS_INFO("A* local planner: costmap initialized.");
        } catch (const std::exception& e) {
            ROS_WARN("Failed to initialize local costmap: %s", e.what());
        }

        subGoalPoint = nh_.subscribe("/graph_planner/path/global_path", 1,
                                     &AStarPlannerNode::CallbackGoalPoint, this);
        pubCommand   = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1, true);
        pubLocalPlan = nh_.advertise<nav_msgs::Path>("/astar_local_plan", 1);
    }

    void CallbackGoalPoint(const nav_msgs::Path& msg) {
        if (msg.poses.empty()) return;
        global_path_   = msg;
        path_received_ = true;
    }

    // -------------------------------------------------------------------------
    void Plan() {
        std::string planner_type;
        nh_.param<std::string>("/local_planner_type", planner_type, "astar");
        if (planner_type != "astar") return;

        // pause / state gate (same contract as the old TEB node)
        bool is_paused = false;
        nh_.param<bool>("/exploration_paused", is_paused, false);
        std::string state = "IDLE";
        nh_.param<std::string>("/exploration_state", state, "IDLE");
        if (is_paused || state == "IDLE" || state == "STOP" || state == "RECOVERY") {
            publishStop();
            return;
        }

        if (!costmap_ros_) return;

        // ---- resolve goal: hardcoded store [MAIN], else graph_planner [FALLBACK]
        double gx, gy;
        bool   from_path = false;
        if (!resolveGoal(gx, gy, from_path)) {
            ROS_WARN_THROTTLE(2.0,
                "No goal: set rosparam /target_store (name or category) "
                "or publish /graph_planner/path/global_path.");
            return;
        }

        // ---- robot pose via TF ---------------------------------------------
        double rx, ry, yaw;
        if (!getRobotPose(rx, ry, yaw)) return;

        // ---- final-goal arrival --------------------------------------------
        if (std::hypot(gx - rx, gy - ry) < goal_tolerance_) {
            publishStop();
            ROS_INFO_THROTTLE(1.0, "Goal reached (%.2f, %.2f).", gx, gy);
            return;
        }

        costmap_2d::Costmap2D* cm = costmap_ros_->getCostmap();
        double min_obs = getMinObstacleDistance(cm, rx, ry);

        // ---- 1) crash-avoidance override -----------------------------------
        if (min_obs < crash_dist_) {
            ROS_WARN_THROTTLE(0.5, "OBSTACLE TOO CLOSE (%.2fm). Backing up.", min_obs);
            geometry_msgs::Twist c; c.linear.x = -0.03;
            pubCommand.publish(c);
            consecutive_failures_ = 0; recovery_state_ = 0;
            return;
        }

        // ---- 2) active recovery (backup -> spin) ---------------------------
        if (recovery_state_ == 1) {
            double dt = (ros::Time::now() - recovery_start_time_).toSec();
            if (dt < 1.5) { geometry_msgs::Twist c; c.linear.x = -0.03; pubCommand.publish(c); return; }
            recovery_state_ = 2; recovery_start_time_ = ros::Time::now();
        }
        if (recovery_state_ == 2) {
            double dt = (ros::Time::now() - recovery_start_time_).toSec();
            if (dt < 2.0) { geometry_msgs::Twist c; c.angular.z = 0.15; pubCommand.publish(c); return; }
            recovery_state_ = 0; consecutive_failures_ = 0;
            ROS_INFO("Recovery complete. Retrying A* planner.");
        }

        // ---- 3) choose carrot inside the local costmap ---------------------
        geometry_msgs::Point carrot;
        bool ok = from_path ? selectCarrotOnPath(cm, rx, ry, carrot)
                            : selectCarrotToGoal(cm, rx, ry, gx, gy, carrot);
        if (!ok) {
            ROS_WARN_THROTTLE(1.0, "No valid carrot inside local costmap.");
            handlePlanFailure();
            return;
        }

        // ---- 4) A* on the local costmap ------------------------------------
        std::vector<geometry_msgs::Point> path;
        if (!planAStar(cm, rx, ry, carrot.x, carrot.y, path) || path.size() < 2) {
            ROS_WARN_THROTTLE(1.0, "A* failed (consec %d/10).", consecutive_failures_);
            handlePlanFailure();
            return;
        }
        consecutive_failures_ = 0; recovery_state_ = 0;

        // ---- 5) pure-pursuit -> holonomic cmd_vel --------------------------
        geometry_msgs::Twist cmd = followPath(path, rx, ry, yaw);
        double angular_factor = 0.4 + (std::min(min_obs, 1.2) / 1.2) * 0.6;
        cmd.angular.z *= angular_factor;
        pubCommand.publish(cmd);

        // ---- viz at ~5 Hz --------------------------------------------------
        if (++plan_publish_counter_ >= 4) {
            plan_publish_counter_ = 0;
            nav_msgs::Path p;
            p.header.stamp = ros::Time::now();
            p.header.frame_id = "map";
            for (auto& pt : path) {
                geometry_msgs::PoseStamped ps;
                ps.header = p.header;
                ps.pose.position = pt;
                ps.pose.orientation.w = 1.0;
                p.poses.push_back(ps);
            }
            pubLocalPlan.publish(p);
        }

        ROS_INFO_THROTTLE(1.0,
            "A* OK: goal(%.2f,%.2f)%s cmd[vx %.2f vy %.2f w %.2f] obs %.2fm len %zu",
            gx, gy, from_path ? "[path]" : "[store]",
            cmd.linear.x, cmd.linear.y, cmd.angular.z, min_obs, path.size());
    }

private:
    // ---- goal resolution -----------------------------------------------------
    bool resolveGoal(double& gx, double& gy, bool& from_path) {
        std::string target;
        nh_.param<std::string>("/target_store", target, "");
        if (!target.empty()) {
            // exact store name first, then category (first match)
            for (const auto& s : STORES)
                if (target == s.name) { gx = s.x; gy = s.y; from_path = false; return true; }
            for (const auto& s : STORES)
                if (target == s.category) { gx = s.x; gy = s.y; from_path = false; return true; }
            ROS_WARN_THROTTLE(2.0, "/target_store '%s' not in store table.", target.c_str());
            // fall through to graph_planner fallback
        }
        if (path_received_ && !global_path_.poses.empty()) {
            gx = global_path_.poses.back().pose.position.x;
            gy = global_path_.poses.back().pose.position.y;
            from_path = true;
            return true;
        }
        return false;
    }

    bool getRobotPose(double& rx, double& ry, double& yaw) {
        try {
            auto t = tf_buffer_.lookupTransform("map", "base_footprint",
                                                ros::Time(0), ros::Duration(0.2));
            rx  = t.transform.translation.x;
            ry  = t.transform.translation.y;
            yaw = tf2::getYaw(t.transform.rotation);
            return true;
        } catch (tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(1.0, "Could not get robot pose: %s", ex.what());
            return false;
        }
    }

    // ---- carrot from a single goal point (hardcoded-store mode) -------------
    // March along the straight line robot->goal, keeping the last free, in-bounds
    // point within carrot_max_dist. A* then routes to it around obstacles.
    bool selectCarrotToGoal(costmap_2d::Costmap2D* cm, double rx, double ry,
                            double gx, double gy, geometry_msgs::Point& carrot) {
        double d = std::hypot(gx - rx, gy - ry);
        unsigned int mx, my;
        if (d <= carrot_max_dist_ && cm->worldToMap(gx, gy, mx, my)
            && cm->getCost(mx, my) < lethal_cost_) {
            carrot.x = gx; carrot.y = gy; return true;       // goal directly visible
        }
        double step = std::max(0.01, cm->getResolution());
        bool found = false;
        double tmax = std::min(d, carrot_max_dist_);
        for (double t = step; t <= tmax; t += step) {
            double px = rx + (gx - rx) * t / d;
            double py = ry + (gy - ry) * t / d;
            if (!cm->worldToMap(px, py, mx, my)) break;       // left the costmap window
            if (cm->getCost(mx, my) >= lethal_cost_) break;   // blocked ahead on the ray
            carrot.x = px; carrot.y = py; found = true;
        }
        return found;
    }

    // ---- carrot along the global path (graph_planner fallback mode) ---------
    bool selectCarrotOnPath(costmap_2d::Costmap2D* cm, double rx, double ry,
                            geometry_msgs::Point& carrot) {
        double best = 1e9; int closest = last_closest_idx_;
        int end = (int)global_path_.poses.size();
        for (int i = std::max(0, last_closest_idx_);
             i < std::min(end, last_closest_idx_ + 100); ++i) {
            auto& p = global_path_.poses[i].pose.position;
            double dd = std::hypot(p.x - rx, p.y - ry);
            if (dd < best) { best = dd; closest = i; }
        }
        last_closest_idx_ = closest;

        bool found = false;
        for (int i = closest; i < end; ++i) {
            auto& p = global_path_.poses[i].pose.position;
            if (std::hypot(p.x - rx, p.y - ry) > carrot_max_dist_) break;
            unsigned int mx, my;
            if (!cm->worldToMap(p.x, p.y, mx, my)) { if (found) break; else continue; }
            if (cm->getCost(mx, my) >= lethal_cost_) { if (found) break; else continue; }
            carrot.x = p.x; carrot.y = p.y; found = true;
        }
        if (!found) {
            auto& g = global_path_.poses.back().pose.position;
            unsigned int mx, my;
            if (cm->worldToMap(g.x, g.y, mx, my) && cm->getCost(mx, my) < lethal_cost_) {
                carrot = g; found = true;
            }
        }
        return found;
    }

    // ---- A* over the costmap grid (8-connected, inflation-aware) ------------
    struct Node { int idx; double f; };
    struct Cmp { bool operator()(const Node& a, const Node& b) const { return a.f > b.f; } };

    bool planAStar(costmap_2d::Costmap2D* cm, double sx, double sy,
                   double gx, double gy, std::vector<geometry_msgs::Point>& out) {
        unsigned int smx, smy, gmx, gmy;
        if (!cm->worldToMap(sx, sy, smx, smy)) return false;
        if (!cm->worldToMap(gx, gy, gmx, gmy)) return false;

        const int W = cm->getSizeInCellsX();
        const int H = cm->getSizeInCellsY();
        auto id = [&](int x, int y) { return y * W + x; };
        const int start = id(smx, smy);
        const int goal  = id(gmx, gmy);

        std::priority_queue<Node, std::vector<Node>, Cmp> open;
        std::unordered_map<int, double> g_cost;
        std::unordered_map<int, int> came_from;
        auto heur = [&](int x, int y) {
            double dx = x - (int)gmx, dy = y - (int)gmy;
            return std::sqrt(dx * dx + dy * dy);
        };
        g_cost[start] = 0.0;
        open.push({start, heur(smx, smy)});

        const int dx8[8] = {1, -1, 0, 0, 1, 1, -1, -1};
        const int dy8[8] = {0, 0, 1, -1, 1, -1, 1, -1};

        bool reached = false;
        int iters = 0, max_iters = W * H + 10;
        while (!open.empty() && iters++ < max_iters) {
            Node cur = open.top(); open.pop();
            if (cur.idx == goal) { reached = true; break; }
            int cx = cur.idx % W, cy = cur.idx / W;
            double cg = g_cost[cur.idx];
            for (int k = 0; k < 8; ++k) {
                int nx = cx + dx8[k], ny = cy + dy8[k];
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                unsigned char c = cm->getCost(nx, ny);
                if (c >= lethal_cost_) continue;                       // hard block
                int nidx = id(nx, ny);
                double step = (dx8[k] != 0 && dy8[k] != 0) ? 1.41421356 : 1.0;
                double soft = 1.0 + inflation_weight_ * (double)c / 252.0;
                double ng = cg + step * soft;
                auto it = g_cost.find(nidx);
                if (it == g_cost.end() || ng < it->second) {
                    g_cost[nidx] = ng;
                    came_from[nidx] = cur.idx;
                    open.push({nidx, ng + heur(nx, ny)});
                }
            }
        }
        if (!reached) return false;

        std::vector<int> rev; int n = goal;
        while (n != start) { rev.push_back(n); n = came_from[n]; }
        rev.push_back(start);
        out.clear();
        for (auto it = rev.rbegin(); it != rev.rend(); ++it) {
            double wx, wy; cm->mapToWorld(*it % W, *it / W, wx, wy);
            geometry_msgs::Point p; p.x = wx; p.y = wy; out.push_back(p);
        }
        return true;
    }

    // ---- pure-pursuit follower (holonomic) ----------------------------------
    geometry_msgs::Twist followPath(const std::vector<geometry_msgs::Point>& path,
                                    double rx, double ry, double yaw) {
        geometry_msgs::Point target = path.back();
        for (const auto& p : path)
            if (std::hypot(p.x - rx, p.y - ry) >= lookahead_dist_) { target = p; break; }

        double dx = target.x - rx, dy = target.y - ry;
        double desired_yaw = std::atan2(dy, dx);
        double yaw_err = std::atan2(std::sin(desired_yaw - yaw),
                                    std::cos(desired_yaw - yaw));

        double vx_r =  std::cos(yaw) * dx + std::sin(yaw) * dy;
        double vy_r = -std::sin(yaw) * dx + std::cos(yaw) * dy;
        double norm = std::hypot(vx_r, vy_r);
        if (norm > 1e-6) { vx_r /= norm; vy_r /= norm; }

        geometry_msgs::Twist cmd;
        cmd.linear.x  = vx_r * max_vel_x_;
        cmd.linear.y  = vy_r * max_vel_y_;
        cmd.angular.z = std::max(-max_vel_theta_, std::min(max_vel_theta_, 1.5 * yaw_err));
        return cmd;
    }

    void handlePlanFailure() {
        consecutive_failures_++;
        if (consecutive_failures_ >= 10) {
            recovery_state_ = 1; recovery_start_time_ = ros::Time::now();
            ROS_WARN("Robot STUCK. Initiating backup & spin recovery.");
        } else {
            publishStop();
        }
    }

    void publishStop() { geometry_msgs::Twist c; pubCommand.publish(c); }

    double getMinObstacleDistance(costmap_2d::Costmap2D* cm, double rx, double ry) {
        unsigned int mx, my;
        if (!cm->worldToMap(rx, ry, mx, my)) return 1.5;
        double res = cm->getResolution();
        int r = (int)(0.5 / res);
        double min_dist = 1.5;
        for (int ddx = -r; ddx <= r; ++ddx)
            for (int ddy = -r; ddy <= r; ++ddy) {
                int cx = (int)mx + ddx, cy = (int)my + ddy;
                if (cx < 0 || cy < 0 || cx >= (int)cm->getSizeInCellsX()
                    || cy >= (int)cm->getSizeInCellsY()) continue;
                if (cm->getCost(cx, cy) > 50) {
                    double wx, wy; cm->mapToWorld(cx, cy, wx, wy);
                    double d = std::hypot(wx - rx, wy - ry);
                    if (d < min_dist) min_dist = d;
                }
            }
        return min_dist;
    }

    // ---- members ------------------------------------------------------------
    ros::NodeHandle nh_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::shared_ptr<costmap_2d::Costmap2DROS> costmap_ros_;

    ros::Subscriber subGoalPoint;
    ros::Publisher  pubCommand;
    ros::Publisher  pubLocalPlan;

    nav_msgs::Path global_path_;
    bool path_received_ = false;
    int  last_closest_idx_ = 0;
    int  plan_publish_counter_ = 0;

    int consecutive_failures_ = 0;
    int recovery_state_ = 0;        // 0 none, 1 backup, 2 spin
    ros::Time recovery_start_time_;

    double max_vel_x_, max_vel_y_, max_vel_theta_;
    double lookahead_dist_, goal_tolerance_, inflation_weight_;
    double crash_dist_, carrot_max_dist_;
    int    lethal_cost_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "astar_planner_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    double freq = 20.0;
    pnh.param<double>("planner_frequency", freq, 20.0);
    ROS_INFO("A* local planner loop rate: %.1f Hz", freq);

    AStarPlannerNode node(nh);
    ros::Rate rate(freq);
    while (ros::ok()) {
        ros::spinOnce();
        node.Plan();
        rate.sleep();
    }
    return 0;
}