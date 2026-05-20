#!/usr/bin/env python3

import rospy
import math
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import tf.transformations as tf_trans

def normalize_pi_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

class CustomVectorPlanner:
    def __init__(self):
        rospy.init_node('custom_vector_planner_node')

        # APF parameters
        self.K_ATTRACTIVE = 1.3
        self.K_REPULSIVE = 0.4
        self.REPULSION_DIST = 1.2  # meters
        self.ARRIVAL_THRES = 0.4   # meters
        
        # Max velocity limits
        self.MAX_VEL_X = 0.8
        self.MAX_VEL_Y = 0.4
        self.MAX_VEL_W = 2.0

        # State variables
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_yaw = 0.0
        
        self.global_path = []
        self.local_map = None
        self.last_closest_idx = 0
        self.was_active = False

        # Subscribers
        rospy.Subscriber("/map/local_map/obstacle", OccupancyGrid, self.cb_occupancy_grid)
        rospy.Subscriber("/map_odom", Odometry, self.cb_ego_odom)
        rospy.Subscriber("/graph_planner/path/global_path", Path, self.cb_global_path)

        # Publishers
        self.pubCommand = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.pubSelectedMotion = rospy.Publisher("/points/selected_motion", PointCloud2, queue_size=1)

        rospy.loginfo("Custom Mecanum Vector Field Planner Node Initialized.")

    def cb_ego_odom(self, msg):
        self.ego_x = msg.pose.pose.position.x
        self.ego_y = msg.pose.pose.position.y
        q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        _, _, self.ego_yaw = tf_trans.euler_from_quaternion(q)

    def cb_global_path(self, msg):
        self.global_path = msg.poses
        self.last_closest_idx = 0

    def cb_occupancy_grid(self, msg):
        self.local_map = msg

    def publish_vector_visualization(self, dx, dy):
        # Generate a line along computed force vector in base_footprint local frame
        points = []
        length = math.hypot(dx, dy)
        if length > 0.01:
            ux = dx / length
            uy = dy / length
            for i in range(12):
                d = i * 0.12
                points.append([d * ux, d * uy, 0.0])
        
        header = Header()
        header.frame_id = "base_footprint"
        header.stamp = rospy.Time.now()
        
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        
        pc_msg = pc2.create_cloud(header, fields, points)
        self.pubSelectedMotion.publish(pc_msg)

    def plan(self):
        # 1. Gate parameter check
        planner_type = rospy.get_param("/local_planner_type", "control_space")
        if planner_type != "custom_vector":
            if self.was_active:
                # Stop the robot on transition
                cmd = Twist()
                self.pubCommand.publish(cmd)
                self.was_active = False
            return
        
        # Check emergency stop condition
        is_paused = rospy.get_param("/exploration_paused", False)
        state = rospy.get_param("/exploration_state", "IDLE")
        if is_paused or state in ["IDLE", "STOP", "RECOVERY"]:
            if self.global_path:
                cmd = Twist()
                self.pubCommand.publish(cmd)
                self.global_path = []
            return

        self.was_active = True

        # 2. Check prerequisites
        if not self.global_path or self.local_map is None:
            return

        # 3. Find closest path waypoint
        closest_idx = self.last_closest_idx
        min_dist = float('inf')
        search_start = max(0, self.last_closest_idx - 10)
        search_end = min(len(self.global_path), self.last_closest_idx + 100)
        
        for i in range(search_start, search_end):
            p = self.global_path[i].pose.position
            d = math.hypot(p.x - self.ego_x, p.y - self.ego_y)
            if d < min_dist:
                min_dist = d
                closest_idx = i
        self.last_closest_idx = closest_idx

        # 4. Find lookahead target
        lookahead_dist = 1.3
        target_idx = len(self.global_path) - 1
        for i in range(closest_idx, len(self.global_path)):
            p = self.global_path[i].pose.position
            d = math.hypot(p.x - self.ego_x, p.y - self.ego_y)
            if d >= lookahead_dist:
                target_idx = i
                break
        
        target_pose = self.global_path[target_idx]
        target_x = target_pose.pose.position.x
        target_y = target_pose.pose.position.y

        # 5. Attractive Force (local frame)
        del_x = target_x - self.ego_x
        del_y = target_y - self.ego_y
        
        # Transform goal position to robot local frame
        local_target_x = math.cos(-self.ego_yaw) * del_x - math.sin(-self.ego_yaw) * del_y
        local_target_y = math.sin(-self.ego_yaw) * del_x + math.cos(-self.ego_yaw) * del_y
        
        dist_to_target = math.hypot(local_target_x, local_target_y)
        if dist_to_target > 0.01:
            F_att_x = (local_target_x / dist_to_target) * self.K_ATTRACTIVE
            F_att_y = (local_target_y / dist_to_target) * self.K_ATTRACTIVE
        else:
            F_att_x = 0.0
            F_att_y = 0.0

        # 6. Repulsive Force (local frame - occupancy grid is already in base_footprint frame)
        F_rep_x = 0.0
        F_rep_y = 0.0
        
        w = self.local_map.info.width
        h = self.local_map.info.height
        resol = self.local_map.info.resolution
        origin_x = self.local_map.info.origin.position.x
        origin_y = self.local_map.info.origin.position.y
        
        # Robot center in cell coordinates
        center_col = int(-origin_x / resol)
        center_row = int(-origin_y / resol)
        
        search_radius_cells = int(self.REPULSION_DIST / resol) + 2
        col_start = max(0, center_col - search_radius_cells)
        col_end = min(w, center_col + search_radius_cells + 1)
        row_start = max(0, center_row - search_radius_cells)
        row_end = min(h, center_row + search_radius_cells + 1)
        
        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                cost = self.local_map.data[r * w + c]
                if cost > 0: # Check all non-zero cells for smooth potential field boundary
                    # Cell position in base_footprint frame
                    cx = origin_x + c * resol + 0.5 * resol
                    cy = origin_y + r * resol + 0.5 * resol
                    
                    d = math.hypot(cx, cy)
                    if d < 0.05:
                        d = 0.05 # Avoid divide by zero
                        
                    if d < self.REPULSION_DIST:
                        # Scale repulsion force by the cell occupancy value (cost) to match inflation weight
                        cost_scale = float(cost) / 100.0
                        force_mag = self.K_REPULSIVE * cost_scale * (1.0 / d - 1.0 / self.REPULSION_DIST) * (1.0 / (d * d))
                        
                        # Vector points away from obstacle cell: robot (0,0) - obstacle (cx, cy)
                        rx = -cx / d
                        ry = -cy / d
                        
                        F_rep_x += rx * force_mag
                        F_rep_y += ry * force_mag

        # 7. Compute combined total force vector
        F_tot_x = F_att_x + F_rep_x
        F_tot_y = F_att_y + F_rep_y
        
        # Desired heading relative to base_footprint
        desired_local_yaw = math.atan2(F_tot_y, F_tot_x)
        
        # 8. Command translation velocities (camera field of view alignment check)
        cmd = Twist()
        fov_half = 42.6 * (math.pi / 180.0) # Camera half field of view
        
        if abs(desired_local_yaw) <= fov_half:
            # Desired movement vector is within camera FOV: go forward and slide as needed
            cmd.linear.x = max(0.0, F_tot_x)
            cmd.linear.y = F_tot_y
            cmd.angular.z = 2.0 * desired_local_yaw
        else:
            # Desired movement direction is outside camera vision: turn in place to align first
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = 2.0 * desired_local_yaw
        
        # Clamp controls
        cmd.linear.x = max(0.0, min(self.MAX_VEL_X, cmd.linear.x))
        cmd.linear.y = max(-self.MAX_VEL_Y, min(self.MAX_VEL_Y, cmd.linear.y))
        cmd.angular.z = max(-self.MAX_VEL_W, min(self.MAX_VEL_W, cmd.angular.z))
        
        # 9. Arrival rule scaling and stop condition
        final_goal = self.global_path[-1].pose.position
        dist_to_final = math.hypot(final_goal.x - self.ego_x, final_goal.y - self.ego_y)
        
        if dist_to_final < self.ARRIVAL_THRES:
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = 0.0
            rospy.loginfo_throttle(5.0, "Vector Planner: Goal Reached!")
        elif dist_to_final < 1.2:
            scale = (dist_to_final - self.ARRIVAL_THRES) / (1.2 - self.ARRIVAL_THRES)
            cmd.linear.x *= scale
            cmd.linear.y *= scale
            cmd.angular.z *= scale
            
        self.pubCommand.publish(cmd)
        self.publish_vector_visualization(F_tot_x, F_tot_y)

    def run(self):
        rate = rospy.Rate(50.0)
        while not rospy.is_shutdown():
            self.plan()
            rate.sleep()

if __name__ == '__main__':
    try:
        vp = CustomVectorPlanner()
        vp.run()
    except rospy.ROSInterruptException:
        pass
