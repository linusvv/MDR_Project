import rospy
import math
import numpy as np
import tf
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import tf.transformations as tf_trans

def normalize_pi_to_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

class Node:
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0, delta=0.0, 
                 cost_control=0.0, cost_colli=0.0, cost_total=0.0, idx=-1, collision=False, minDistGoal=float('inf'), v=0.0, w=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.delta = delta
        self.minDistGoal = minDistGoal
        self.cost_control = cost_control
        self.cost_colli = cost_colli
        self.cost_total = cost_total
        self.idx = idx
        self.collision = collision
        self.v = v
        self.w = w

class MotionPlanner:
    def __init__(self):
        rospy.init_node('control_space_planner_python')
        self.tf_listener = tf.TransformListener()

        # DWA local planner mapped weights
        self.W_COST_DIRECTION = 4.5  # Increased to prioritize heading to goal
        self.W_COST_TRAVERSABILITY = 3.5
        self.W_COST_STEERING = 1.0

        self.mapMinX = -5.0
        self.mapMaxX = 15.0
        self.mapMinY = -10.0
        self.mapMaxY = 10.0
        self.mapResol = 0.1
        self.OCCUPANCY_THRES = 40

        self.origin_x = 0.0
        self.origin_y = 0.0
        self.frame_id = "base_link"

        self.FOV = 85.2 * (math.pi / 180.0)
        self.MAX_SENSOR_RANGE = 10.0
        self.WHEELBASE = 0.1
        self.DIST_RESOL = 0.1
        self.TIME_RESOL = 0.2
        self.ARRIVAL_THRES = 0.4
        self.INFLATION_SIZE = int(0.20 / self.mapResol)

        self.bGetMap = False
        self.bGetGoal = False
        self.bGetLocalNode = False
        self.bGetEgoOdom = False
        
        self.localMap = None
        self.egoOdom = None
        self.goalPose = None
        self.global_path = []
        self.LOOKAHEAD_DIST = 1.0  # Slightly shorter for tighter following
        self.prev_w = 0.0
        self.recovery_time = 0
        self.recovery_w = 1.5
        self.search_multiplier = 1.0
        
        # Pose smoothing buffers
        self.odom_buffer_size = 5
        self.odom_x_buffer = []
        self.odom_y_buffer = []
        self.odom_sin_buffer = []
        self.odom_cos_buffer = []

        # ... (rest of init)
        self.obstacle_memory = {}
        self.obstacle_memory_duration = 2.0
        self.prev_v = 0.0
        self.v_smooth_alpha = 0.6
        self.last_log_time = 0
        
        # Sub/Pub
        rospy.Subscriber("/map/local_map/obstacle", OccupancyGrid, self.cb_occupancy_grid)
        rospy.Subscriber("/graph_planner/path/global_path", Path, self.cb_global_path)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.cb_goal_point)

        self.pubCommand = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.pubTruncTarget = rospy.Publisher("/car/trunc_target", PoseStamped, queue_size=1)
        self.pubSelectedMotion = rospy.Publisher("/points/selected_motion", PointCloud2, queue_size=1)
        self.pubMotionPrimitives = rospy.Publisher("/points/motion_primitives", PointCloud2, queue_size=1)

    def update_ego_pose(self):
        """Uses TF to get robot pose in the map frame instead of raw odom."""
        try:
            # We want the robot pose in the frame that the path is tracking (usually 'map')
            path_frame = "map"
            if self.global_path:
                path_frame = self.global_path[0].header.frame_id
            
            (trans, rot) = self.tf_listener.lookupTransform(path_frame, 'base_footprint', rospy.Time(0))
            raw_x, raw_y = trans[0], trans[1]
            euler = tf_trans.euler_from_quaternion(rot)
            raw_yaw = euler[2]

            self.odom_x_buffer.append(raw_x)
            self.odom_y_buffer.append(raw_y)
            self.odom_sin_buffer.append(math.sin(raw_yaw))
            self.odom_cos_buffer.append(math.cos(raw_yaw))
            
            if len(self.odom_x_buffer) > self.odom_buffer_size:
                self.odom_x_buffer.pop(0)
                self.odom_y_buffer.pop(0)
                self.odom_sin_buffer.pop(0)
                self.odom_cos_buffer.pop(0)
            
            self.ego_x = np.mean(self.odom_x_buffer)
            self.ego_y = np.mean(self.odom_y_buffer)
            # Correct angular averaging using sin/cos components
            self.ego_yaw = math.atan2(np.mean(self.odom_sin_buffer), np.mean(self.odom_cos_buffer))
            
            self.bGetEgoOdom = True
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

    def cb_occupancy_grid(self, msg):
        self.localMap = msg
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.frame_id = msg.header.frame_id
        self.mapResol = msg.info.resolution
        self.bGetMap = True
        # Update obstacle memory with newly observed occupied cells
        try:
            now = rospy.Time.now().to_sec()
        except Exception:
            now = rospy.get_time()

        w = msg.info.width
        h = msg.info.height
        # store indices that are above threshold as seen now
        for idx, val in enumerate(msg.data):
            if val > self.OCCUPANCY_THRES:
                # record last seen timestamp for this grid cell index
                self.obstacle_memory[idx] = now

        # purge old memory entries
        expiry = self.obstacle_memory_duration
        stale = [k for k, t in self.obstacle_memory.items() if now - t > expiry]
        for k in stale:
            del self.obstacle_memory[k]

    def cb_goal_point(self, msg):
        self.goalPose = msg
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        _, _, yaw = tf_trans.euler_from_quaternion(q)
        self.goal_yaw = yaw
        self.bGetGoal = True

    def interpolate_path(self, path, step=0.1):
        dense_path = []
        for i in range(len(path) - 1):
            p1 = path[i].pose.position
            p2 = path[i+1].pose.position
            dist = math.hypot(p2.x - p1.x, p2.y - p1.y)
            num_steps = max(1, int(dist / step))
            for j in range(num_steps):
                new_pose = PoseStamped()
                new_pose.header = path[i].header
                new_pose.pose.position.x = p1.x + (p2.x - p1.x) * j / num_steps
                new_pose.pose.position.y = p1.y + (p2.y - p1.y) * j / num_steps
                new_pose.pose.position.z = p1.z
                new_pose.pose.orientation = path[i].pose.orientation
                dense_path.append(new_pose)
        if len(path) > 0:
            dense_path.append(path[-1])
        return dense_path

    def cb_global_path(self, msg):
        self.global_path = self.interpolate_path(msg.poses)
        if len(self.global_path) > 0:
            target = self.global_path[-1]
            self.goalPose = target
            self.goal_x = target.pose.position.x
            self.goal_y = target.pose.position.y
            q = [target.pose.orientation.x, target.pose.orientation.y, target.pose.orientation.z, target.pose.orientation.w]
            _, _, yaw = tf_trans.euler_from_quaternion(q)
            self.goal_yaw = yaw
            self.bGetGoal = True
        else:
            self.bGetGoal = False
            self.global_path = []

    def global_to_local_node(self, globalNode):
        delX = globalNode.x - self.ego_x
        delY = globalNode.y - self.ego_y

        newX = math.cos(-self.ego_yaw) * delX - math.sin(-self.ego_yaw) * delY
        newY = math.sin(-self.ego_yaw) * delX + math.cos(-self.ego_yaw) * delY
        newYaw = normalize_pi_to_pi(globalNode.yaw - self.ego_yaw)

        return Node(newX, newY, globalNode.z, newYaw)

    def local_to_planner_coordinate(self, localNode):
        mapX = (localNode.x - self.origin_x) / self.mapResol
        mapY = (localNode.y - self.origin_y) / self.mapResol
        return Node(mapX, mapY, localNode.z, localNode.yaw)

    def check_collision_and_dist(self, map_pt):
        w = self.localMap.info.width
        h = self.localMap.info.height
        r = self.INFLATION_SIZE
        
        is_col = False
        min_dist = float('inf')
        
        # Reduced search radius for performance
        search_radius = r + 2 
        try:
            now = rospy.Time.now().to_sec()
        except Exception:
            now = rospy.get_time()

        # Optimization: first check if the point itself is occupied (including memory)
        tx_center = int(map_pt.x)
        ty_center = int(map_pt.y)
        
        # Bounds check
        if tx_center < 0 or tx_center >= w or ty_center < 0 or ty_center >= h:
            return True, 0.0 # Out of bounds is collision

        for i in range(-search_radius, search_radius + 1):
            for j in range(-search_radius, search_radius + 1):
                tx = tx_center + i
                ty = ty_center + j
                if 0 <= tx < w and 0 <= ty < h:
                    idx = ty * w + tx
                    val = self.localMap.data[idx]
                    occupied = False
                    if val > self.OCCUPANCY_THRES:
                        occupied = True
                    else:
                        # consult memory
                        last_seen = self.obstacle_memory.get(idx, None)
                        if last_seen is not None and (now - last_seen) <= self.obstacle_memory_duration:
                            occupied = True

                    if occupied:
                        dist = math.hypot(i, j) * self.mapResol
                        if dist < min_dist:
                            min_dist = dist
                        if dist <= self.INFLATION_SIZE * self.mapResol:
                            is_col = True
                            return True, min_dist
        return is_col, min_dist

    def rollout_motion(self, v, w):
        motion = []
        curr = Node(x=0.0, y=0.0, yaw=0.0, v=v, w=w)
        
        sim_time = 1.0  # Reduced to reduce computation
        dt = self.TIME_RESOL
        t = 0.0
        
        minDistGoal = float('inf')
        minDistObstacle = float('inf')
        
        while t < sim_time:
            curr.x += v * math.cos(curr.yaw) * dt
            curr.y += v * math.sin(curr.yaw) * dt
            curr.yaw += w * dt
            curr.yaw = normalize_pi_to_pi(curr.yaw)

            if self.bGetLocalNode:
                distGoal = math.hypot(curr.x - self.truncLocalNode.x, curr.y - self.truncLocalNode.y)
                minDistGoal = min(minDistGoal, distGoal)
            curr.minDistGoal = minDistGoal

            map_pt = self.local_to_planner_coordinate(curr)
            
            is_col, dist_obs = self.check_collision_and_dist(map_pt)
            minDistObstacle = min(minDistObstacle, dist_obs)
            
            if is_col:
                curr.collision = True
                motion.append(Node(curr.x, curr.y, curr.z, curr.yaw, v=v, w=w, collision=True, minDistGoal=curr.minDistGoal))
                motion[-1].minDistObstacle = minDistObstacle
                return motion

            motion.append(Node(curr.x, curr.y, curr.z, curr.yaw, v=v, w=w, minDistGoal=curr.minDistGoal))
            motion[-1].minDistObstacle = minDistObstacle
            t += dt
            
        return motion

    def generate_motion_primitives(self):
        primitives = []
        # Fetch dynamic limits from ROS parameter server
        max_vel = rospy.get_param("/robot/max_vel", 0.06)
        max_omega = rospy.get_param("/robot/max_vel_theta", 0.3)
        velocities = np.linspace(0.0, max_vel, 6) 
        omegas = np.linspace(-max_omega, max_omega, 11)
        
        self.total_primitives = 0
        for v in velocities:
            for w in omegas:
                if v == 0.0 and w == 0.0:
                    continue
                self.total_primitives += 1
                motion = self.rollout_motion(v, w)
                if motion:
                    primitives.append(motion)
        return primitives
        return primitives

    def select_motion(self, primitives):
        minCost = float('inf')
        bestMotion = []
        
        self.collision_free_primitives = 0
        for pm in primitives:
            if not pm or pm[-1].collision:
                continue
            
            self.collision_free_primitives += 1
            endNode = pm[-1]
            distGoal = endNode.minDistGoal
            
            # Goal direction: aim at target idx on global path
            goal_yaw = math.atan2(self.truncLocalNode.y - endNode.y, self.truncLocalNode.x - endNode.x)
            yaw_diff = abs(normalize_pi_to_pi(goal_yaw - endNode.yaw))
            
            # Consistency
            consistency_cost = abs(endNode.w - self.prev_w)

            # Clearance
            clearance_cost = 0.0
            if hasattr(endNode, 'minDistObstacle') and endNode.minDistObstacle < float('inf'):
                clearance_cost = 1.0 / (endNode.minDistObstacle + 0.01)

            # Balanced DWA cost
            # Stronger heading pull (5.0) and moderate velocity preference (2.5)
            # Added a penalty for being stationary to prevent wiggling in place
            v_cost = -3.0 * endNode.v
            if endNode.v < 0.05:
                v_cost += 1.0 # Penalty for near-zero velocity

            cost = (5.0 * yaw_diff + 
                    2.0 * distGoal + 
                    0.2 * consistency_cost +
                    3.5 * clearance_cost +
                    v_cost)
            
            endNode.cost_total = cost
            
            if cost < minCost:
                minCost = cost
                bestMotion = pm
                
        return bestMotion

    def publish_command(self, motion):
        cmd = Twist()
        
        now = rospy.get_time()
        should_log = (now - self.last_log_time > 1.0)
        
        if not motion:
            cmd.angular.z = 0.0
            cmd.linear.x = 0.0
            if should_log:
                rospy.logwarn(f"[ControlSpace] NO FEASIBLE PATH! (Checked {self.total_primitives} samples, 0 collision-free)")
                self.last_log_time = now
        elif motion[-1].v == 0.0:
            cmd.angular.z = 0.0
            cmd.linear.x = 0.0
            if should_log:
                rospy.loginfo(f"[ControlSpace] Chosen motion has v=0 (Progress blocked). Free: {self.collision_free_primitives}/{self.total_primitives}")
                self.last_log_time = now
        else:
            self.recovery_time = 0
            self.search_multiplier = 1.0
            
            bestNode = motion[-1]
            # Smooth velocity
            cmd.linear.x = (self.v_smooth_alpha * bestNode.v) + ((1.0 - self.v_smooth_alpha) * self.prev_v)
            cmd.angular.z = bestNode.w

            if self.bGetGoal and self.global_path:
                final_node = self.global_path[-1].pose.position
                dist = math.hypot(final_node.x - self.ego_x, final_node.y - self.ego_y)
                if dist < self.ARRIVAL_THRES:
                    cmd.angular.z = 0.0
                    cmd.linear.x = 0.0
                    if should_log:
                        rospy.loginfo("[ControlSpace] Arrived at goal.")
                        self.last_log_time = now

            if should_log:
                rospy.loginfo(f"[ControlSpace] Moving | v: {cmd.linear.x:.2f} | w: {cmd.angular.z:.2f} | Cost: {bestNode.cost_total:.2f} | Free: {self.collision_free_primitives}/{self.total_primitives}")
                self.last_log_time = now

        self.prev_w = cmd.angular.z
        self.prev_v = cmd.linear.x
        self.pubCommand.publish(cmd)

    def plan(self):
        # Check if we should be active
        is_paused = rospy.get_param("/exploration_paused", False)
        explore_state = rospy.get_param("/exploration_state", "IDLE")
        
        if is_paused or explore_state == "IDLE":
            # If we were moving, stop.
            if self.prev_v != 0.0 or self.prev_w != 0.0:
                self.pubCommand.publish(Twist())
                self.prev_v = 0.0
                self.prev_w = 0.0
            return

        self.update_ego_pose()
        if not (self.bGetMap and self.bGetGoal and self.bGetEgoOdom):
            return
            
        if self.global_path:
            closest_dist = float('inf')
            closest_idx = 0
            for i, p in enumerate(self.global_path):
                d = math.hypot(p.pose.position.x - self.ego_x, p.pose.position.y - self.ego_y)
                if d < closest_dist:
                    closest_dist = d
                    closest_idx = i

            target_idx = closest_idx
            accumulated_dist = 0.0
            for i in range(closest_idx, len(self.global_path) - 1):
                p1 = self.global_path[i].pose.position
                p2 = self.global_path[i+1].pose.position
                accumulated_dist += math.hypot(p2.x - p1.x, p2.y - p1.y)
                if accumulated_dist > self.LOOKAHEAD_DIST:
                    target_idx = i
                    break
            else:
                target_idx = len(self.global_path) - 1

            target = self.global_path[target_idx]
            self.goal_x = target.pose.position.x
            self.goal_y = target.pose.position.y
            q = [target.pose.orientation.x, target.pose.orientation.y, target.pose.orientation.z, target.pose.orientation.w]
            _, _, self.goal_yaw = tf_trans.euler_from_quaternion(q)
            
        gNode = Node(self.goal_x, self.goal_y, 0, self.goal_yaw)
        self.localNode = self.global_to_local_node(gNode)
        
        self.truncLocalNode = Node(
            max(self.mapMinX, min(self.localNode.x, self.mapMaxX)),
            max(self.mapMinY, min(self.localNode.y, self.mapMaxY)),
            self.localNode.z,
            self.localNode.yaw
        )
        self.bGetLocalNode = True

        primitives = self.generate_motion_primitives()
        best = self.select_motion(primitives)
        
        # Publish selected motion for visualization
        if best:
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "base_link"
            pts = [[p.x, p.y, 0.0] for p in best]
            try:
                msg = pc2.create_cloud_xyz32(header, pts)
                self.pubSelectedMotion.publish(msg)
            except:
                pass

        self.publish_command(best)

    def run(self):
        rate = rospy.Rate(15.0) # Reduced from 50Hz to 15Hz for Python performance
        while not rospy.is_shutdown():
            self.plan()
            rate.sleep()

if __name__ == '__main__':
    try:
        mp = MotionPlanner()
        mp.run()
    except rospy.ROSInterruptException:
        pass
