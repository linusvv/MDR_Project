#!/usr/bin/python3
# coding=utf8
"""
Mission Orchestrator Node  (AprilTag hand-off version)
═════════════════════════════════════════════════════════════════════════════
The "brain" that owns the GLOBAL MISSION STATE and coordinates the navigation
teammate, the grab YOLO and the arm — all over ROS topics.

Division of labour
──────────────────
  • NAV TEAMMATE (separate node, not ours): receives a SIGNBOARD number (an
    AprilTag, 1..16), drives to it, uses signboards.yaml (tag / store / direction)
    to find the shop, stops in front of it, and HANDS CONTROL BACK to us by
    publishing /mission/shop_arrived.
  • THIS NODE: decides which signboard to send next, grants control by
    publishing the signboard number, and once control is handed back it switches
    the grab YOLO on, checks whether the needed item is actually there, drives
    the arm to pick (possibly several items), counts items as they disappear,
    and then hands the next signboard back to the nav teammate. When everything
    is collected it sends the Pickup-Point signboard and sets the finish flag.

Hand-off protocol (matches the agreed interface)
────────────────────────────────────────────────
  OUT  /mission/goto_signboard  std_msgs/Int32   signboard # to drive to
                                                 (publishing == "go + you have control")
  IN   /mission/shop_arrived    std_msgs/String  teammate returns control
                                                 (content optional; we service the
                                                  signboard we last dispatched)
  OUT  /mission/finished        std_msgs/Bool    set True when heading to pickup
  OUT  /mission/goto_info       std_msgs/String  JSON {signboard, store_type, direction}
                                                 (debug / optional for the teammate)

Other topics
────────────
  IN   /mission/item_counts  std_msgs/String  JSON {"drug":..,"hamburger":..,..}
                                              -> set global state, start mission
  IN   /pick_arm_heartbeat   std_msgs/String  arm progress ("picked up object" ...)
  IN   /target_visible       std_msgs/Bool    grab pipeline currently sees target
  OUT  /yolo_grab/activate    std_msgs/Bool   grab YOLO on/off
  OUT  /yolo_nav/activate     std_msgs/Bool   nav YOLO off (CPU saving)
  OUT  /target_item           std_msgs/String grab target class
  OUT  /mission/state         std_msgs/String JSON snapshot of the global state

Counting rule
─────────────
  After the arm picks and backs up, if the grab target is no longer visible the
  item counts as collected (remaining[item] -= 1). Multiple identical items at
  one shop are handled by re-checking visibility after each pick.
"""

import os
import re
import math
import json
import threading

import rospy
import rospkg
import yaml
from std_msgs.msg import String, Bool, Int32


# ═════════════════════════════════════════════════════════════════════════════
# ── HARDCODED CONFIG  (edit right before the competition) ────────────────────
# ═════════════════════════════════════════════════════════════════════════════

# Grab-YOLO class names (confirmed from model.names). These are the four
# pickable items and the canonical keys of the global state.
ITEM_TYPES = ["drug", "hamburger", "iceCoffee", "mug"]
ITEM_ORDER = ITEM_TYPES

# Item -> grab-YOLO class string published on /target_item.
ITEM_TO_YOLO_CLASS = {
    "drug":      "drug",
    "hamburger": "hamburger",
    "iceCoffee": "iceCoffee",
    "mug":       "mug",
}

# Which STORE TYPES (as written in signboards.yaml) each item may be found in.
# NOTE: not finalized — adjust before the competition. Strings are compared
# case-insensitively against the yaml 'store_type' values.
ITEM_SHOP_CONSTRAINTS = {
    "drug":      ["Pharmacy",  "Convenience Store"],
    "hamburger": ["Convenience Store", "Hamburger"],
    "iceCoffee": ["Cafe",      "Convenience Store"],
    "mug":       ["Cafe",      "Convenience Store"],
    # Each list is ordered by preference: the primary store type first, then the
    # Convenience Store as a fallback (every item is also sold there).
}

# The store_type that represents the final drop-off in signboards.yaml.
PICKUP_STORE_TYPE = "Pickup Point"

# Optional explicit pickup-zone coordinate (map frame), behind stores 5/7.
# If None, the robot just navigates to a Pickup-Point signboard (apriltag only).
# Set e.g. [0.0, 1.9] once the exact drop-off point is known.
PICKUP_LOCATION = None

# How far (m) in front of a store the robot parks before picking. The robot
# approaches along the signboard->store line (corridor side) and faces the store.
STORE_APPROACH_DIST = 0.30

# ── Timing / tolerances ─────────────────────────────────────────────────────
PRESENCE_TIMEOUT   = 6.0    # s to wait for grab-YOLO to confirm an item is here
PICK_CYCLE_TIMEOUT = 60.0   # s to wait for one pick_arm pick-and-place cycle
SETTLE_AFTER_PICK  = 1.5    # s to let the scene settle after pick_arm backs up
DISAPPEAR_CONFIRM  = 1.0    # s the target must stay invisible to count as gone
SWITCH_SETTLE      = 0.3    # s to let a latched YOLO switch take effect
MAX_PICKS_PER_SHOP = 12     # safety cap on picks at a single shop


# ═════════════════════════════════════════════════════════════════════════════
# ── Orchestrator ────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
class MissionOrchestrator(object):

    # Mission phases (this is the high-level "state").
    IDLE         = "IDLE"
    NAVIGATING   = "NAVIGATING"      # nav teammate driving to a signboard
    PICKING      = "PICKING"         # we have control at a shop, grab YOLO on
    GO_TO_PICKUP = "GO_TO_PICKUP"    # all items collected, heading to pickup zone
    DONE         = "DONE"            # arrived at pickup zone

    def __init__(self):
        rospy.init_node("mission_orchestrator", anonymous=False)

        # ── Configurable topic names (align these with the nav teammate) ─────
        self.goto_topic     = rospy.get_param("~goto_topic",    "/mission/goto_signboard")
        self.arrived_topic  = rospy.get_param("~arrived_topic", "/mission/shop_arrived")
        self.counts_topic   = rospy.get_param("~counts_topic",  "/mission/item_counts")
        self.finished_topic = rospy.get_param("~finished_topic","/mission/finished")

        # ── GLOBAL STATE ────────────────────────────────────────────────────
        self.lock          = threading.Lock()
        self.remaining     = {k: 0 for k in ITEM_TYPES}
        self.picked_total  = 0
        self.mission_state = self.IDLE

        # Dispatch bookkeeping
        self._current_sb        = None    # signboard # we last sent
        self._current_storetypes = []     # store_types at that signboard
        self._current_target    = None    # {sb, location, corridor_yaw, store_type}
        self._tried_signboards  = set()   # signboards already serviced this mission
        self._tried_stores      = set()   # store locations already serviced
        self._busy              = False

        # Live signals
        self._target_visible = False
        self._pick_cycles    = 0
        # The item the arm is currently picking. While this is set, a
        # "picked up object" heartbeat decrements remaining[this item] — so the
        # count always drops the moment a real pick completes.
        self._active_pick_item = None

        # Counting policy: decrement on each completed pick (the "picked up
        # object" heartbeat). If True, ALSO require the grab target to vanish
        # afterward as confirmation (more strict, but can stall counting if the
        # YOLO keeps seeing it). Default False so a successful pick always counts.
        self.require_disappear = rospy.get_param("~require_disappear", False)

        # ── Load the arena map (signboards.yaml) ─────────────────────────────
        self._load_signboards()

        # ── Publishers ──────────────────────────────────────────────────────
        self.pub_goto  = rospy.Publisher(self.goto_topic,     Int32,  queue_size=1, latch=True)
        # Rich goto for web_server (Plan B): signboard + store location + heading.
        self.pub_goto_rich = rospy.Publisher("/mission/goto", String, queue_size=1, latch=True)
        self.pub_info  = rospy.Publisher("/mission/goto_info", String, queue_size=1, latch=True)
        self.pub_fin   = rospy.Publisher(self.finished_topic, Bool,   queue_size=1, latch=True)
        self.pub_grab  = rospy.Publisher("/yolo_grab/activate", Bool,   queue_size=1, latch=True)
        self.pub_nav   = rospy.Publisher("/yolo_nav/activate",  Bool,   queue_size=1, latch=True)
        self.pub_item  = rospy.Publisher("/target_item",        String, queue_size=1, latch=True)
        self.pub_state = rospy.Publisher("/mission/state",      String, queue_size=1, latch=True)

        # ── Subscribers ─────────────────────────────────────────────────────
        rospy.Subscriber(self.counts_topic,   String, self._item_counts_cb)
        rospy.Subscriber(self.arrived_topic,  String, self._shop_arrived_cb)
        rospy.Subscriber("/pick_arm_heartbeat", String, self._heartbeat_cb)
        rospy.Subscriber("/target_visible",     Bool,   self._target_visible_cb)

        # Idle config: grab + nav YOLO off, no target, finish flag low.
        self.pub_grab.publish(Bool(False))
        self.pub_nav.publish(Bool(False))
        self.pub_item.publish(String(data=""))
        self.pub_fin.publish(Bool(False))
        self._publish_state()

        rospy.loginfo(f"[orchestrator] Ready. goto='{self.goto_topic}', "
                      f"arrived='{self.arrived_topic}'. Waiting for item counts ...")

    # ─────────────────────────────────────────────────────────────────────────
    # ── signboards.yaml loading
    # ─────────────────────────────────────────────────────────────────────────
    def _load_signboards(self):
        path = rospy.get_param("~signboards_yaml", "")
        candidates = [path] if path else []
        try:
            pkg = rospkg.RosPack().get_path("controller")
            candidates.append(os.path.join(pkg, "signboards.yaml"))
        except Exception:
            pass
        try:
            pkg = rospkg.RosPack().get_path("controller")
            candidates.append(os.path.join(pkg, "signboards_cheet.yaml"))
        except Exception:
            pass
        candidates += [
            "/home/ee478_team1/catkin_ws/src/MDR_Project/catkin_ws/src/controller/signboards_cheet.yaml",
            "/home/ee478_team1/catkin_ws/src/MDR_Project/catkin_ws/src/controller/signboards.yaml",
        ]

        cfg = None
        used = None
        for p in candidates:
            if p and os.path.exists(p):
                with open(p, "r") as f:
                    cfg = yaml.safe_load(f)
                used = p
                break

        import math
        _DIRO = {"up": 0.0, "left": 90.0, "right": -90.0, "down": 180.0}

        # signboard# -> (x, y)
        self.signboard_xy = {}
        # signboard# -> [store_type, ...]
        self.signboard_storetypes = {}
        # store_type(lower) -> [signboard#, ...]   (kept for compatibility)
        self.storetype_signboards = {}
        # store_type(lower) -> [ {sb, location:[x,y]|None, corridor_yaw_deg}, ... ]
        # This is the Plan-B routing table: each "target" carries the actual
        # store location and the heading from the signboard toward the store.
        self.storetype_targets = {}

        if not cfg:
            rospy.logwarn("[orchestrator] signboards.yaml not found — dispatch will be empty!")
            return

        for board_name, body in cfg.items():
            m = re.search(r"(\d+)", str(board_name))
            if not m:
                continue
            sb = int(m.group(1))
            # signboard heading psi (deg) and position from the tag field
            psi = 0.0
            if isinstance(body, dict) and isinstance(body.get("tag"), (list, tuple)) and len(body["tag"]) >= 6:
                try:
                    psi = float(body["tag"][5])
                    self.signboard_xy[sb] = (float(body["tag"][2]), float(body["tag"][3]))
                except (ValueError, TypeError):
                    psi = 0.0
            # Support both {stores: {id: {...}}} and {id: {...}} layouts.
            stores = body.get("stores", body) if isinstance(body, dict) else {}
            for _id, data in stores.items():
                if not isinstance(data, dict) or "store_type" not in data:
                    continue
                st = str(data["store_type"]).strip()
                key = st.lower()
                direction = str(data.get("direction", "Up")).strip().lower()
                corridor_yaw = (psi + _DIRO.get(direction, 0.0)) % 360.0
                loc = data.get("location", None)
                if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    loc = [float(loc[0]), float(loc[1])]
                else:
                    loc = None
                wp = data.get("waypoint", None)
                if isinstance(wp, (list, tuple)) and len(wp) >= 2:
                    wp = [float(wp[0]), float(wp[1])]
                else:
                    wp = None

                self.signboard_storetypes.setdefault(sb, [])
                if st not in self.signboard_storetypes[sb]:
                    self.signboard_storetypes[sb].append(st)
                self.storetype_signboards.setdefault(key, [])
                if sb not in self.storetype_signboards[key]:
                    self.storetype_signboards[key].append(sb)
                self.storetype_targets.setdefault(key, []).append({
                    "sb": sb, "location": loc, "waypoint": wp,
                    "corridor_yaw": corridor_yaw,
                })

        rospy.loginfo(f"[orchestrator] Loaded signboards from {used}: "
                      f"{len(self.signboard_storetypes)} boards, "
                      f"store types: {sorted(self.storetype_signboards.keys())}")

    # ─────────────────────────────────────────────────────────────────────────
    # ── State helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _publish_state(self):
        with self.lock:
            snapshot = {
                "mission_state": self.mission_state,
                "remaining":     dict(self.remaining),
                "picked_total":  self.picked_total,
                "current_signboard": self._current_sb,
            }
        self.pub_state.publish(String(data=json.dumps(snapshot)))

    def _set_state(self, new_state):
        with self.lock:
            self.mission_state = new_state
        rospy.loginfo(f"[orchestrator] mission_state -> {new_state}")
        self._publish_state()

    def _total_remaining(self):
        with self.lock:
            return sum(self.remaining.values())

    # ─────────────────────────────────────────────────────────────────────────
    # ── grab YOLO switching ("절체")
    # ─────────────────────────────────────────────────────────────────────────
    def _grab_on(self):
        self.pub_nav.publish(Bool(False))
        rospy.sleep(SWITCH_SETTLE)
        self.pub_grab.publish(Bool(True))
        rospy.sleep(SWITCH_SETTLE)
        rospy.loginfo("[orchestrator] grab YOLO ON")

    def _grab_off(self):
        self.pub_item.publish(String(data=""))
        self.pub_grab.publish(Bool(False))
        rospy.sleep(SWITCH_SETTLE)
        rospy.loginfo("[orchestrator] grab YOLO OFF")

    # ─────────────────────────────────────────────────────────────────────────
    # ── Callbacks
    # ─────────────────────────────────────────────────────────────────────────
    def _item_counts_cb(self, msg):
        """Store LLM-parsed item counts as the global state and dispatch."""
        try:
            counts = json.loads(msg.data) if msg.data else {}
        except ValueError:
            rospy.logwarn(f"[orchestrator] Bad item_counts JSON: {msg.data!r}")
            return
        with self.lock:
            self.remaining    = {k: int(counts.get(k, 0)) for k in ITEM_TYPES}
            self.picked_total = 0
            self._tried_signboards = set()
            self._tried_stores = set()
        rospy.loginfo(f"[orchestrator] New mission. Need: {self.remaining}")
        self.pub_fin.publish(Bool(False))
        self._dispatch_next()

    def _shop_arrived_cb(self, msg):
        """Nav teammate hands control back: we are in front of a shop (or the
        pickup zone). Service it in a worker thread so callbacks stay live."""
        if self.mission_state == self.GO_TO_PICKUP:
            # We were heading to the pickup zone — mission complete.
            self._set_state(self.DONE)
            self.pub_fin.publish(Bool(True))
            rospy.loginfo("[orchestrator] Arrived at pickup zone — DONE.")
            return
        if self.mission_state != self.NAVIGATING:
            rospy.logwarn(f"[orchestrator] shop_arrived ignored (state={self.mission_state})")
            return
        if self._busy:
            return
        t = threading.Thread(target=self._service_current_shop, daemon=True)
        t.start()

    def _heartbeat_cb(self, msg):
        if msg.data != "picked up object":
            return
        # A real pick just finished. Bump the cycle counter (used by
        # _wait_one_pick_cycle) AND decrement the active item right here, so the
        # count never gets stuck even if the per-shop loop's timing is off.
        counted_item = None
        with self.lock:
            self._pick_cycles += 1
            item = self._active_pick_item
            if item is not None and self.remaining.get(item, 0) > 0:
                self.remaining[item] -= 1
                self.picked_total   += 1
                counted_item = item
                if self.remaining[item] <= 0:
                    self._active_pick_item = None
        if counted_item is not None:
            rospy.loginfo(f"[orchestrator] COUNTED '{counted_item}' on pick. "
                          f"Remaining[{counted_item}]={self.remaining[counted_item]}")
            self._publish_state()

    def _target_visible_cb(self, msg):
        self._target_visible = bool(msg.data)

    # ─────────────────────────────────────────────────────────────────────────
    # ── Dispatch: choose and send the next signboard
    # ─────────────────────────────────────────────────────────────────────────
    def _dispatch_next(self):
        """Pick the next store that can supply a still-needed item and send the
        nav command (signboard + store location). If nothing is left, pickup."""
        if self._total_remaining() == 0:
            self._dispatch_pickup()
            return

        choice = self._choose_target()
        if choice is None:
            rospy.logwarn("[orchestrator] No store left for the remaining items "
                          f"{self.remaining}. Heading to pickup anyway.")
            self._dispatch_pickup()
            return

        store_type, tgt = choice
        self._current_target = dict(tgt, store_type=store_type)
        self._current_sb = tgt["sb"]
        self._current_storetypes = [store_type]
        self._set_state(self.NAVIGATING)

        loc = tgt["location"]
        wp = tgt.get("waypoint") or loc   # fall back to storefront if no waypoint
        goto = {
            "signboard":        tgt["sb"],
            "store_x":          loc[0],
            "store_y":          loc[1],
            "waypoint_x":       wp[0],
            "waypoint_y":       wp[1],
            "corridor_yaw_deg": tgt["corridor_yaw"],
            "store_type":       store_type,
            "approach_dist":    STORE_APPROACH_DIST,
            "pickup":           False,
        }
        # Rich command (web_server drives both legs). Also publish the bare
        # signboard number for the nav teammate's node (compatibility).
        self.pub_goto_rich.publish(String(data=json.dumps(goto)))
        self.pub_goto.publish(Int32(data=tgt["sb"]))
        self.pub_info.publish(String(data=json.dumps(goto)))
        rospy.loginfo(f"[orchestrator] -> GOTO signboard {tgt['sb']} then store "
                      f"{store_type}@({loc[0]:.2f},{loc[1]:.2f}) "
                      f"yaw={tgt['corridor_yaw']:.0f}°. Remaining: {self.remaining}")

    def _choose_target(self):
        """Return (store_type, target) for the next needed item — the nearest
        un-serviced store of an allowed type. Target carries location + heading."""
        for item in ITEM_ORDER:
            with self.lock:
                if self.remaining.get(item, 0) <= 0:
                    continue
            for st in ITEM_SHOP_CONSTRAINTS.get(item, []):
                # Unique un-serviced stores of this type; for each store pick the
                # signboard closest to it (shortest, safest approach leg).
                best_per_store = {}
                for tgt in self.storetype_targets.get(st.lower(), []):
                    loc = tgt.get("location")
                    if not loc:
                        continue
                    key = (round(loc[0], 2), round(loc[1], 2))
                    if key in self._tried_stores:
                        continue
                    sbxy = self.signboard_xy.get(tgt["sb"])
                    d = (math.hypot(sbxy[0] - loc[0], sbxy[1] - loc[1])
                         if sbxy else 0.0)
                    if key not in best_per_store or d < best_per_store[key][0]:
                        best_per_store[key] = (d, tgt)
                if best_per_store:
                    # nearest store overall (by signboard-store leg length)
                    _, tgt = min(best_per_store.values(), key=lambda v: v[0])
                    return st, tgt
        return None

    def _dispatch_pickup(self):
        """Send the pickup zone and raise the finish flag."""
        self._set_state(self.GO_TO_PICKUP)
        self.pub_fin.publish(Bool(True))          # finish flag

        targets = self.storetype_targets.get(PICKUP_STORE_TYPE.lower(), [])
        if not targets and PICKUP_LOCATION is None:
            rospy.logwarn("[orchestrator] No Pickup Point in signboards.yaml!")
            self._publish_state()
            return
        tgt = targets[0] if targets else {"sb": 0, "corridor_yaw": 0.0}
        sb = tgt["sb"]
        self._current_sb = sb
        loc = PICKUP_LOCATION  # may be None -> web_server does apriltag-only leg
        goto = {
            "signboard":        sb,
            "store_x":          (loc[0] if loc else None),
            "store_y":          (loc[1] if loc else None),
            "corridor_yaw_deg": tgt.get("corridor_yaw", 0.0),
            "store_type":       PICKUP_STORE_TYPE,
            "approach_dist":    STORE_APPROACH_DIST,
            "pickup":           True,
        }
        self.pub_goto_rich.publish(String(data=json.dumps(goto)))
        self.pub_goto.publish(Int32(data=sb))
        self.pub_info.publish(String(data=json.dumps(goto)))
        rospy.loginfo(f"[orchestrator] All items collected. -> Sent pickup "
                      f"(signboard {sb}) and set finish flag.")

    # ─────────────────────────────────────────────────────────────────────────
    # ── Service a shop (we have control)
    # ─────────────────────────────────────────────────────────────────────────
    def _service_current_shop(self):
        self._busy = True
        tgt = self._current_target or {}
        sb = self._current_sb
        store_type = tgt.get("store_type", "")
        storetypes = [store_type] if store_type else list(self._current_storetypes)
        self._set_state(self.PICKING)
        rospy.loginfo(f"[orchestrator] === At {store_type} store (signboard {sb}) ===")
        try:
            self._grab_on()
            # Which still-needed items are sold at this store's type?
            candidates = self._candidates_for(storetypes)
            if not candidates:
                rospy.loginfo(f"[orchestrator] No needed item belongs at {storetypes}.")
            picks = 0
            for item in candidates:
                picks += self._pick_item_repeatedly(item, sb, picks)
                if self._total_remaining() == 0:
                    break
        except Exception as e:
            rospy.logerr(f"[orchestrator] Error servicing signboard {sb}: {e}")
        finally:
            self._grab_off()
            self._tried_signboards.add(sb)
            loc = tgt.get("location")
            if loc:
                self._tried_stores.add((round(loc[0], 2), round(loc[1], 2)))
            self._busy = False
            rospy.loginfo(f"[orchestrator] === Done at signboard {sb}. "
                          f"Remaining: {self.remaining} ===")
            # Dispatch the next store (or pickup).
            self._dispatch_next()

    def _candidates_for(self, storetypes):
        sts = [s.lower() for s in storetypes]
        with self.lock:
            remaining = dict(self.remaining)
        out = []
        for item in ITEM_ORDER:
            if remaining.get(item, 0) <= 0:
                continue
            allowed = [s.lower() for s in ITEM_SHOP_CONSTRAINTS.get(item, [])]
            if any(s in allowed for s in sts):
                out.append(item)
        return out

    def _pick_item_repeatedly(self, item, sb, picks_so_far):
        """Pick this item until it is no longer needed or no longer visible here.
        Returns the number of successful picks. (Step 7: several at one shop.)"""
        yolo_class = ITEM_TO_YOLO_CLASS.get(item, item)
        picked = 0
        while (self.remaining.get(item, 0) > 0
               and (picks_so_far + picked) < MAX_PICKS_PER_SHOP):
            # Tell the heartbeat handler which item the arm is going for, so the
            # "picked up object" heartbeat decrements THIS item.
            with self.lock:
                self._active_pick_item = item
            rospy.loginfo(f"[orchestrator] Targeting '{item}' "
                          f"(need={self.remaining[item]}) at signboard {sb}")
            self.pub_item.publish(String(data=yolo_class))

            # Step 5: is the item actually present here?
            if not self._wait_visible(PRESENCE_TIMEOUT):
                rospy.loginfo(f"[orchestrator] '{item}' not visible at signboard {sb}.")
                break

            # Step 6: arm picks autonomously; wait for one full cycle (the
            # pick_arm "picked up object" heartbeat marks completion AND, in
            # _heartbeat_cb, decrements remaining[item]).
            before = self.remaining.get(item, 0)
            if not self._wait_one_pick_cycle(PICK_CYCLE_TIMEOUT):
                rospy.logwarn(f"[orchestrator] Pick cycle timed out for '{item}' "
                              f"(no 'picked up object' heartbeat).")
                break

            rospy.sleep(SETTLE_AFTER_PICK)

            # The heartbeat normally already counted it. Fall back to counting
            # here only if it somehow did not, so we never loop on one object.
            after = self.remaining.get(item, 0)
            if after < before:
                picked += (before - after)
            else:
                with self.lock:
                    if self.remaining.get(item, 0) > 0:
                        self.remaining[item] -= 1
                        self.picked_total   += 1
                picked += 1
                self._publish_state()
            rospy.loginfo(f"[orchestrator] Picked '{item}'. "
                          f"Remaining[{item}]={self.remaining.get(item, 0)}")
            # Step 7: loop back -> re-check visibility to see if there is another
            # one of this item to grab here. Stops automatically once the needed
            # count hits 0 (while condition) or nothing is visible anymore.

        # Done with this item at this shop: stop the arm from re-targeting it.
        with self.lock:
            self._active_pick_item = None
        self.pub_item.publish(String(data=""))
        return picked

    # ─────────────────────────────────────────────────────────────────────────
    # ── Waiting helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _wait_visible(self, timeout):
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        r = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self._target_visible:
                return True
            r.sleep()
        return False

    def _wait_one_pick_cycle(self, timeout):
        with self.lock:
            start = self._pick_cycles
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        r = rospy.Rate(10)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            with self.lock:
                if self._pick_cycles > start:
                    return True
            r.sleep()
        return False

    def _confirm_disappeared(self, confirm_time):
        steady_until = rospy.Time.now() + rospy.Duration(confirm_time)
        r = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.Time.now() < steady_until:
            if self._target_visible:
                return False
            r.sleep()
        return True


if __name__ == "__main__":
    MissionOrchestrator()
    rospy.spin()
