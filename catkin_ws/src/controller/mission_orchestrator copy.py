#!/usr/bin/python3
# coding=utf8
"""
Mission Orchestrator Node
═════════════════════════════════════════════════════════════════════════════
The "brain" that sequences a shopping mission. It owns the **global mission
state** (how many of each item still need to be picked, and the high-level
mission phase) and coordinates the other nodes purely over ROS topics, so the
existing nodes (web_server_node, yolo_detector, pick_arm_node) stay almost
untouched and can simply be left running ("just kept up").

─────────────────────────────────────────────────────────────────────────────
Responsibilities (maps 1:1 to the project spec)
─────────────────────────────────────────────────────────────────────────────
  (1) Receive the per-item counts parsed by the LLM and store them as the
      authoritative GLOBAL STATE.                         -> _item_counts_cb
  (2) yolo_detector (grab) stays loaded on GPU but idle.  -> nothing to do here;
      we only flip /yolo_grab/activate when we actually reach a shop.
  (3) pick_arm_node stays up.                             -> nothing to do here.
  (4) When web_server detects/arrives at a shop, switch YOLO
      (nav OFF -> grab ON) so the grab pipeline starts consuming CPU.
                                                          -> _shop_arrived_cb
  (5) Decide what still needs to be picked AND whether the
      grab-YOLO actually sees it in this shop.            -> _pick_at_shop
  (6) Let pick_arm pick; once it has backed up and the target
      is no longer visible, decrement the count.          -> _pick_one_item
  (7) Switch back to nav (grab OFF -> nav ON) and let web_server
      drive to the next shop.                             -> end of _pick_at_shop
  (8) When everything is collected, only SET THE STATE to
      GO_TO_PICKUP (web_server performs the actual drive).-> _maybe_finish_shopping

─────────────────────────────────────────────────────────────────────────────
ROS interface
─────────────────────────────────────────────────────────────────────────────
Subscribes
  /mission/item_counts   std_msgs/String  JSON {"drug":1,"hamburger":2,...}
                                           -> start mission, fill global state
  /mission/shop_arrived  std_msgs/String  shop category, e.g. "PHARMACY"
                                           -> begin picking at this shop
  /mission/finished      std_msgs/Bool    web_server reached the pickup zone
  /pick_arm_heartbeat    std_msgs/String  pick_arm progress strings
  /target_visible        std_msgs/Bool    grab pipeline currently sees target

Publishes
  /yolo_nav/activate     std_msgs/Bool    nav-YOLO on/off  (latched)
  /yolo_grab/activate    std_msgs/Bool    grab-YOLO on/off (latched)
  /target_item           std_msgs/String  class name the grab pipeline targets
  /mission/pick_complete std_msgs/Bool    "done picking at this shop, resume nav"
  /mission/state         std_msgs/String  JSON snapshot of the global state (UI)

The handshake with web_server_node is intentionally thin: web_server keeps
ownership of navigation; this node owns counting, YOLO switching and picking.
"""

import json
import threading

import rospy
from std_msgs.msg import String, Bool


# ═════════════════════════════════════════════════════════════════════════════
# ── HARDCODED CONFIG (constraints are NOT finalized — edit these freely) ─────
# ═════════════════════════════════════════════════════════════════════════════

# The four pickable item types. These strings are the canonical keys used in
# the global state and in /mission/item_counts.
ITEM_TYPES = ["drug", "hamburger", "iceCoffee", "mug"]

# Shop categories. Must match web_server_node.normalize_category() output
# (UPPER CASE):  "CONVENIENCE STORE", "CAFE", "PHARMACY", "HAMBURGER".
SHOP_TYPES = ["CONVENIENCE STORE", "CAFE", "PHARMACY", "HAMBURGER"]

# ── Constraint table: which shop types an item is allowed to appear in. ──────
#   "An item can only be picked at a shop whose category is in this list."
#   iceCoffee can only be in a Cafe or a Convenience store, etc.
#   These are PLACEHOLDERS — adjust once the real rules are confirmed.
ITEM_SHOP_CONSTRAINTS = {
    "drug":      ["PHARMACY", "CONVENIENCE STORE"],
    "hamburger": ["HAMBURGER", "CONVENIENCE STORE"],
    "iceCoffee": ["CAFE", "CONVENIENCE STORE"],
    "mug":       ["CONVENIENCE STORE", "CAFE"],   # TODO: confirm where mugs live
}

# ── Item -> grab-YOLO class name. ───────────────────────────────────────────
#   The grab model (best.engine) emits these class strings. If your trained
#   class names differ (e.g. "ice_coffee"), only change the right-hand side.
ITEM_TO_YOLO_CLASS = {
    "drug":      "drug",
    "hamburger": "hamburger",
    "iceCoffee": "iceCoffee",
    "mug":       "mug",
}

# ── Timing / tolerances ─────────────────────────────────────────────────────
PRESENCE_TIMEOUT   = 6.0    # s to wait for grab-YOLO to confirm an item is here
PICK_CYCLE_TIMEOUT = 60.0   # s to wait for one pick_arm pick-and-place cycle
SETTLE_AFTER_PICK  = 1.5    # s to let the scene settle after pick_arm backs up
DISAPPEAR_CONFIRM  = 1.0    # s the target must stay invisible to count as gone
SWITCH_SETTLE      = 0.3    # s to let latched YOLO switch take effect
MAX_PICKS_PER_ITEM = 10     # safety cap against infinite loops at one shop


# ═════════════════════════════════════════════════════════════════════════════
# ── Orchestrator ────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
class MissionOrchestrator(object):

    # High-level mission phases (this IS the "state" the spec refers to).
    IDLE         = "IDLE"
    SHOPPING     = "SHOPPING"        # roving between shops, nav-YOLO active
    PICKING      = "PICKING"         # parked at a shop, grab-YOLO active
    GO_TO_PICKUP = "GO_TO_PICKUP"    # all items collected, heading to pickup zone
    DONE         = "DONE"            # arrived at pickup zone

    def __init__(self):
        rospy.init_node("mission_orchestrator", anonymous=False)

        # ── GLOBAL STATE ────────────────────────────────────────────────────
        self.lock          = threading.Lock()
        self.remaining     = {k: 0 for k in ITEM_TYPES}   # items left to pick
        self.picked_total  = 0
        self.mission_state = self.IDLE

        # Live signals updated by callbacks
        self._target_visible = False
        self._pick_cycles    = 0          # increments each completed pick_arm cycle
        self._busy           = False      # True while a shop is being serviced

        # ── Publishers ──────────────────────────────────────────────────────
        self.pub_nav   = rospy.Publisher("/yolo_nav/activate",  Bool,   queue_size=1, latch=True)
        self.pub_grab  = rospy.Publisher("/yolo_grab/activate", Bool,   queue_size=1, latch=True)
        self.pub_item  = rospy.Publisher("/target_item",        String, queue_size=1, latch=True)
        self.pub_done  = rospy.Publisher("/mission/pick_complete", Bool, queue_size=1)
        self.pub_state = rospy.Publisher("/mission/state",      String, queue_size=1, latch=True)

        # ── Subscribers ─────────────────────────────────────────────────────
        rospy.Subscriber("/mission/item_counts",  String, self._item_counts_cb)
        rospy.Subscriber("/mission/shop_arrived", String, self._shop_arrived_cb)
        rospy.Subscriber("/mission/finished",     Bool,   self._finished_cb)
        rospy.Subscriber("/pick_arm_heartbeat",   String, self._heartbeat_cb)
        rospy.Subscriber("/target_visible",       Bool,   self._target_visible_cb)

        # Start in a clean roving configuration: nav ON, grab OFF, no target.
        self._switch_to_nav()
        self._publish_state()

        rospy.loginfo("[orchestrator] Ready. Waiting for /mission/item_counts ...")

    # ─────────────────────────────────────────────────────────────────────────
    # ── State helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _publish_state(self):
        with self.lock:
            snapshot = {
                "mission_state": self.mission_state,
                "remaining":     dict(self.remaining),
                "picked_total":  self.picked_total,
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
    # ── YOLO switching ("절체")
    # ─────────────────────────────────────────────────────────────────────────
    def _switch_to_grab(self):
        """nav-YOLO OFF -> grab-YOLO ON. Grab pipeline now consumes CPU/GPU."""
        self.pub_nav.publish(Bool(False))
        rospy.sleep(SWITCH_SETTLE)
        self.pub_grab.publish(Bool(True))
        rospy.sleep(SWITCH_SETTLE)
        rospy.loginfo("[orchestrator] YOLO switched: nav OFF / grab ON")

    def _switch_to_nav(self):
        """grab-YOLO OFF -> nav-YOLO ON. Back to roving/navigation mode."""
        self.pub_item.publish(String(data=""))      # clear grab target
        self.pub_grab.publish(Bool(False))
        rospy.sleep(SWITCH_SETTLE)
        self.pub_nav.publish(Bool(True))
        rospy.sleep(SWITCH_SETTLE)
        rospy.loginfo("[orchestrator] YOLO switched: grab OFF / nav ON")

    # ─────────────────────────────────────────────────────────────────────────
    # ── Callbacks
    # ─────────────────────────────────────────────────────────────────────────
    def _item_counts_cb(self, msg):
        """(1) Store LLM-parsed item counts as the global state and start."""
        try:
            counts = json.loads(msg.data) if msg.data else {}
        except ValueError:
            rospy.logwarn(f"[orchestrator] Bad item_counts JSON: {msg.data!r}")
            return

        with self.lock:
            self.remaining    = {k: int(counts.get(k, 0)) for k in ITEM_TYPES}
            self.picked_total = 0
        rospy.loginfo(f"[orchestrator] New mission. Need: {self.remaining}")

        self._set_state(self.SHOPPING)
        self._switch_to_nav()          # roving config while web_server navigates
        self._publish_state()

    def _shop_arrived_cb(self, msg):
        """(4) web_server reports it has arrived at / detected a shop."""
        shop = (msg.data or "").strip().upper()
        if self.mission_state not in (self.SHOPPING, self.PICKING):
            rospy.loginfo(f"[orchestrator] Ignoring shop_arrived '{shop}' "
                          f"(state={self.mission_state})")
            self.pub_done.publish(Bool(True))   # let web_server move on anyway
            return
        if self._busy:
            rospy.logwarn("[orchestrator] Already servicing a shop; ignoring.")
            return

        # Run the (blocking) pick routine in its own thread so callbacks stay live.
        t = threading.Thread(target=self._pick_at_shop, args=(shop,), daemon=True)
        t.start()

    def _finished_cb(self, msg):
        """(8) web_server reports it reached the pickup zone."""
        if msg.data:
            self._set_state(self.DONE)
            rospy.loginfo("[orchestrator] Mission complete — at pickup zone.")

    def _heartbeat_cb(self, msg):
        # pick_and_place() publishes "picked up object" once it has gripped,
        # backed the chassis up and placed — i.e. one full pick cycle finished.
        if msg.data == "picked up object":
            with self.lock:
                self._pick_cycles += 1

    def _target_visible_cb(self, msg):
        self._target_visible = bool(msg.data)

    # ─────────────────────────────────────────────────────────────────────────
    # ── Per-shop pick routine  (runs in a worker thread)
    # ─────────────────────────────────────────────────────────────────────────
    def _pick_at_shop(self, shop):
        self._busy = True
        self._set_state(self.PICKING)
        rospy.loginfo(f"[orchestrator] === Servicing shop: {shop} ===")

        try:
            # (4) switch to grab so the grab pipeline starts working
            self._switch_to_grab()

            # (5) which still-needed items are allowed in THIS shop type?
            candidates = self._candidates_for_shop(shop)
            if not candidates:
                rospy.loginfo(f"[orchestrator] No needed items can be at {shop}.")
            for item in candidates:
                self._pick_item_repeatedly(item, shop)
                if self._total_remaining() == 0:
                    break
        except Exception as e:   # never strand web_server waiting
            rospy.logerr(f"[orchestrator] Error while picking at {shop}: {e}")
        finally:
            # (7) back to nav and tell web_server it may drive to the next shop
            self._switch_to_nav()
            if self._total_remaining() == 0:
                self._maybe_finish_shopping()
            else:
                self._set_state(self.SHOPPING)
            self.pub_done.publish(Bool(True))
            self._busy = False
            rospy.loginfo(f"[orchestrator] === Done at {shop}. "
                          f"Remaining: {self.remaining} ===")

    def _candidates_for_shop(self, shop):
        with self.lock:
            return [it for it in ITEM_TYPES
                    if self.remaining.get(it, 0) > 0
                    and shop in ITEM_SHOP_CONSTRAINTS.get(it, [])]

    def _pick_item_repeatedly(self, item, shop):
        """Pick this item type until none remain needed or it is no longer here."""
        yolo_class = ITEM_TO_YOLO_CLASS.get(item, item)
        attempts = 0
        while self.remaining.get(item, 0) > 0 and attempts < MAX_PICKS_PER_ITEM:
            attempts += 1
            rospy.loginfo(f"[orchestrator] Targeting '{item}' "
                          f"(yolo='{yolo_class}', need={self.remaining[item]})")
            self.pub_item.publish(String(data=yolo_class))

            # (5) Is it actually in this shop? Wait for the grab pipeline to see it.
            if not self._wait_visible(PRESENCE_TIMEOUT):
                rospy.loginfo(f"[orchestrator] '{item}' not visible at {shop} "
                              f"-> moving on.")
                break

            # (6) pick_arm auto-approaches and picks. Wait for one full cycle.
            if not self._wait_one_pick_cycle(PICK_CYCLE_TIMEOUT):
                rospy.logwarn(f"[orchestrator] Pick cycle timed out for '{item}'.")
                break

            # (6) After pick_arm has backed up: if the target is GONE, it counts.
            rospy.sleep(SETTLE_AFTER_PICK)
            if self._confirm_disappeared(DISAPPEAR_CONFIRM):
                with self.lock:
                    self.remaining[item] -= 1
                    self.picked_total   += 1
                rospy.loginfo(f"[orchestrator] '{item}' picked & gone. "
                              f"Remaining[{item}]={self.remaining[item]}")
                self._publish_state()
            else:
                rospy.logwarn(f"[orchestrator] '{item}' still visible after pick "
                              f"-> grasp likely failed, retrying.")

    # ─────────────────────────────────────────────────────────────────────────
    # ── Waiting helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _wait_visible(self, timeout):
        """Return True as soon as the grab target becomes visible."""
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        r = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self._target_visible:
                return True
            r.sleep()
        return False

    def _wait_one_pick_cycle(self, timeout):
        """Block until pick_arm reports one completed pick ('picked up object')."""
        with self.lock:
            start_cycles = self._pick_cycles
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        r = rospy.Rate(10)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            with self.lock:
                if self._pick_cycles > start_cycles:
                    return True
            r.sleep()
        return False

    def _confirm_disappeared(self, confirm_time):
        """Target counts as 'gone' only if invisible continuously for confirm_time."""
        r = rospy.Rate(20)
        steady_until = rospy.Time.now() + rospy.Duration(confirm_time)
        while not rospy.is_shutdown() and rospy.Time.now() < steady_until:
            if self._target_visible:
                return False     # reappeared -> not gone
            r.sleep()
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # ── Completion
    # ─────────────────────────────────────────────────────────────────────────
    def _maybe_finish_shopping(self):
        """(8) All items collected -> only SET the state. web_server drives to
        the pickup zone; we flip to DONE when /mission/finished arrives."""
        self._set_state(self.GO_TO_PICKUP)
        rospy.loginfo("[orchestrator] All items collected. "
                      "State set to GO_TO_PICKUP (web_server drives to zone).")


if __name__ == "__main__":
    MissionOrchestrator()
    rospy.spin()
