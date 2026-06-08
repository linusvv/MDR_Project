#!/usr/bin/python3
"""
YOLO Gating Speed Test
Demonstrates how fast the boolean-gate approach switches between two YOLO models.
Both models stay loaded — only a flag decides which one runs inference.
"""
import time
import threading
import numpy as np
import os

def run_test():
    from ultralytics import YOLO

    grab_path = "/home/ee478_team1/catkin_ws/src/MDR_Project/catkin_ws/src/controller/best.engine"
    nav_path  = "/home/ee478_team1/catkin_ws/src/MDR_Project/catkin_ws/src/robot_web_ui/yolo_models/shops.engine"
    if not os.path.exists(nav_path):
        nav_path = "/home/ee478_team1/catkin_ws/src/MDR_Project/catkin_ws/src/robot_web_ui/yolo_models/navigation.engine"

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    # ── Load both models (simulates startup) ─────────────────────
    print("=" * 60)
    print("YOLO Gating Speed Test")
    print("=" * 60)
    print("\nLoading both models into GPU memory...")

    t0 = time.time()
    grab_model = YOLO(grab_path, task="detect")
    nav_model  = YOLO(nav_path, task="detect")
    print(f"  Both models loaded in {time.time()-t0:.2f}s")

    # Warmup both
    print("  Warming up grab model...")
    grab_model(dummy, conf=0.5, device=0, verbose=False)
    print("  Warming up nav model...")
    nav_model(dummy, conf=0.5, device=0, verbose=False)
    print("  Warmup done.\n")

    # ── Gating simulation ────────────────────────────────────────
    grab_active = False
    nav_active  = True
    gate_lock   = threading.Lock()

    def switch_to_grab():
        nonlocal grab_active, nav_active
        with gate_lock:
            nav_active  = False
            grab_active = True

    def switch_to_nav():
        nonlocal grab_active, nav_active
        with gate_lock:
            grab_active = False
            nav_active  = True

    def run_grab_inference():
        with gate_lock:
            if not grab_active:
                return None
        return grab_model(dummy, conf=0.5, device=0, verbose=False)

    def run_nav_inference():
        with gate_lock:
            if not nav_active:
                return None
        return nav_model(dummy, conf=0.5, device=0, verbose=False)

    # ── Test 1: Gate toggle speed (no inference) ─────────────────
    print("─" * 60)
    print("TEST 1: Pure gate toggle speed (10,000 switches)")
    t0 = time.time()
    for _ in range(5000):
        switch_to_grab()
        switch_to_nav()
    t_toggle = time.time() - t0
    print(f"  10,000 toggles in {t_toggle*1000:.2f} ms")
    print(f"  Per switch: {t_toggle/10000*1e6:.1f} µs")

    # ── Test 2: Gated inference skipping ─────────────────────────
    print("\n" + "─" * 60)
    print("TEST 2: Gated skip speed (grab OFF, calling grab 1000x)")
    grab_active = False
    nav_active  = True

    t0 = time.time()
    for _ in range(1000):
        run_grab_inference()  # should return None instantly
    t_skip = time.time() - t0
    print(f"  1,000 skipped inferences in {t_skip*1000:.2f} ms")
    print(f"  Per skip: {t_skip/1000*1e6:.1f} µs")

    # ── Test 3: Full switch + first inference latency ────────────
    print("\n" + "─" * 60)
    print("TEST 3: Switch NAV→GRAB + first inference after switch")
    nav_active  = True
    grab_active = False

    # Run nav inference to simulate "currently navigating"
    run_nav_inference()

    t0 = time.time()
    switch_to_grab()
    t_switch = time.time() - t0

    t1 = time.time()
    run_grab_inference()
    t_first_inf = time.time() - t1

    t_total = time.time() - t0
    print(f"  Gate switch:       {t_switch*1000:.3f} ms")
    print(f"  First inference:   {t_first_inf*1000:.1f} ms")
    print(f"  Total (switch+inf): {t_total*1000:.1f} ms")

    # ── Test 4: Full switch GRAB→NAV + first inference ───────────
    print("\n" + "─" * 60)
    print("TEST 4: Switch GRAB→NAV + first inference after switch")

    t0 = time.time()
    switch_to_nav()
    t_switch = time.time() - t0

    t1 = time.time()
    run_nav_inference()
    t_first_inf = time.time() - t1

    t_total = time.time() - t0
    print(f"  Gate switch:       {t_switch*1000:.3f} ms")
    print(f"  First inference:   {t_first_inf*1000:.1f} ms")
    print(f"  Total (switch+inf): {t_total*1000:.1f} ms")

    # ── Test 5: Rapid alternation (10 cycles) ────────────────────
    print("\n" + "─" * 60)
    print("TEST 5: Rapid alternation — 10x (switch + infer each)")
    times = []
    for i in range(10):
        t0 = time.time()
        switch_to_grab()
        run_grab_inference()
        switch_to_nav()
        run_nav_inference()
        times.append(time.time() - t0)

    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    print(f"  Per full cycle (2 switches + 2 inferences):")
    print(f"    Avg: {avg:.1f} ms   Std: {std:.1f} ms")
    print(f"    Min: {min(times)*1000:.1f} ms   Max: {max(times)*1000:.1f} ms")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Gate toggle:           {t_toggle/10000*1e6:.0f} µs  (microseconds!)")
    print(f"  Skipped inference:     {t_skip/1000*1e6:.0f} µs")
    print(f"  Switch + 1st infer:    {t_total*1000:.0f} ms")
    print(f"  Full cycle (2x each):  {avg:.0f} ms")
    print(f"\n  → Gating is essentially INSTANT. No model reload needed.")

if __name__ == "__main__":
    run_test()
