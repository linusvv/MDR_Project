#!/usr/bin/python3
"""
YOLO VRAM / Load Time Benchmark
Tests how long it takes to load/unload each YOLO model and measures memory impact.
"""
import time
import gc
import os
import subprocess

def get_memory_mb():
    """Get current process RSS memory in MB."""
    import psutil
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / 1024 / 1024

def get_system_memory():
    """Get system free memory from /proc/meminfo."""
    with open('/proc/meminfo') as f:
        for line in f:
            if 'MemAvailable' in line:
                return int(line.split()[1]) / 1024  # MB
    return 0

def run_test():
    print("=" * 60)
    print("YOLO VRAM / Load Time Benchmark")
    print("=" * 60)

    from ultralytics import YOLO
    import torch

    grab_path = "/home/ee478_team1/catkin_ws/src/MDR_Project/catkin_ws/src/controller/best.engine"
    nav_engine = "/home/ee478_team1/catkin_ws/src/MDR_Project/catkin_ws/src/robot_web_ui/yolo_models/navigation.engine"
    nav_pt     = "/home/ee478_team1/catkin_ws/src/MDR_Project/catkin_ws/src/robot_web_ui/yolo_models/navigation.pt"

    # Use .engine if available, else .pt
    nav_path = nav_engine if os.path.exists(nav_engine) else nav_pt

    print(f"\nGrab model: {os.path.basename(grab_path)} ({os.path.getsize(grab_path)/1024/1024:.1f} MB)")
    print(f"Nav  model: {os.path.basename(nav_path)} ({os.path.getsize(nav_path)/1024/1024:.1f} MB)")
    print(f"\nSystem free memory: {get_system_memory():.0f} MB")
    print(f"Process RSS: {get_memory_mb():.0f} MB")

    # ── Test 1: Load grab model ──────────────────────────────────
    print("\n" + "─" * 60)
    print("TEST 1: Loading GRAB model (best.engine)...")
    mem_before = get_system_memory()
    rss_before = get_memory_mb()

    t0 = time.time()
    grab_model = YOLO(grab_path, task="detect")
    t_load_grab = time.time() - t0

    mem_after = get_system_memory()
    rss_after = get_memory_mb()
    grab_vram = mem_before - mem_after

    print(f"  Load time:    {t_load_grab:.2f}s")
    print(f"  System free:  {mem_before:.0f} → {mem_after:.0f} MB (Δ = {grab_vram:.0f} MB)")
    print(f"  Process RSS:  {rss_before:.0f} → {rss_after:.0f} MB (Δ = {rss_after-rss_before:.0f} MB)")

    # ── Test 1b: Run one warmup inference ────────────────────────
    import numpy as np
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    print("\n  Running warmup inference (grab)...")
    t0 = time.time()
    grab_model(dummy, conf=0.5, device=0, verbose=False)
    t_warmup_grab = time.time() - t0
    mem_after_inf = get_system_memory()
    print(f"  Warmup inference: {t_warmup_grab:.2f}s")
    print(f"  System free after inference: {mem_after_inf:.0f} MB (Δ = {mem_after - mem_after_inf:.0f} MB extra)")

    # ── Test 2: Unload grab model ────────────────────────────────
    print("\n" + "─" * 60)
    print("TEST 2: Unloading GRAB model...")
    mem_before_unload = get_system_memory()

    t0 = time.time()
    del grab_model
    torch.cuda.empty_cache()
    gc.collect()
    t_unload_grab = time.time() - t0

    time.sleep(1)  # Give OS time to reclaim
    mem_after_unload = get_system_memory()
    grab_freed = mem_after_unload - mem_before_unload

    print(f"  Unload time:  {t_unload_grab:.3f}s")
    print(f"  Memory freed: {grab_freed:.0f} MB")
    print(f"  System free:  {mem_after_unload:.0f} MB")

    # ── Test 3: Load nav model ───────────────────────────────────
    print("\n" + "─" * 60)
    print(f"TEST 3: Loading NAV model ({os.path.basename(nav_path)})...")
    mem_before = get_system_memory()

    t0 = time.time()
    nav_model = YOLO(nav_path, task="detect")
    t_load_nav = time.time() - t0

    mem_after = get_system_memory()
    nav_vram = mem_before - mem_after

    print(f"  Load time:    {t_load_nav:.2f}s")
    print(f"  VRAM used:    ~{nav_vram:.0f} MB")

    # Warmup
    print("  Running warmup inference (nav)...")
    t0 = time.time()
    nav_model(dummy, conf=0.5, device=0, verbose=False)
    t_warmup_nav = time.time() - t0
    print(f"  Warmup inference: {t_warmup_nav:.2f}s")

    # ── Test 4: Unload nav model ─────────────────────────────────
    print("\n" + "─" * 60)
    print("TEST 4: Unloading NAV model...")
    mem_before_unload = get_system_memory()

    t0 = time.time()
    del nav_model
    torch.cuda.empty_cache()
    gc.collect()
    t_unload_nav = time.time() - t0

    time.sleep(1)
    mem_after_unload = get_system_memory()
    nav_freed = mem_after_unload - mem_before_unload

    print(f"  Unload time:  {t_unload_nav:.3f}s")
    print(f"  Memory freed: {nav_freed:.0f} MB")

    # ── Test 5: Full swap cycle ──────────────────────────────────
    print("\n" + "─" * 60)
    print("TEST 5: Full swap cycle (load grab → unload grab → load nav)...")

    t0 = time.time()
    grab_model = YOLO(grab_path, task="detect")
    grab_model(dummy, conf=0.5, device=0, verbose=False)
    t_phase1 = time.time() - t0

    t0 = time.time()
    del grab_model
    torch.cuda.empty_cache()
    gc.collect()
    t_phase2 = time.time() - t0

    t0 = time.time()
    nav_model = YOLO(nav_path, task="detect")
    nav_model(dummy, conf=0.5, device=0, verbose=False)
    t_phase3 = time.time() - t0

    t_total = t_phase1 + t_phase2 + t_phase3

    del nav_model
    torch.cuda.empty_cache()
    gc.collect()

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  GRAB model ({os.path.basename(grab_path)}):")
    print(f"    Load time:     {t_load_grab:.2f}s")
    print(f"    VRAM usage:    ~{grab_vram:.0f} MB")
    print(f"    Unload time:   {t_unload_grab:.3f}s")
    print(f"    Warmup infer:  {t_warmup_grab:.2f}s")
    print(f"  NAV model ({os.path.basename(nav_path)}):")
    print(f"    Load time:     {t_load_nav:.2f}s")
    print(f"    VRAM usage:    ~{nav_vram:.0f} MB")
    print(f"    Unload time:   {t_unload_nav:.3f}s")
    print(f"    Warmup infer:  {t_warmup_nav:.2f}s")
    print(f"\n  Full swap cycle: {t_total:.2f}s")
    print(f"    (load: {t_phase1:.2f}s + unload: {t_phase2:.3f}s + load: {t_phase3:.2f}s)")
    print(f"\n  → If swap < ~3s, full unload/reload is viable!")
    print(f"  → If swap > ~5s, keep gating approach (both in VRAM)")

if __name__ == "__main__":
    run_test()
