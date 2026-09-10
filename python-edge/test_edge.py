"""
Self-contained verification test suite for AeroVision Edge Pipeline.
Runs pure Python assertions with zero external test dependencies (<0.2s).
"""
import os
import sys
import json
import math

# Ensure python-edge is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from mapper import (
    AreaGridMap,
    MonocularObstacleProjector,
    ReactiveAvoidanceEngine,
    CELL_UNKNOWN,
    CELL_FREE,
    CELL_OBSTACLE,
    CELL_TARGET,
)
from detector import DroneEdgeDetector


def test_coordinate_transforms():
    """Verify world-to-grid and grid-to-world roundtripping and bounds."""
    gmap = AreaGridMap(site_width=10.0, site_length=10.0, standoff_distance=5.0, resolution=0.2)
    assert gmap.cols == 50, f"Expected 50 cols, got {gmap.cols}"
    assert gmap.rows == 75, f"Expected 75 rows, got {gmap.rows}"

    # Drone launch pad at (5.0, 0.2)
    col, row = gmap.world_to_grid(5.0, 0.2)
    assert 0 <= col < gmap.cols, f"Col {col} out of bounds"
    assert 0 <= row < gmap.rows, f"Row {row} out of bounds"
    # Row should be near bottom (South)
    assert row == 73, f"Expected row 73 for Y=0.2m, got {row}"

    # Roundtrip check
    wx, wy = gmap.grid_to_world(col, row)
    assert abs(wx - 5.1) < 0.2, f"X roundtrip error: {wx}"
    assert abs(wy - 0.3) < 0.2, f"Y roundtrip error: {wy}"

    print("[PASS] Coordinate Transforms (world <-> grid)")


def test_standoff_and_site_boundary():
    """Verify standoff approach corridor and 10x10m square search site logic."""
    gmap = AreaGridMap(site_width=10.0, site_length=10.0, standoff_distance=5.0)

    # Launch pad (standoff corridor)
    assert not gmap.is_inside_search_site(5.0, 0.2), "Launch pad should not be inside search site"
    assert not gmap.is_inside_search_site(5.0, 4.9), "Standoff 4.9m should be outside search site"

    # Search site (Y in [5.0, 15.0])
    assert gmap.is_inside_search_site(5.0, 5.0), "Site gate 5.0m should be inside search site"
    assert gmap.is_inside_search_site(1.0, 10.0), "Center of site should be inside search site"
    assert gmap.is_inside_search_site(5.0, 15.0), "North wall 15.0m should be inside search site"

    # Outside bounds
    assert not gmap.is_inside_search_site(5.0, 15.1), "Y > 15.0m should be outside search site"
    assert not gmap.is_inside_search_site(-0.1, 8.0), "X < 0.0m should be outside search site"
    assert not gmap.is_inside_search_site(10.1, 8.0), "X > 10.0m should be outside search site"

    print("[PASS] Standoff Corridor & 10x10m Search Site Boundary")


def test_monocular_projection_hfov():
    """Verify metric projection geometry and HFOV aspect ratio consistency."""
    projector = MonocularObstacleProjector(hfov_deg=65.0, assumed_camera_height=1.8, camera_pitch_deg=0.0)

    # Human detection in middle of frame
    det = {
        "bbox": [590, 260, 690, 560],  # Box height = 300px
        "normalized_center": (0.5, 0.5),
        "class_name": "person",
        "confidence": 0.92,
    }

    proj = projector.project_detection(
        det=det,
        frame_width=1280,
        frame_height=720,
        drone_x=5.0,
        drone_y=0.2,
        drone_heading_deg=0.0,
    )

    assert proj["is_target"] is True
    assert proj["distance"] > 1.0, f"Distance too small: {proj['distance']}"
    assert proj["distance"] < 8.0, f"Distance too large: {proj['distance']}"
    assert abs(proj["world_x"] - 5.0) < 0.2, f"Target off-center X: {proj['world_x']}"
    assert proj["world_y"] > 0.2, f"Target should be in front: {proj['world_y']}"

    print("[PASS] Monocular Projection HFOV Geometry")


def test_reactive_avoidance_sectors():
    """Verify 3-sector clearance, perimeter fences, and lateral hazard handling."""
    engine = ReactiveAvoidanceEngine(site_width=10.0, total_length=15.0, site_y_start=5.0)

    # 1. Critical proximity brake
    crit_obs = [{"distance": 0.6, "bearing_deg": 0.0}]
    adv = engine.evaluate_clearance(5.0, 6.0, 0.0, crit_obs)
    assert adv["status"] == "HALT", f"Expected HALT, got {adv['status']}"
    assert adv["urgency"] == "CRITICAL"
    assert adv["speed"] == 0.0

    # 2. Obstacle ahead -> steer right
    ahead_obs = [{"distance": 1.8, "bearing_deg": 0.0}]
    adv = engine.evaluate_clearance(5.0, 6.0, 0.0, ahead_obs)
    assert adv["status"] in ("AVOID_RIGHT", "AVOID_LEFT"), f"Unexpected: {adv['status']}"
    assert adv["speed"] > 0.0

    # 3. Lateral hazard on left flank -> nudge right
    left_obs = [{"distance": 1.0, "bearing_deg": 335.0}]  # -25 deg relative
    adv = engine.evaluate_clearance(5.0, 6.0, 0.0, left_obs)
    assert adv["status"] == "NUDGE_RIGHT", f"Expected NUDGE_RIGHT, got {adv['status']}"
    assert adv["steer_deg"] > 0

    # 4. Lateral hazard on right flank -> nudge left
    right_obs = [{"distance": 1.0, "bearing_deg": 25.0}]  # +25 deg relative
    adv = engine.evaluate_clearance(5.0, 6.0, 0.0, right_obs)
    assert adv["status"] == "NUDGE_LEFT", f"Expected NUDGE_LEFT, got {adv['status']}"
    assert adv["steer_deg"] < 0

    # 5. Perimeter boundaries
    adv_west = engine.evaluate_clearance(0.4, 6.0, 0.0, [])
    assert adv_west["status"] == "BOUNDARY"
    assert adv_west["steer_deg"] > 0  # Steer East

    adv_east = engine.evaluate_clearance(9.6, 6.0, 0.0, [])
    assert adv_east["status"] == "BOUNDARY"
    assert adv_east["steer_deg"] < 0  # Steer West

    adv_north = engine.evaluate_clearance(5.0, 14.6, 0.0, [])
    assert adv_north["status"] == "BOUNDARY"
    assert adv_north["steer_deg"] == 180.0  # Turn South

    print("[PASS] Reactive Avoidance 3-Sector Clearance")


def test_drone_motion_and_coverage():
    """Verify drone movement, ray-cast clearing, and site coverage percentage progression."""
    gmap = AreaGridMap(site_width=10.0, site_length=10.0, standoff_distance=5.0)

    # Initial state
    assert gmap.drone_x == 5.0
    assert gmap.drone_y == 0.2
    assert gmap.get_site_coverage_percentage() == 0.0

    # Drone advances forward along standoff corridor
    gmap.update_drone_pose(dx=0.0, dy=4.0, d_heading=0.0)
    assert gmap.drone_y == 4.2
    assert len(gmap.flight_path) > 1

    # Drone enters the 10x10m search site
    gmap.update_drone_pose(dx=0.0, dy=2.0, d_heading=0.0)
    assert gmap.drone_y == 6.2
    assert gmap.is_inside_search_site(gmap.drone_x, gmap.drone_y)

    # Sweep FOV clearance cone inside site
    gmap.mark_free_cone(fov_deg=65.0, max_range=3.8)
    cov = gmap.get_site_coverage_percentage()
    assert cov > 0.0, f"Coverage should be >0% after sweeping site, got {cov}%"

    print("[PASS] Drone Motion & Survey Coverage Progression")


def test_telemetry_json_serialization():
    """Verify structured telemetry JSON serialization safety."""
    detections = [
        {
            "bbox": [100, 150, 250, 400],
            "confidence": 0.88,
            "class_name": "person",
            "pixel_center": (175, 275),
            "normalized_center": (0.2734, 0.3819),
            "offset_from_center": (-465, -85),
            "projection": {
                "world_x": 4.2,
                "world_y": 3.8,
                "distance": 3.9,
                "bearing_deg": 348.5,
                "is_target": True,
            },
        }
    ]

    telemetry = DroneEdgeDetector.format_target_telemetry(detections, frame_width=1280, frame_height=720)
    encoded = json.dumps(telemetry, indent=2)
    decoded = json.loads(encoded)

    assert decoded["total_targets"] == 1
    assert decoded["targets"][0]["class"] == "person"
    assert decoded["targets"][0]["projection"]["world_x"] == 4.2

    print("[PASS] Telemetry JSON Serialization Safety")


if __name__ == "__main__":
    print("\n--- Running AeroVision Edge Verification Tests ---")
    test_coordinate_transforms()
    test_standoff_and_site_boundary()
    test_monocular_projection_hfov()
    test_reactive_avoidance_sectors()
    test_drone_motion_and_coverage()
    test_telemetry_json_serialization()
    print("--------------------------------------------------")
    print("All 6 test suites passed cleanly!\n")
