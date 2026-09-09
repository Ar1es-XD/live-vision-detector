"""
AeroVision SAR - Drone Edge Vision Pipeline
Real-time emulator window / stream ingestion with YOLO11 detection,
coordinate extraction, 10x10m area mapping, and reactive obstacle avoidance.
"""
import os
import cv2
import time
import json
import math
import argparse
import numpy as np

from detector import DroneEdgeDetector
from capture import create_capture_source, WindowStreamGrabber
from mapper import (
    AreaGridMap,
    MonocularObstacleProjector,
    ReactiveAvoidanceEngine,
    render_minimap_hud,
)


def draw_tactical_hud(
    frame: np.ndarray,
    detections: list,
    fps: float,
    source_name: str,
    sar_mode: bool,
    humans_only: bool,
    grid_map: AreaGridMap,
    advisory: dict,
    show_minimap: bool = True,
    auto_fly: bool = True,
) -> np.ndarray:
    """
    Renders high-contrast Tactical SAR HUD with bounding boxes, center coordinates,
    optical crosshairs, reactive avoidance guidance banner, and 10x10m radar minimap.
    """
    h, w = frame.shape[:2]
    canvas = frame.copy()
    cx_frame, cy_frame = w // 2, h // 2

    # 1. Optical Center Reticle (SAR Mode)
    if sar_mode:
        cv2.circle(canvas, (cx_frame, cy_frame), 30, (0, 255, 180), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx_frame - 45, cy_frame), (cx_frame - 35, cy_frame), (0, 255, 180), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx_frame + 35, cy_frame), (cx_frame + 45, cy_frame), (0, 255, 180), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx_frame, cy_frame - 45), (cx_frame, cy_frame - 35), (0, 255, 180), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx_frame, cy_frame + 35), (cx_frame, cy_frame + 45), (0, 255, 180), 1, cv2.LINE_AA)

        # Viewfinder 4-corner brackets
        b_len = 24
        m = 16
        cv2.line(canvas, (m, m + b_len), (m, m), (0, 255, 180), 2)
        cv2.line(canvas, (m, m), (m + b_len, m), (0, 255, 180), 2)
        cv2.line(canvas, (w - m - b_len, m), (w - m, m), (0, 255, 180), 2)
        cv2.line(canvas, (w - m, m), (w - m, m + b_len), (0, 255, 180), 2)
        cv2.line(canvas, (m, h - m - b_len), (m, h - m), (0, 255, 180), 2)
        cv2.line(canvas, (m, h - m), (m + b_len, h - m), (0, 255, 180), 2)
        cv2.line(canvas, (w - m - b_len, h - m), (w - m, h - m), (0, 255, 180), 2)
        cv2.line(canvas, (w - m, h - m), (w - m, h - m - b_len), (0, 255, 180), 2)

    # 2. Render Detections & Coordinate Tags
    victim_count = 0
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class_name"]
        conf = det["confidence"]
        cx, cy = det["pixel_center"]
        norm_x, norm_y = det["normalized_center"]
        proj = det.get("projection")
        is_person = cls.lower() == "person"

        if is_person:
            victim_count += 1

        # Color scheme
        if sar_mode and is_person:
            box_color = (0, 0, 255)      # Bright Red for SAR survivor
            tag_color = (30, 30, 200)
            text_color = (255, 255, 255)
        elif is_person:
            box_color = (0, 220, 100)    # Emerald for human
            tag_color = (20, 140, 60)
            text_color = (255, 255, 255)
        else:
            box_color = (255, 165, 0)    # Amber/Cyan for general object
            tag_color = (180, 110, 0)
            text_color = (255, 255, 255)

        # Smooth Bounding Box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

        # Corner Reticles
        c_size = min(12, (x2 - x1) // 4, (y2 - y1) // 4)
        if c_size > 3:
            cv2.line(canvas, (x1, y1), (x1 + c_size, y1), box_color, 3)
            cv2.line(canvas, (x1, y1), (x1, y1 + c_size), box_color, 3)
            cv2.line(canvas, (x2, y1), (x2 - c_size, y1), box_color, 3)
            cv2.line(canvas, (x2, y1), (x2, y1 + c_size), box_color, 3)
            cv2.line(canvas, (x1, y2), (x1 + c_size, y2), box_color, 3)
            cv2.line(canvas, (x1, y2), (x1, y2 - c_size), box_color, 3)
            cv2.line(canvas, (x2, y2), (x2 - c_size, y2), box_color, 3)
            cv2.line(canvas, (x2, y2), (x2, y2 - c_size), box_color, 3)

        # Target Centroid Reticle
        cv2.circle(canvas, (cx, cy), 4, box_color, -1)
        cv2.drawMarker(canvas, (cx, cy), box_color, cv2.MARKER_CROSS, 16, 1)

        # Vector line from center to target in SAR mode
        if sar_mode and is_person:
            cv2.line(canvas, (cx_frame, cy_frame), (cx, cy), (0, 0, 255), 1, cv2.LINE_AA)

        # Label Banner
        dist_str = f" ~{proj['distance']:.1f}m" if proj else ""
        if sar_mode and is_person:
            title_text = f"VICTIM #{victim_count} [{conf*100:.0f}%]{dist_str}"
        else:
            title_text = f"{cls.upper()} [{conf*100:.0f}%]{dist_str}"

        # Coordinate Tag
        coord_text = f"XY: [{norm_x:.3f}, {norm_y:.3f}] px: ({cx}, {cy})"
        if proj:
            coord_text += f" | 10m Pos: ({proj['world_x']:.1f}m, {proj['world_y']:.1f}m)"

        # Upper label badge
        label_y = max(24, y1 - 8)
        (tw1, th1), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, label_y - th1 - 4), (x1 + tw1 + 8, label_y + 2), tag_color, -1)
        cv2.putText(canvas, title_text, (x1 + 4, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        # Lower coordinate plaque
        (tw2, th2), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        coord_y = min(h - 8, y2 + th2 + 6)
        cv2.rectangle(canvas, (x1, coord_y - th2 - 4), (x1 + tw2 + 8, coord_y + 2), (20, 20, 24), -1)
        cv2.rectangle(canvas, (x1, coord_y - th2 - 4), (x1 + tw2 + 8, coord_y + 2), box_color, 1)
        cv2.putText(canvas, coord_text, (x1 + 4, coord_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)

    # 3. Top Frosted Telemetry Banner
    banner_h = 36
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (12, 16, 22), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    cv2.line(canvas, (0, banner_h), (w, banner_h), (50, 60, 75), 1)

    # Telemetry metrics
    mode_str = "SAR TACTICAL" if sar_mode else "SURVEILLANCE"
    flight_str = "AUTO-NAV" if auto_fly else "PAUSED"
    flight_color = (0, 255, 180) if auto_fly else (0, 180, 255)
    cov = grid_map.get_site_coverage_percentage()
    inside_tag = "INSIDE SITE" if grid_map.is_inside_search_site(grid_map.drone_x, grid_map.drone_y) else "APPROACH"
    stat_left = f"AEROVISION | {mode_str} [{flight_str}] | FPS: {fps:.1f} | SITE: {cov:.0f}% ({inside_tag})"
    stat_right = f"POS: ({grid_map.drone_x:.1f}m, {grid_map.drone_y:.1f}m) HDG: {int(grid_map.heading_deg)}°"

    cv2.putText(canvas, stat_left, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, flight_color, 1, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(stat_right, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(canvas, stat_right, (w - rw - 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    # 4. Reactive Obstacle Avoidance Guidance Banner
    adv_msg = advisory.get("message", "SYSTEM STANDBY")
    adv_urgency = advisory.get("urgency", "LOW")

    if adv_urgency == "CRITICAL":
        adv_bg = (0, 0, 180)     # Red
        adv_text_col = (255, 255, 255)
    elif adv_urgency == "MEDIUM" or adv_urgency == "HIGH":
        adv_bg = (0, 120, 220)   # Orange / Amber
        adv_text_col = (255, 255, 255)
    else:
        adv_bg = (20, 80, 45)    # Emerald
        adv_text_col = (180, 255, 210)

    adv_banner_h = 24
    adv_y0 = banner_h + 6
    adv_text = f"AVOIDANCE GUIDANCE: {adv_msg}"
    (atw, ath), _ = cv2.getTextSize(adv_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(canvas, (12, adv_y0), (12 + atw + 16, adv_y0 + adv_banner_h), adv_bg, -1)
    cv2.rectangle(canvas, (12, adv_y0), (12 + atw + 16, adv_y0 + adv_banner_h), (255, 255, 255), 1)
    cv2.putText(canvas, adv_text, (20, adv_y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, adv_text_col, 1, cv2.LINE_AA)

    # 5. 10m x 10m Tactical Radar Minimap (Picture-in-Picture)
    if show_minimap and w >= 640 and h >= 360:
        minimap_img = render_minimap_hud(grid_map, advisory)
        mm_h, mm_w = minimap_img.shape[:2]
        pad = 12
        x_mm = w - mm_w - pad
        y_mm = banner_h + pad

        # Drop shadow / backdrop border
        cv2.rectangle(canvas, (x_mm - 2, y_mm - 2), (x_mm + mm_w + 2, y_mm + mm_h + 2), (0, 255, 180), 1)
        canvas[y_mm : y_mm + mm_h, x_mm : x_mm + mm_w] = minimap_img

    # 6. Bottom Quick-Key Bar
    bot_h = 24
    bot_overlay = canvas.copy()
    cv2.rectangle(bot_overlay, (0, h - bot_h), (w, h), (12, 16, 22), -1)
    cv2.addWeighted(bot_overlay, 0.85, canvas, 0.15, 0, canvas)
    guide_text = "[Q] Quit | [A] Auto-Nav | [IJKL] Nudge | [S] SAR | [C] Humans | [M] Map | [R] Reset | [W] Calibrate | [SPACE] Snap"
    cv2.putText(canvas, guide_text, (12, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 170, 185), 1, cv2.LINE_AA)

    return canvas


def run():
    parser = argparse.ArgumentParser(
        description="AeroVision Drone Edge Live Video Detector (10x10m Mapping & Obstacle Avoidance)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="bluestacks",
        help="Input source: 'bluestacks', window name, 'x,y,w,h' coordinates, stream URL (rtsp/udp/http), or camera index (0, 1). Default: 'bluestacks'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model path or identifier. Default: 'yolo11n.pt'",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold. Default: 0.35",
    )
    parser.add_argument(
        "--sar",
        action="store_true",
        help="Enable SAR tactical survivor lock on launch",
    )
    parser.add_argument(
        "--humans",
        action="store_true",
        help="Filter to humans only on launch",
    )
    parser.add_argument(
        "--map-size",
        type=float,
        default=10.0,
        help="Square search site dimensions in meters (e.g. 10.0 for 10x10m area). Default: 10.0",
    )
    parser.add_argument(
        "--standoff",
        type=float,
        default=5.0,
        help="Launch standoff distance from the search site in meters. Default: 5.0",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Hide the 10x10m radar minimap PiP by default",
    )
    parser.add_argument(
        "--no-auto-fly",
        action="store_true",
        help="Start with autonomous navigation paused instead of active on launch",
    )
    parser.add_argument(
        "--camera-pitch",
        type=float,
        default=0.0,
        help="Camera downward tilt angle in degrees. Default: 0.0 (level forward)",
    )
    args = parser.parse_args()

    os.makedirs("captures", exist_ok=True)

    auto_fly = not args.no_auto_fly

    print("=" * 65)
    print("  AEROVISION SAR - 10x10m AREA MAPPING & AVOIDANCE PIPELINE")
    print("=" * 65)
    print(f"  Target Source   : {args.source}")
    print(f"  YOLO Model      : {args.model}")
    print(f"  Search Site     : {args.map_size}m x {args.map_size}m Square (100m²)")
    print(f"  Launch Standoff : {args.standoff}m South (Middle X={args.map_size/2.0:.1f}m, Y=0.2m)")
    print(f"  Auto Navigation : {'ACTIVE (ON LAUNCH)' if auto_fly else 'PAUSED'}")
    print(f"  Camera Pitch    : {args.camera_pitch}°")
    print(f"  Confidence      : {args.conf}")
    print(f"  SAR Mode        : {'ACTIVE' if args.sar else 'STANDBY'}")
    print(f"  Humans Only     : {'ENABLED' if args.humans else 'DISABLED'}")
    print(f"  Minimap Radar   : {'ENABLED' if not args.no_map else 'HIDDEN'}")
    print("=" * 65 + "\n")

    # Initialize YOLO Edge Detector
    detector = DroneEdgeDetector(model_name=args.model, conf_threshold=args.conf)
    if args.humans:
        detector.set_target_classes(["person"])

    # Initialize 10x10m Area Mapping & Avoidance Engine with 5m Standoff
    grid_map = AreaGridMap(
        site_width=args.map_size,
        site_length=args.map_size,
        standoff_distance=args.standoff,
    )
    projector = MonocularObstacleProjector(hfov_deg=65.0, camera_pitch_deg=args.camera_pitch)
    avoidance_engine = ReactiveAvoidanceEngine(
        site_width=args.map_size,
        total_length=args.map_size + args.standoff,
        site_y_start=args.standoff,
    )
    show_minimap = not args.no_map

    # Initialize Capture Source
    source_handler, source_name = create_capture_source(args.source)

    sar_mode = args.sar
    humans_only = args.humans

    prev_time = time.time()
    fps = 0.0
    frame_count = 0

    window_title = "AeroVision SAR - Drone Edge Viewfinder & 10x10m Mapper"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = source_handler.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            h_f, w_f = frame.shape[:2]

            # 1. Update Free Space in Drone's Field of View
            grid_map.mark_free_cone(fov_deg=65.0, max_range=3.8)

            # 2. YOLO11 Inference & Coordinate Extraction
            detections = detector.infer(frame)

            # 3. Project Obstacles into 10x10m Spatial Map
            projected_obstacles = []
            for det in detections:
                proj = projector.project_detection(
                    det=det,
                    frame_width=w_f,
                    frame_height=h_f,
                    drone_x=grid_map.drone_x,
                    drone_y=grid_map.drone_y,
                    drone_heading_deg=grid_map.heading_deg,
                )
                det["projection"] = proj
                projected_obstacles.append(proj)

                # Insert into occupancy grid map
                grid_map.add_obstacle(
                    x=proj["world_x"],
                    y=proj["world_y"],
                    radius=0.35,
                    is_target=proj["is_target"],
                )

            # 4. Evaluate Reactive Avoidance Guidance
            advisory = avoidance_engine.evaluate_clearance(
                drone_x=grid_map.drone_x,
                drone_y=grid_map.drone_y,
                drone_heading=grid_map.heading_deg,
                projected_obstacles=projected_obstacles,
            )

            # 5. Simulated Flight Navigation (Auto-Fly active or manual)
            if auto_fly:
                speed = float(advisory.get("speed", 0.0))
                steer = float(advisory.get("steer_deg", 0.0))
                step_dt = min(0.1, max(0.01, dt))
                h_rad = math.radians(grid_map.heading_deg)
                dx = speed * math.sin(h_rad) * step_dt
                dy = speed * math.cos(h_rad) * step_dt
                d_heading = steer * step_dt * 1.5
                grid_map.update_drone_pose(dx=dx, dy=dy, d_heading=d_heading)

            # 6. Render Tactical HUD with Minimap & Guidance Banner
            annotated_frame = draw_tactical_hud(
                frame=frame,
                detections=detections,
                fps=fps,
                source_name=source_name,
                sar_mode=sar_mode,
                humans_only=humans_only,
                grid_map=grid_map,
                advisory=advisory,
                show_minimap=show_minimap,
                auto_fly=auto_fly,
            )

            cv2.imshow(window_title, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[STOP] Exit requested by operator.")
                break
            elif key == ord("a"):
                auto_fly = not auto_fly
                print(f"[NAV] Autonomous Navigation: {'ACTIVE' if auto_fly else 'PAUSED'}")
            elif key in (ord("i"), 82):  # 'i' or Up arrow: Forward nudge
                h_rad = math.radians(grid_map.heading_deg)
                grid_map.update_drone_pose(dx=0.35 * math.sin(h_rad), dy=0.35 * math.cos(h_rad))
            elif key in (ord("k"), 84):  # 'k' or Down arrow: Back nudge
                h_rad = math.radians(grid_map.heading_deg)
                grid_map.update_drone_pose(dx=-0.35 * math.sin(h_rad), dy=-0.35 * math.cos(h_rad))
            elif key in (ord("j"), 81):  # 'j' or Left arrow: Yaw left
                grid_map.update_drone_pose(d_heading=-15.0)
            elif key in (ord("l"), 83):  # 'l' or Right arrow: Yaw right
                grid_map.update_drone_pose(d_heading=15.0)
            elif key == ord("s"):
                sar_mode = not sar_mode
                print(f"[MODE] SAR Tactical Mode: {'ACTIVE' if sar_mode else 'STANDBY'}")
            elif key == ord("c"):
                humans_only = not humans_only
                detector.set_target_classes(["person"] if humans_only else None)
                print(f"[FILTER] Target Filter: {'HUMANS ONLY' if humans_only else 'ALL 80 CLASSES'}")
            elif key == ord("m"):
                show_minimap = not show_minimap
                print(f"[HUD] Minimap Radar: {'VISIBLE' if show_minimap else 'HIDDEN'}")
            elif key == ord("r"):
                grid_map.reset()
                print("[MAP] 10x10m Spatial Map Reset.")
            elif key == ord("w"):
                if isinstance(source_handler, WindowStreamGrabber):
                    cv2.destroyWindow(window_title)
                    success = source_handler.select_roi_interactive()
                    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                    if success:
                        source_name = source_handler.window_title
                else:
                    print("[INFO] Calibrate region is only available for window/screen capture mode.")
            elif key == 32:  # SPACE bar
                timestamp = int(time.time())
                img_path = f"captures/incident_{timestamp}.jpg"
                json_path = f"captures/incident_{timestamp}.json"

                cv2.imwrite(img_path, annotated_frame)

                # Export structured telemetry JSON with target coordinates and 10x10m map state
                telemetry_data = detector.format_target_telemetry(detections, w_f, h_f)
                telemetry_data["metadata"] = {
                    "source": source_name,
                    "sar_mode": sar_mode,
                    "humans_only": humans_only,
                    "auto_fly": auto_fly,
                    "fps": round(fps, 1),
                    "search_site_size": {"width": grid_map.site_width, "length": grid_map.site_length},
                    "standoff_distance": grid_map.standoff_distance,
                    "drone_position": {"x": round(grid_map.drone_x, 2), "y": round(grid_map.drone_y, 2)},
                    "drone_heading": round(grid_map.heading_deg, 1),
                    "site_coverage_percent": grid_map.get_site_coverage_percentage(),
                    "is_inside_site": grid_map.is_inside_search_site(grid_map.drone_x, grid_map.drone_y),
                    "avoidance_advisory": advisory,
                }

                with open(json_path, "w") as f:
                    json.dump(telemetry_data, f, indent=2)

                print(f"\n[CAPTURED] Saved Incident Snapshot:")
                print(f"  Image     : {img_path}")
                print(f"  Telemetry : {json_path}")
                print(f"  Targets   : {len(detections)}")
                print(f"  Advisory  : {advisory['message']}\n")

    finally:
        source_handler.release()
        cv2.destroyAllWindows()
        print("[CLEANUP] Pipeline resources safely released.")


if __name__ == "__main__":
    run()
