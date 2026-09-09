"""
10m x 10m Search Site Spatial Grid Mapping & Reactive Obstacle Avoidance
Features:
- 5m Standoff Launch Positioning (Drone starts at center X=5m, Y=0m, facing North)
- 10m x 10m Square Search Site (from Y=5m to Y=15m)
- Monocular Obstacle Projection & Safety Inflation
- 3-Sector Reactive Avoidance Guidance & Boundary Protection
- Tactical Minimap PiP with Standoff Corridor and Search Site Demarcation
"""
import math
import numpy as np
import cv2

# Grid Cell States
CELL_UNKNOWN = 0   # Unexplored
CELL_FREE = 1      # Surveyed free space
CELL_OBSTACLE = 2  # Detected hazard/obstacle
CELL_TARGET = 3    # Detected person/victim (SAR target)
CELL_TRAIL = 4     # Flight path breadcrumbs


class AreaGridMap:
    """
    Spatial Occupancy Grid with 5m Standoff Launch & 10m x 10m Square Search Site.
    Coordinate System:
    - X-axis (West to East): 0.0m to 10.0m (Width = 10m)
    - Y-axis (South to North): 0.0m to 15.0m (Total Length = 15m)
      - Y in [0.0, 5.0m]: 5m Standoff Approach Corridor
      - Y in [5.0, 15.0m]: 10m x 10m Square Search Site
    - Takeoff / Start Pose: (X=5.0m, Y=0.2m, Heading=0.0 deg North)
    """

    def __init__(
        self,
        site_width: float = 10.0,
        site_length: float = 10.0,
        standoff_distance: float = 5.0,
        resolution: float = 0.2,
    ):
        self.site_width = site_width                      # 10.0m
        self.site_length = site_length                    # 10.0m (forms 10x10m square)
        self.standoff_distance = standoff_distance        # 5.0m away from site
        self.resolution = resolution                      # 0.2m

        self.total_width = site_width                     # 10.0m
        self.total_length = site_length + standoff_distance  # 15.0m

        self.site_y_start = standoff_distance             # 5.0m (site entrance)
        self.site_y_end = self.total_length               # 15.0m (site north wall)

        self.cols = int(self.total_width / resolution)    # 50 cols
        self.rows = int(self.total_length / resolution)   # 75 rows

        # Occupancy grid matrix: 0=Unknown, 1=Free, 2=Obstacle, 3=Target, 4=Trail
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # Drone start pose: middle of left/right extremes (5.0m), 5m standoff (0.2m South)
        self.drone_x = self.site_width / 2.0              # 5.0m
        self.drone_y = 0.2                                # 0.2m (launch pad)
        self.heading_deg = 0.0                            # Facing North towards site entrance
        self.flight_path = [(self.drone_x, self.drone_y)]

    def reset(self):
        self.grid.fill(CELL_UNKNOWN)
        self.drone_x = self.site_width / 2.0
        self.drone_y = 0.2
        self.heading_deg = 0.0
        self.flight_path = [(self.drone_x, self.drone_y)]

    def world_to_grid(self, x: float, y: float):
        """Converts world coordinates (meters) to grid indices (col, row)."""
        col = int(x / self.resolution)
        # Row 0 is North (Y = total_length = 15m), Row (rows - 1) is South (Y = 0m)
        row = self.rows - 1 - int(y / self.resolution)
        col = max(0, min(self.cols - 1, col))
        row = max(0, min(self.rows - 1, row))
        return col, row

    def grid_to_world(self, col: int, row: int):
        """Converts grid indices to world coordinates (meters)."""
        x = (col + 0.5) * self.resolution
        y = (self.rows - 1 - row + 0.5) * self.resolution
        return x, y

    def is_inside(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.total_width and 0.0 <= y <= self.total_length

    def is_inside_search_site(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.site_width and self.site_y_start <= y <= self.site_y_end

    def update_drone_pose(self, dx: float = 0.0, dy: float = 0.0, d_heading: float = 0.0):
        """Update drone simulated position and log flight path."""
        new_x = max(0.4, min(self.total_width - 0.4, self.drone_x + dx))
        new_y = max(0.2, min(self.total_length - 0.4, self.drone_y + dy))
        self.drone_x = new_x
        self.drone_y = new_y
        self.heading_deg = (self.heading_deg + d_heading) % 360.0

        c, r = self.world_to_grid(self.drone_x, self.drone_y)
        if self.grid[r, c] not in (CELL_OBSTACLE, CELL_TARGET):
            self.grid[r, c] = CELL_TRAIL

        last_x, last_y = self.flight_path[-1]
        if math.hypot(self.drone_x - last_x, self.drone_y - last_y) > 0.25:
            self.flight_path.append((self.drone_x, self.drone_y))

    def mark_free_cone(self, fov_deg: float = 65.0, max_range: float = 3.8):
        """
        Ray-cast clearance cone along the drone's heading, marking swept cells as free.
        """
        angles = np.linspace(-fov_deg / 2.0, fov_deg / 2.0, 23)
        for rel_angle in angles:
            ray_angle = math.radians(self.heading_deg + rel_angle)
            for dist in np.arange(0.3, max_range, self.resolution):
                rx = self.drone_x + dist * math.sin(ray_angle)
                ry = self.drone_y + dist * math.cos(ray_angle)
                if not self.is_inside(rx, ry):
                    break
                c, r = self.world_to_grid(rx, ry)
                if self.grid[r, c] in (CELL_OBSTACLE, CELL_TARGET):
                    break
                self.grid[r, c] = CELL_FREE

    def add_obstacle(self, x: float, y: float, radius: float = 0.35, is_target: bool = False):
        """
        Inflates and registers an obstacle or target in the occupancy grid.
        """
        if not self.is_inside(x, y):
            return

        c_center, r_center = self.world_to_grid(x, y)
        cell_radius = max(1, int(radius / self.resolution))
        state = CELL_TARGET if is_target else CELL_OBSTACLE

        for dr in range(-cell_radius, cell_radius + 1):
            for dc in range(-cell_radius, cell_radius + 1):
                if dr * dr + dc * dc <= cell_radius * cell_radius:
                    rr = r_center + dr
                    cc = c_center + dc
                    if 0 <= rr < self.rows and 0 <= cc < self.cols:
                        self.grid[rr, cc] = state

    def get_site_coverage_percentage(self) -> float:
        """
        Calculates survey coverage specifically over the 10m x 10m square search site
        (Y from 5.0m to 15.0m).
        """
        site_row_top = 0
        site_row_bottom = self.rows - int(self.site_y_start / self.resolution)
        site_slice = self.grid[site_row_top:site_row_bottom, :]

        total_cells = site_slice.size
        if total_cells == 0:
            return 0.0
        explored = np.count_nonzero(site_slice != CELL_UNKNOWN)
        return round(float(explored / total_cells) * 100.0, 1)

    def get_overall_coverage_percentage(self) -> float:
        explored = np.count_nonzero(self.grid != CELL_UNKNOWN)
        return round(float(explored / (self.rows * self.cols)) * 100.0, 1)


class MonocularObstacleProjector:
    """
    Estimates 3D ground coordinates of detected objects from 2D bounding boxes
    using monocular pinhole ray-casting geometry.
    """

    def __init__(
        self,
        hfov_deg: float = 65.0,
        assumed_camera_height: float = 1.8,
        camera_pitch_deg: float = 0.0,
    ):
        self.hfov_rad = math.radians(hfov_deg)
        self.camera_height = assumed_camera_height
        self.pitch_rad = math.radians(camera_pitch_deg)

    def project_detection(
        self,
        det: dict,
        frame_width: int,
        frame_height: int,
        drone_x: float,
        drone_y: float,
        drone_heading_deg: float,
    ) -> dict:
        """
        Projects a detection bounding box into 10x15m world coordinates.
        Returns: { 'world_x', 'world_y', 'distance', 'bearing', 'is_target' }
        """
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        norm_x, norm_y = det["normalized_center"]
        cls_name = det["class_name"].lower()
        is_target = (cls_name == "person")

        # 1. Bearing offset relative to camera optical axis
        bearing_offset_rad = (norm_x - 0.5) * self.hfov_rad
        world_bearing_rad = math.radians(drone_heading_deg) + bearing_offset_rad

        # 2. Monocular Distance Estimation
        # Fix: Horizontal FOV corresponds to frame_width
        box_h = max(1, y2 - y1)
        object_real_height = 1.7 if is_target else 1.0
        focal_px = (frame_width / 2.0) / math.tan(self.hfov_rad / 2.0)
        dist_scale = (focal_px * object_real_height) / max(10, box_h)

        # Ground intersection calculation accounting for camera pitch
        cy_offset = y2 - (frame_height / 2.0)
        alpha = math.atan(cy_offset / max(1.0, focal_px))
        total_elevation = alpha + self.pitch_rad

        if total_elevation > 0.05:
            dist_ground = self.camera_height / math.tan(total_elevation)
            raw_distance = 0.6 * dist_scale + 0.4 * dist_ground
        else:
            raw_distance = dist_scale

        distance = float(np.clip(raw_distance, 0.5, 9.5))

        # 3. Compute World Coordinates
        world_x = drone_x + distance * math.sin(world_bearing_rad)
        world_y = drone_y + distance * math.cos(world_bearing_rad)

        return {
            "world_x": round(world_x, 2),
            "world_y": round(world_y, 2),
            "distance": round(distance, 2),
            "bearing_deg": round(math.degrees(world_bearing_rad) % 360.0, 1),
            "is_target": is_target,
            "class_name": det["class_name"],
            "confidence": det["confidence"],
        }


class ReactiveAvoidanceEngine:
    """
    Evaluates obstacles in the drone's path and enforces search site boundary limits.
    """

    def __init__(
        self,
        site_width: float = 10.0,
        total_length: float = 15.0,
        site_y_start: float = 5.0,
        warning_distance: float = 2.2,
    ):
        self.site_width = site_width
        self.total_length = total_length
        self.site_y_start = site_y_start
        self.warning_dist = warning_distance

    def evaluate_clearance(
        self,
        drone_x: float,
        drone_y: float,
        drone_heading: float,
        projected_obstacles: list,
    ) -> dict:
        """
        Evaluates 3 forward sectors (Left, Center, Right) and boundaries.
        """
        # 1. Perimeter Boundary Guard
        margin = 0.8
        if drone_x < margin:
            return {
                "status": "BOUNDARY",
                "message": "WEST PERIMETER REACHED -> STEER EAST",
                "steer_deg": 40.0,
                "speed": 0.4,
                "urgency": "HIGH",
            }
        if drone_x > self.site_width - margin:
            return {
                "status": "BOUNDARY",
                "message": "EAST PERIMETER REACHED -> STEER WEST",
                "steer_deg": -40.0,
                "speed": 0.4,
                "urgency": "HIGH",
            }
        if drone_y > self.total_length - margin:
            return {
                "status": "BOUNDARY",
                "message": "NORTH PERIMETER REACHED -> TURN SOUTH",
                "steer_deg": 180.0,
                "speed": 0.3,
                "urgency": "HIGH",
            }

        # 2. Obstacle Proximity in 3 Forward Sectors
        left_clearance = 5.0
        center_clearance = 5.0
        right_clearance = 5.0

        for obs in projected_obstacles:
            dist = obs["distance"]
            rel_bearing = (obs["bearing_deg"] - drone_heading + 180) % 360 - 180

            if -12.0 <= rel_bearing <= 12.0:
                center_clearance = min(center_clearance, dist)
            elif -40.0 <= rel_bearing < -12.0:
                left_clearance = min(left_clearance, dist)
            elif 12.0 < rel_bearing <= 40.0:
                right_clearance = min(right_clearance, dist)

        if center_clearance < 0.9:
            steer = 35.0 if left_clearance < right_clearance else -35.0
            return {
                "status": "HALT",
                "message": f"HAZARD PROXIMITY ({center_clearance:.1f}m) -> BRAKE & HOVER",
                "steer_deg": steer,
                "speed": 0.0,
                "urgency": "CRITICAL",
            }

        if center_clearance < self.warning_dist:
            if right_clearance >= left_clearance:
                return {
                    "status": "AVOID_RIGHT",
                    "message": f"OBSTACLE AHEAD ({center_clearance:.1f}m) -> STEER RIGHT +30°",
                    "steer_deg": 30.0,
                    "speed": 0.5,
                    "urgency": "MEDIUM",
                }
            else:
                return {
                    "status": "AVOID_LEFT",
                    "message": f"OBSTACLE AHEAD ({center_clearance:.1f}m) -> STEER LEFT -30°",
                    "steer_deg": -30.0,
                    "speed": 0.5,
                    "urgency": "MEDIUM",
                }

        # Check lateral flanks for close hazards
        if left_clearance < 1.3:
            return {
                "status": "NUDGE_RIGHT",
                "message": f"HAZARD LEFT FLANK ({left_clearance:.1f}m) -> STEER RIGHT +20°",
                "steer_deg": 20.0,
                "speed": 0.6,
                "urgency": "LOW",
            }
        elif right_clearance < 1.3:
            return {
                "status": "NUDGE_LEFT",
                "message": f"HAZARD RIGHT FLANK ({right_clearance:.1f}m) -> STEER LEFT -20°",
                "steer_deg": -20.0,
                "speed": 0.6,
                "urgency": "LOW",
            }

        # Standoff Approach phase vs Search Site phase
        if drone_y < self.site_y_start:
            dist_to_gate = self.site_y_start - drone_y
            return {
                "status": "APPROACH",
                "message": f"APPROACHING 10x10m SITE ({dist_to_gate:.1f}m TO ENTRANCE) -> FORWARD",
                "steer_deg": 0.0,
                "speed": 0.8,
                "urgency": "LOW",
            }

        return {
            "status": "CLEAR",
            "message": f"SITE CLEAR ({center_clearance:.1f}m) -> FORWARD SURVEY",
            "steer_deg": 0.0,
            "speed": 0.8,
            "urgency": "LOW",
        }


def render_minimap_hud(grid_map: AreaGridMap, avoidance_advisory: dict) -> np.ndarray:
    """
    Renders a tactical radar minimap showing:
    - 5m Standoff Launch Pad (center X=5m, Y=0m)
    - 10m x 10m Square Search Site (Y=5m to Y=15m)
    - Surveyed space, detected obstacles, flight trail, and avoidance vector.
    """
    map_w = 230
    map_h = 320
    hud = np.zeros((map_h, map_w, 3), dtype=np.uint8)
    hud[:] = (12, 16, 22)  # Dark frosted tactical background

    margin_x = 22
    margin_y = 20
    arena_w = map_w - 2 * margin_x
    scale = arena_w / grid_map.total_width  # Pixels per meter (~18.6 px/m)
    arena_h = int(grid_map.total_length * scale)

    # 1. Staging Corridor (0m to 5m) Background
    y_launch_px = map_h - margin_y
    y_gate_px = int(map_h - margin_y - grid_map.site_y_start * scale)
    y_north_px = int(map_h - margin_y - grid_map.total_length * scale)

    # Staging zone fill
    cv2.rectangle(
        hud,
        (margin_x, y_gate_px),
        (map_w - margin_x, y_launch_px),
        (18, 24, 32),
        -1,
    )

    # 2. 10m x 10m Square Search Site Area (5m to 15m)
    cv2.rectangle(
        hud,
        (margin_x, y_north_px),
        (map_w - margin_x, y_gate_px),
        (16, 28, 28),
        -1,
    )
    # Search site boundary frame (illuminated emerald)
    cv2.rectangle(
        hud,
        (margin_x, y_north_px),
        (map_w - margin_x, y_gate_px),
        (0, 200, 140),
        1,
    )

    # Dashed Entrance Gate Line at Y = 5.0m
    for gx in range(margin_x, map_w - margin_x, 8):
        cv2.line(hud, (gx, y_gate_px), (min(gx + 4, map_w - margin_x), y_gate_px), (0, 255, 255), 1)

    # 2m Grid Line Subdivisions
    for m in range(2, int(grid_map.total_width), 2):
        px = int(margin_x + m * scale)
        cv2.line(hud, (px, y_north_px), (px, y_launch_px), (28, 36, 46), 1)

    for m in range(2, int(grid_map.total_length), 2):
        py = int(map_h - margin_y - m * scale)
        cv2.line(hud, (margin_x, py), (map_w - margin_x, py), (28, 36, 46), 1)

    # 3. Draw Occupancy Grid Cells
    cell_w_px = arena_w / grid_map.cols
    cell_h_px = arena_h / grid_map.rows
    for r in range(grid_map.rows):
        for c in range(grid_map.cols):
            state = grid_map.grid[r, c]
            if state == CELL_UNKNOWN:
                continue

            x0 = int(margin_x + c * cell_w_px)
            y0 = int(y_north_px + r * cell_h_px)
            x1 = int(x0 + cell_w_px + 1)
            y1 = int(y0 + cell_h_px + 1)

            if state == CELL_FREE:
                cv2.rectangle(hud, (x0, y0), (x1, y1), (25, 65, 45), -1)  # Muted green
            elif state == CELL_OBSTACLE:
                cv2.rectangle(hud, (x0, y0), (x1, y1), (0, 165, 255), -1)  # Amber
            elif state == CELL_TARGET:
                cv2.rectangle(hud, (x0, y0), (x1, y1), (0, 0, 255), -1)    # Bright red
            elif state == CELL_TRAIL:
                cv2.rectangle(hud, (x0, y0), (x1, y1), (60, 95, 80), -1)

    # 4. Flight Breadcrumb Trail
    if len(grid_map.flight_path) > 1:
        points = []
        for fx, fy in grid_map.flight_path:
            px = int(margin_x + fx * scale)
            py = int(map_h - margin_y - fy * scale)
            points.append((px, py))
        for i in range(1, len(points)):
            cv2.line(hud, points[i - 1], points[i], (80, 160, 120), 1, cv2.LINE_AA)

    # 5. Takeoff / Launch Pad (X=5.0m, Y=0.0m)
    pad_x = int(margin_x + 5.0 * scale)
    pad_y = int(map_h - margin_y - 0.2 * scale)
    cv2.circle(hud, (pad_x, pad_y), 6, (0, 220, 255), 1)
    cv2.drawMarker(hud, (pad_x, pad_y), (0, 220, 255), cv2.MARKER_CROSS, 8, 1)

    # 6. Drone Position & Heading Cone
    dx_px = int(margin_x + grid_map.drone_x * scale)
    dy_px = int(map_h - margin_y - grid_map.drone_y * scale)

    # Radar Beam Cone
    beam_len = 32
    h_rad = math.radians(grid_map.heading_deg)
    fov_half = math.radians(30)
    p1 = (
        int(dx_px + beam_len * math.sin(h_rad - fov_half)),
        int(dy_px - beam_len * math.cos(h_rad - fov_half)),
    )
    p2 = (
        int(dx_px + beam_len * math.sin(h_rad + fov_half)),
        int(dy_px - beam_len * math.cos(h_rad + fov_half)),
    )
    beam_overlay = hud.copy()
    cv2.fillPoly(beam_overlay, [np.array([(dx_px, dy_px), p1, p2])], (40, 160, 100))
    cv2.addWeighted(beam_overlay, 0.25, hud, 0.75, 0, hud)

    # Drone Triangle Icon
    arrow_len = 8
    tip = (int(dx_px + arrow_len * math.sin(h_rad)), int(dy_px - arrow_len * math.cos(h_rad)))
    left_pt = (
        int(dx_px + arrow_len * 0.7 * math.sin(h_rad + 2.5)),
        int(dy_px - arrow_len * 0.7 * math.cos(h_rad + 2.5)),
    )
    right_pt = (
        int(dx_px + arrow_len * 0.7 * math.sin(h_rad - 2.5)),
        int(dy_px - arrow_len * 0.7 * math.cos(h_rad - 2.5)),
    )
    cv2.fillPoly(hud, [np.array([tip, left_pt, right_pt])], (0, 255, 180))
    cv2.circle(hud, (dx_px, dy_px), 2, (255, 255, 255), -1)

    # Avoidance Vector Arrow
    steer = avoidance_advisory.get("steer_deg", 0.0)
    if steer != 0.0:
        steer_rad = math.radians(grid_map.heading_deg + steer)
        steer_len = 22
        arrow_tip = (
            int(dx_px + steer_len * math.sin(steer_rad)),
            int(dy_px - steer_len * math.cos(steer_rad)),
        )
        arrow_color = (0, 140, 255) if steer > 0 else (255, 100, 0)
        cv2.arrowedLine(hud, (dx_px, dy_px), arrow_tip, arrow_color, 2, tipLength=0.35)

    # 7. Labels & Scale Annotations
    cv2.putText(hud, "10x10m SEARCH SITE", (margin_x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 180), 1, cv2.LINE_AA)
    site_cov = grid_map.get_site_coverage_percentage()
    cv2.putText(hud, f"{site_cov:.0f}% COVERED", (map_w - margin_x - 66, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 190, 200), 1, cv2.LINE_AA)

    # Boundary text markers
    cv2.putText(hud, "NORTH (15m)", (margin_x + 2, y_north_px + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 140), 1)
    cv2.putText(hud, "SITE GATE (5m)", (margin_x + 2, y_gate_px - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)
    cv2.putText(hud, "STANDOFF (5m)", (margin_x + 2, y_gate_px + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (140, 150, 160), 1)
    cv2.putText(hud, "LAUNCH (0m)", (margin_x + 2, y_launch_px + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 220, 255), 1)
    cv2.putText(hud, "10m", (map_w - margin_x - 16, y_launch_px + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140, 150, 160), 1)

    return hud
