"""
YOLO11 Edge Inference Engine for Physical Drone Deployment
Supports Apple Silicon GPU (MPS), NVIDIA CUDA, and CPU.
"""
import time

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    YOLO = None
    HAS_YOLO = False


class DroneEdgeDetector:
    def __init__(self, model_name: str = "yolo11n.pt", conf_threshold: float = 0.35):
        if not HAS_TORCH or not HAS_YOLO:
            raise RuntimeError(
                "Packages 'torch' and 'ultralytics' are required for DroneEdgeDetector. "
                "Install with: pip install -r requirements.txt"
            )

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        print(f"[EDGE AI] Loading {model_name} on device: {self.device.upper()}")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.target_classes = None

    def set_target_classes(self, class_names: list[str] | None):
        if class_names is None:
            self.target_classes = None
            return
        name_to_id = {v: k for k, v in self.model.names.items()}
        self.target_classes = [name_to_id[c] for c in class_names if c in name_to_id]

    def infer(self, frame):
        h, w = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.conf_threshold,
            classes=self.target_classes,
            verbose=False
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Calculate target center & normalized coordinates
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                box_w = int(x2 - x1)
                box_h = int(y2 - y1)
                norm_x = round(float(cx / max(1, w)), 4)
                norm_y = round(float(cy / max(1, h)), 4)
                offset_x = cx - (w // 2)
                offset_y = cy - (h // 2)

                detections.append({
                    "bbox": xyxy,
                    "confidence": conf,
                    "class_name": self.model.names[cls_id],
                    "pixel_center": (cx, cy),
                    "normalized_center": (norm_x, norm_y),
                    "width": box_w,
                    "height": box_h,
                    "offset_from_center": (offset_x, offset_y),
                })
        return detections

    @staticmethod
    def format_target_telemetry(detections: list, frame_width: int, frame_height: int) -> dict:
        """
        Format detections into structured SAR / Drone telemetry payload.
        Ensures all types are native Python primitives for safe JSON serialization.
        """
        targets = []
        for d in detections:
            target_entry = {
                "class": str(d["class_name"]),
                "confidence": round(float(d["confidence"]), 3),
                "bbox": [int(v) for v in d["bbox"]],
                "center": {
                    "pixel": [int(d["pixel_center"][0]), int(d["pixel_center"][1])],
                    "normalized": [float(d["normalized_center"][0]), float(d["normalized_center"][1])],
                },
                "offset": [int(d["offset_from_center"][0]), int(d["offset_from_center"][1])],
            }
            if "projection" in d and d["projection"]:
                proj = d["projection"]
                target_entry["projection"] = {
                    "world_x": float(proj["world_x"]),
                    "world_y": float(proj["world_y"]),
                    "distance": float(proj["distance"]),
                    "bearing_deg": float(proj["bearing_deg"]),
                    "is_target": bool(proj["is_target"]),
                }
            targets.append(target_entry)

        return {
            "timestamp": float(time.time()),
            "frame_dimensions": {"width": int(frame_width), "height": int(frame_height)},
            "total_targets": len(targets),
            "targets": targets,
        }

