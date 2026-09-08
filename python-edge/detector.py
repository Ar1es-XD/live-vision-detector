"""
YOLO11 Edge Inference Engine for Physical Drone Deployment
Supports Apple Silicon GPU (MPS), NVIDIA CUDA, and CPU.
"""
import torch
from ultralytics import YOLO

class DroneEdgeDetector:
    def __init__(self, model_name: str = "yolo11n.pt", conf_threshold: float = 0.35):
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
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                detections.append({
                    "bbox": xyxy,
                    "confidence": conf,
                    "class_name": self.model.names[cls_id],
                })
        return detections
