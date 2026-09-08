"""
Live Camera Object Detection on Drone Edge
"""
import os
import cv2
import time
from detector import DroneEdgeDetector

def run():
    detector = DroneEdgeDetector(model_name="yolo11n.pt", conf_threshold=0.35)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARN] Camera index 0 failed. Trying camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[FATAL] Could not open camera.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    os.makedirs("captures", exist_ok=True)

    sar_mode = False
    humans_only = False
    print("\n[READY] AeroVision Drone Edge Vision Running")
    print("Keys: [Q] Quit | [S] Toggle SAR Mode | [C] Humans Only | [SPACE] Snapshot\n")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        detections = detector.infer(frame)

        # Draw overlays
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']*100:.1f}%"
            color = (0, 0, 255) if (sar_mode and det['class_name'] == 'person') else (255, 165, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if sar_mode and det['class_name'] == 'person':
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        # Top banner
        cv2.putText(frame, f"FPS: {fps:.1f} | Mode: {'SAR TACTICAL' if sar_mode else 'NORMAL'}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("AeroVision SAR (Drone Edge)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            sar_mode = not sar_mode
            print(f"[MODE] SAR Mode: {'ACTIVE' if sar_mode else 'STANDBY'}")
        elif key == ord('c'):
            humans_only = not humans_only
            detector.set_target_classes(["person"] if humans_only else None)
            print(f"[FILTER] Humans Only: {'ON' if humans_only else 'OFF'}")
        elif key == 32:
            snap_path = f"captures/incident_{int(time.time())}.jpg"
            cv2.imwrite(snap_path, frame)
            print(f"[CAPTURED] Saved {snap_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
