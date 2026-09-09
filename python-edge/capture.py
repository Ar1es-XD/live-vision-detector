"""
Video Capture and Window Stream Ingestion for Drone Edge Pipeline
Supports:
1. Native Emulator Window Capture (BlueStacks, Android Emulator, etc.) via mss
2. Interactive Screen ROI Selection
3. Network Video Streams (UDP, RTSP, HTTP)
4. Local Camera Devices
"""
import os
import json
import time
import ctypes
import numpy as np
import cv2

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

CONFIG_FILE = os.path.join(os.path.dirname(__file__), ".crop_config.json")


def find_macos_window(keyword: str):
    """
    Search for on-screen macOS windows containing `keyword` in owner or window name.
    Returns dict: {'x': int, 'y': int, 'w': int, 'h': int, 'title': str, 'owner': str} or None.
    """
    try:
        from ctypes import c_void_p, c_uint32, c_int
        cg = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        cf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")

        # CoreFoundation functions
        cf.CFArrayGetCount.restype = c_int
        cf.CFArrayGetCount.argtypes = [c_void_p]
        cf.CFArrayGetValueAtIndex.restype = c_void_p
        cf.CFArrayGetValueAtIndex.argtypes = [c_void_p, c_int]

        # Query on-screen windows
        kCGWindowListOptionOnScreenOnly = 1
        kCGNullWindowID = 0
        cg.CGWindowListCopyWindowInfo.restype = c_void_p
        cg.CGWindowListCopyWindowInfo.argtypes = [c_uint32, c_uint32]

        win_list = cg.CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        if not win_list:
            return None

        try:
            import Quartz
            window_info_list = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
            )
            for w in window_info_list:
                owner = str(w.get(Quartz.kCGWindowOwnerName, ""))
                name = str(w.get(Quartz.kCGWindowName, ""))
                combined = f"{owner} {name}".lower()
                if keyword.lower() in combined:
                    bounds = w.get(Quartz.kCGWindowBounds, {})
                    width = int(bounds.get("Width", 0))
                    height = int(bounds.get("Height", 0))
                    if width > 100 and height > 100:
                        return {
                            "x": int(bounds.get("X", 0)),
                            "y": int(bounds.get("Y", 0)),
                            "w": width,
                            "h": height,
                            "title": name or owner,
                            "owner": owner,
                        }
        except ImportError:
            pass
    except Exception as e:
        print(f"[WARN] macOS window search error: {e}")
    return None


class WindowStreamGrabber:
    """
    Low-latency emulator window / screen grabber.
    Uses mss for high-FPS desktop capture.
    """

    def __init__(self, target: str = "bluestacks"):
        if not HAS_MSS:
            raise RuntimeError(
                "Package 'mss' is required for window capture. Install with: pip install mss"
            )

        self.sct = mss.mss()
        self.target = target
        self.region = None
        self.window_title = "Emulator Window"

        self._init_region()

    def _init_region(self):
        # 1. Check if user provided explicit coordinates "x,y,w,h"
        if "," in self.target:
            parts = [int(p.strip()) for p in self.target.split(",")]
            if len(parts) == 4:
                self.region = {
                    "left": parts[0],
                    "top": parts[1],
                    "width": parts[2],
                    "height": parts[3],
                }
                print(f"[CAPTURE] Explicit region configured: {self.region}")
                return

        # 2. Try to find window automatically on macOS
        auto_win = find_macos_window(self.target)
        if auto_win:
            self.region = {
                "left": auto_win["x"],
                "top": auto_win["y"],
                "width": auto_win["w"],
                "height": auto_win["h"],
            }
            self.window_title = f"{auto_win['owner']} ({auto_win['title']})"
            print(f"[CAPTURE] Auto-detected window '{self.window_title}' at {self.region}")
            self.save_config()
            return

        # 3. Check for previously saved config
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
                    if "region" in data:
                        self.region = data["region"]
                        self.window_title = data.get("title", "Saved Emulator Region")
                        print(f"[CAPTURE] Loaded cached emulator region from {CONFIG_FILE}: {self.region}")
                        return
            except Exception:
                pass

        # 4. Fallback to Primary Monitor
        mon = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]
        self.region = {
            "left": mon["left"],
            "top": mon["top"],
            "width": mon["width"],
            "height": mon["height"],
        }
        self.window_title = f"Full Screen ({self.region['width']}x{self.region['height']})"
        print(f"[CAPTURE] Using primary display: {self.region}")
        print("[TIP] Press [W] during execution to interactively crop the emulator window!")

    def select_roi_interactive(self):
        """
        Grab a full screenshot and let user select the emulator window with mouse drag.
        """
        print("\n[CALIBRATE] Capturing full screen for interactive region selection...")
        time.sleep(0.5)
        mon = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]
        screenshot = np.array(self.sct.grab(mon))
        bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # Scale down if display resolution is huge (Retina screens)
        h, w = bgr.shape[:2]
        scale = 1.0
        display_img = bgr
        if w > 1920:
            scale = 1920 / w
            display_img = cv2.resize(bgr, (int(w * scale), int(h * scale)))

        win_name = "DRAG A BOX AROUND THE EMULATOR WINDOW (ENTER to Confirm, C to Cancel)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        roi = cv2.selectROI(win_name, display_img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(win_name)

        rx, ry, rw, rh = roi
        if rw > 50 and rh > 50:
            # Rescale back to native coordinates
            native_x = mon["left"] + int(rx / scale)
            native_y = mon["top"] + int(ry / scale)
            native_w = int(rw / scale)
            native_h = int(rh / scale)

            self.region = {
                "left": native_x,
                "top": native_y,
                "width": native_w,
                "height": native_h,
            }
            self.window_title = f"Calibrated Emulator ({native_w}x{native_h})"
            print(f"[CALIBRATE] Successfully selected region: {self.region}")
            self.save_config()
            return True
        else:
            print("[CALIBRATE] Selection cancelled or too small.")
            return False

    def save_config(self):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump({"region": self.region, "title": self.window_title}, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not save crop config: {e}")

    def read(self):
        """
        Grab next frame.
        Returns: (ret: bool, frame: np.ndarray BGR)
        """
        try:
            sct_img = self.sct.grab(self.region)
            frame = np.frombuffer(sct_img.raw, dtype=np.uint8).reshape((sct_img.height, sct_img.width, 4))
            # Slice BGRA -> BGR (zero copy stride / fast)
            bgr = frame[:, :, :3].copy()
            return True, bgr
        except Exception as e:
            print(f"[ERROR] Frame capture error: {e}")
            return False, None

    def release(self):
        try:
            self.sct.close()
        except Exception:
            pass


def create_capture_source(source_arg: str):
    """
    Factory creating either a WindowStreamGrabber or cv2.VideoCapture depending on input.
    """
    # 1. Numeric camera index
    if source_arg.isdigit():
        idx = int(source_arg)
        print(f"[SOURCE] Opening camera index {idx}...")
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap, f"Camera #{idx}"

    # 2. Network video stream (UDP, RTSP, HTTP) or video file
    lower = source_arg.lower()
    if lower.startswith(("rtsp://", "udp://", "http://", "https://")) or lower.endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    ):
        print(f"[SOURCE] Connecting to stream URL: {source_arg}...")
        # For UDP/RTSP, optimize OpenCV backend flags for low latency
        cap = cv2.VideoCapture(source_arg, cv2.CAP_FFMPEG)
        return cap, f"Stream: {source_arg}"

    # 3. Emulator Window / Desktop Region Capture
    print(f"[SOURCE] Initializing Window Grabber targeting '{source_arg}'...")
    grabber = WindowStreamGrabber(target=source_arg)
    return grabber, grabber.window_title
