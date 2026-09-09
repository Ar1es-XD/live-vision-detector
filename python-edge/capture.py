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
    Tries Quartz first if available, then falls back to built-in osascript (zero-dependency).
    Returns dict: {'x': int, 'y': int, 'w': int, 'h': int, 'title': str, 'owner': str} or None.
    """
    # 1. Try PyObjC Quartz if installed
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
    except Exception:
        pass

    # 2. Fallback to built-in macOS osascript (zero extra dependencies)
    import subprocess
    script = f'''
    tell application "System Events"
        set appList to (name of every process where background only is false)
        repeat with appName in appList
            if (appName as text) contains "{keyword}" then
                tell process appName
                    if (count of windows) > 0 then
                        set winPos to position of window 1
                        set winSize to size of window 1
                        return (item 1 of winPos as text) & "," & (item 2 of winPos as text) & "," & (item 1 of winSize as text) & "," & (item 2 of winSize as text) & "," & appName
                    end if
                end tell
            end if
        end repeat
    end tell
    return ""
    '''
    try:
        res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=2)
        out = res.stdout.strip()
        if out and "," in out:
            parts = [p.strip() for p in out.split(",")]
            if len(parts) >= 4:
                x, y, w, h = int(float(parts[0])), int(float(parts[1])), int(float(parts[2])), int(float(parts[3]))
                owner = parts[4] if len(parts) > 4 else keyword
                if w > 100 and h > 100:
                    return {"x": x, "y": y, "w": w, "h": h, "title": owner, "owner": owner}
    except Exception:
        pass

    return None


def find_windows_window(keyword: str):
    """
    Search for on-screen Windows windows containing `keyword` in title.
    Uses ctypes and user32 (zero external dependencies).
    """
    import sys
    if sys.platform != "win32":
        return None

    try:
        user32 = ctypes.windll.user32

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        found = []

        def enum_cb(hwnd, lparam):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value
                    if keyword.lower() in title.lower():
                        rect = RECT()
                        user32.GetWindowRect(hwnd, ctypes.byref(rect))
                        w = rect.right - rect.left
                        h = rect.bottom - rect.top
                        if w > 100 and h > 100:
                            found.append({"x": rect.left, "y": rect.top, "w": w, "h": h, "title": title, "owner": title})
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        if found:
            return found[0]
    except Exception:
        pass
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

        # 2. Try to find window automatically on OS
        auto_win = find_macos_window(self.target) or find_windows_window(self.target)
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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"[WARN] Camera #{idx} could not be opened directly. Trying default index 0...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera device index {idx} or default camera.")
        return cap, f"Camera #{idx}"

    # 2. Network video stream (UDP, RTSP, HTTP) or video file
    lower = source_arg.lower()
    if lower.startswith(("rtsp://", "udp://", "http://", "https://")) or lower.endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    ):
        print(f"[SOURCE] Connecting to stream URL: {source_arg}...")
        # Optimize OpenCV backend for lowest frame-buffer latency on live streams
        cap = cv2.VideoCapture(source_arg, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Could not connect to video stream: {source_arg}")
        return cap, f"Stream: {source_arg}"

    # 3. Emulator Window / Desktop Region Capture
    print(f"[SOURCE] Initializing Window Grabber targeting '{source_arg}'...")
    grabber = WindowStreamGrabber(target=source_arg)
    return grabber, grabber.window_title
