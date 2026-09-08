# AeroVision SAR: Live Camera & Drone Object Classifier

A real-time, hardware-accelerated computer vision application for detecting and tracking 80 real-world object categories with a specialized **Search and Rescue (SAR) Drone Mode** designed for disaster relief, survivor location, and tactical geolocation.

![AeroVision SAR Banner](https://images.unsplash.com/photo-1508614589041-895b88991e3e?auto=format&fit=crop&w=1200&q=80)

## 🚀 Key Features

* **⚡ Real-Time In-Browser AI (WebGL / WebGPU):** Powered by TensorFlow.js and MobileNet COCO-SSD, running at 30–60 FPS directly on the client's device with zero server latency.
* **🎯 80 Detectable Object Categories:** Detects people, boats, vehicles, backpacks, communication equipment, livestock, and infrastructure.
* **🚁 Tactical SAR Drone Mode:**
  * Isolates human targets in distress with military-grade target reticles.
  * Real-time monocular ray-casting estimating ground **WGS84 GPS Coordinates** based on camera pitch, drone heading, and altitude AGL.
  * **GeoJSON Incident Export:** Instant export of confirmed victim coordinates formatted for ATAK, WinTAK, QGroundControl, and GIS dispatch systems.
* **🎥 Multi-Source Feed Input:**
  * **Live Webcam:** Direct hardware camera ingestion via HTML5 `getUserMedia`.
  * **Aerial Drone Simulation:** Built-in synthetic flood/wilderness disaster terrain loop for testing without a camera.
  * **Custom Video Upload:** Drag-and-drop custom aerial MP4/WebM drone footage for instant inference.
* **🛩️ Physical Drone Companion Pipeline:** Includes `python-edge/` running **YOLO11** with Apple Silicon MPS / NVIDIA CUDA for deployment on companion computers (Jetson Orin Nano, Raspberry Pi 5 + Hailo-8).

---

## 💻 Quickstart (Web App)

```bash
# Install dependencies
npm install

# Start local development server
npm run dev

# Build for production
npm run build
```

---

## 🚁 Physical Drone Deployment (`python-edge/`)

```bash
cd python-edge
pip install -r requirements.txt
python main.py
```

* **`[S]`**: Toggle SAR Drone Mode
* **`[C]`**: Toggle Humans Only Filter
* **`[SPACE]`**: Save High-Res Geotagged Snapshot
* **`[Q]`**: Quit cleanly

---

## 📄 License
MIT License. Built for Search & Rescue, Humanitarian Assistance, and Disaster Relief (HADR).
