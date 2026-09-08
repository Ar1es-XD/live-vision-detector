import React, { useEffect, useState } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { initModel, FilterCategory, Detection } from './utils/detector';
import { DroneTelemetry, getSimulatedTelemetry } from './utils/sarTelemetry';
import { VideoPlayer } from './components/VideoPlayer';
import { ControlBar } from './components/ControlBar';
import { 
  Radio, 
  Cpu, 
  Layers, 
  Terminal, 
  Info, 
  CheckCircle2, 
  Activity,
  Flame,
  LifeBuoy
} from 'lucide-react';

export const App: React.FC = () => {
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [modelStatus, setModelStatus] = useState<string>('Initializing Neural Net...');
  const [isModelReady, setIsModelReady] = useState<boolean>(false);

  const [confidence, setConfidence] = useState<number>(0.35);
  const [filterCategory, setFilterCategory] = useState<FilterCategory>('all');
  const [sarMode, setSarMode] = useState<boolean>(false);
  const [videoSourceType, setVideoSourceType] = useState<'webcam' | 'simulation' | 'upload'>('webcam');

  const [currentDetections, setCurrentDetections] = useState<Detection[]>([]);
  const [currentTelemetry, setCurrentTelemetry] = useState<DroneTelemetry>(getSimulatedTelemetry(0));
  const [showInfoModal, setShowInfoModal] = useState<boolean>(false);

  // Initialize TensorFlow.js and COCO-SSD
  useEffect(() => {
    initModel((status) => setModelStatus(status))
      .then((loadedModel) => {
        setModel(loadedModel);
        setIsModelReady(true);
        setModelStatus('Neural Engine Active (WebGL Accelerated)');
      })
      .catch((err) => {
        console.error('Failed to load model:', err);
        setModelStatus('Failed to load model. Check network/WebGL support.');
      });
  }, []);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col font-sans">
      {/* Top Navigation Header */}
      <header className="border-b border-gray-800/80 bg-gray-950/80 backdrop-blur-md sticky top-0 z-40 px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-900/30">
            <Radio className="w-5 h-5 text-black animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-extrabold text-base tracking-tight text-white">
                AeroVision <span className="text-emerald-400">SAR</span>
              </h1>
              <span className="px-2 py-0.5 text-[10px] font-mono uppercase bg-emerald-950 text-emerald-300 border border-emerald-800 rounded-md">
                Live Drone AI
              </span>
            </div>
            <p className="text-xs text-gray-400 hidden sm:block">
              Real-Time Visual Object Classifier & Disaster Rescue Tracking
            </p>
          </div>
        </div>

        {/* Engine Status & Badges */}
        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-2 px-3 py-1 bg-gray-900 border border-gray-800 rounded-lg text-xs">
            <Cpu className="w-3.5 h-3.5 text-teal-400" />
            <span className="text-gray-400 font-mono">{modelStatus}</span>
            {isModelReady && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />}
          </div>

          <button
            onClick={() => setShowInfoModal(!showInfoModal)}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-900 hover:bg-gray-800 text-gray-300 border border-gray-800 rounded-lg text-xs font-semibold transition"
          >
            <Info className="w-3.5 h-3.5 text-emerald-400" />
            <span className="hidden sm:inline">Classes & Docs</span>
          </button>
        </div>
      </header>

      {/* Main Viewport Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 flex flex-col gap-5">
        {/* Disaster Mode Alert Banner */}
        {sarMode && (
          <div className="flex items-center justify-between p-3.5 bg-red-950/60 border border-red-800/80 rounded-xl text-red-200 text-xs sm:text-sm animate-pulse">
            <div className="flex items-center gap-2.5">
              <Flame className="w-4 h-4 text-red-400" />
              <span className="font-bold tracking-wide">
                TACTICAL SAR ACTIVE:
              </span>
              <span className="text-red-300">
                Prioritizing human distress recognition, survivor isolation, and GPS ray-casting coordinates.
              </span>
            </div>
            <span className="font-mono text-xs font-bold text-red-400 hidden md:inline">
              RELAY: ATAK / CoT SIMULATED
            </span>
          </div>
        )}

        {/* Video Player & Tactical HUD */}
        <VideoPlayer
          model={model}
          confidenceThreshold={confidence}
          filterCategory={filterCategory}
          sarMode={sarMode}
          videoSourceType={videoSourceType}
          onSourceChange={setVideoSourceType}
          onDetectionsUpdate={(dets, telem) => {
            setCurrentDetections(dets);
            setCurrentTelemetry(telem);
          }}
        />

        {/* Tactical Control Bar */}
        <ControlBar
          confidence={confidence}
          onConfidenceChange={setConfidence}
          filterCategory={filterCategory}
          onFilterChange={setFilterCategory}
          sarMode={sarMode}
          onSarModeToggle={() => setSarMode(!sarMode)}
          currentDetections={currentDetections}
          currentTelemetry={currentTelemetry}
        />

        {/* Architecture & Drone Integration Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          {/* Card 1: Live Hardware Acceleration */}
          <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800/80 flex flex-col gap-2">
            <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs uppercase tracking-wider">
              <Activity className="w-4 h-4" /> Client-Side WebGL Inference
            </div>
            <p className="text-xs text-gray-400 leading-relaxed">
              Inference runs directly on your device’s GPU using WebGL shaders. No video frames are transmitted to cloud servers, providing zero latency and 100% privacy.
            </p>
          </div>

          {/* Card 2: SAR Geolocation */}
          <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800/80 flex flex-col gap-2">
            <div className="flex items-center gap-2 text-amber-400 font-bold text-xs uppercase tracking-wider">
              <LifeBuoy className="w-4 h-4" /> Monocular Target Geolocation
            </div>
            <p className="text-xs text-gray-400 leading-relaxed">
              When in SAR mode, bounding boxes are projected using camera intrinsics and simulated drone altitude/heading to estimate real-world WGS84 GPS ground coordinates.
            </p>
          </div>

          {/* Card 3: Physical Drone Companion Edge */}
          <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800/80 flex flex-col gap-2">
            <div className="flex items-center gap-2 text-teal-400 font-bold text-xs uppercase tracking-wider">
              <Terminal className="w-4 h-4" /> Drone Edge Python Pipeline
            </div>
            <p className="text-xs text-gray-400 leading-relaxed">
              For flying on physical drones (Jetson Orin / Raspberry Pi 5 + Hailo), the repository includes <code className="text-gray-200">python-edge/</code> running YOLO11 with MAVLink.
            </p>
          </div>
        </div>
      </main>

      {/* Info & 80-Class Modal */}
      {showInfoModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-gray-900 border border-gray-800 rounded-2xl max-w-2xl w-full max-h-[80vh] flex flex-col shadow-2xl">
            <div className="p-5 border-b border-gray-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers className="w-5 h-5 text-emerald-400" />
                <h3 className="font-bold text-base text-white">
                  AeroVision Model Specifications & 80 Detectable Classes
                </h3>
              </div>
              <button
                onClick={() => setShowInfoModal(false)}
                className="text-gray-400 hover:text-white px-2 py-1 text-sm font-bold"
              >
                ✕
              </button>
            </div>

            <div className="p-6 overflow-y-auto text-xs text-gray-300 space-y-4 leading-relaxed">
              <div>
                <h4 className="font-bold text-emerald-400 text-sm mb-1">
                  1. How Disaster Detection Works
                </h4>
                <p>
                  The model performs live object localization, classification, and confidence scoring at 30–60 FPS. In <strong>SAR Mode</strong>, the pipeline applies monocular ray-casting to compute estimated target latitude/longitude coordinates based on camera focal length, pitch angle, and UAV altitude AGL.
                </p>
              </div>

              <div>
                <h4 className="font-bold text-emerald-400 text-sm mb-1">
                  2. Full Range of Detectable Objects (COCO 80 Benchmark)
                </h4>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 text-[11px] font-mono">
                  <span className="p-1.5 bg-red-950/70 border border-red-800 text-red-300 rounded font-bold">1. person (SAR)</span>
                  <span className="p-1.5 bg-amber-950/70 border border-amber-800 text-amber-300 rounded font-bold">2. boat (SAR)</span>
                  <span className="p-1.5 bg-gray-800 rounded">3. car</span>
                  <span className="p-1.5 bg-gray-800 rounded">4. motorcycle</span>
                  <span className="p-1.5 bg-gray-800 rounded">5. airplane</span>
                  <span className="p-1.5 bg-gray-800 rounded">6. bus</span>
                  <span className="p-1.5 bg-gray-800 rounded">7. train</span>
                  <span className="p-1.5 bg-gray-800 rounded">8. truck</span>
                  <span className="p-1.5 bg-gray-800 rounded">9. traffic light</span>
                  <span className="p-1.5 bg-gray-800 rounded">10. fire hydrant</span>
                  <span className="p-1.5 bg-amber-950/70 border border-amber-800 text-amber-300 rounded">11. backpack</span>
                  <span className="p-1.5 bg-gray-800 rounded">12. umbrella</span>
                  <span className="p-1.5 bg-gray-800 rounded">13. handbag</span>
                  <span className="p-1.5 bg-gray-800 rounded">14. suitcase</span>
                  <span className="p-1.5 bg-gray-800 rounded">15. frisbee</span>
                  <span className="p-1.5 bg-gray-800 rounded">16. skis</span>
                  <span className="p-1.5 bg-gray-800 rounded">17. snowboard</span>
                  <span className="p-1.5 bg-gray-800 rounded">18. sports ball</span>
                  <span className="p-1.5 bg-gray-800 rounded">19. kite</span>
                  <span className="p-1.5 bg-gray-800 rounded">20. baseball bat</span>
                  <span className="p-1.5 bg-gray-800 rounded">21. baseball glove</span>
                  <span className="p-1.5 bg-gray-800 rounded">22. skateboard</span>
                  <span className="p-1.5 bg-gray-800 rounded">23. surfboard</span>
                  <span className="p-1.5 bg-gray-800 rounded">24. tennis racket</span>
                  <span className="p-1.5 bg-gray-800 rounded">25. bottle</span>
                  <span className="p-1.5 bg-gray-800 rounded">26. wine glass</span>
                  <span className="p-1.5 bg-gray-800 rounded">27. cup</span>
                  <span className="p-1.5 bg-gray-800 rounded">28. fork</span>
                  <span className="p-1.5 bg-gray-800 rounded">29. knife</span>
                  <span className="p-1.5 bg-gray-800 rounded">30. spoon</span>
                  <span className="p-1.5 bg-gray-800 rounded">31. bowl</span>
                  <span className="p-1.5 bg-gray-800 rounded">32. banana</span>
                  <span className="p-1.5 bg-gray-800 rounded">33. apple</span>
                  <span className="p-1.5 bg-gray-800 rounded">34. sandwich</span>
                  <span className="p-1.5 bg-gray-800 rounded">35. orange</span>
                  <span className="p-1.5 bg-gray-800 rounded">36. broccoli</span>
                  <span className="p-1.5 bg-gray-800 rounded">37. carrot</span>
                  <span className="p-1.5 bg-gray-800 rounded">38. hot dog</span>
                  <span className="p-1.5 bg-gray-800 rounded">39. pizza</span>
                  <span className="p-1.5 bg-gray-800 rounded">40. donut</span>
                  <span className="p-1.5 bg-gray-800 rounded">41. cake</span>
                  <span className="p-1.5 bg-gray-800 rounded">42. chair</span>
                  <span className="p-1.5 bg-gray-800 rounded">43. couch</span>
                  <span className="p-1.5 bg-gray-800 rounded">44. potted plant</span>
                  <span className="p-1.5 bg-gray-800 rounded">45. bed</span>
                  <span className="p-1.5 bg-gray-800 rounded">46. dining table</span>
                  <span className="p-1.5 bg-gray-800 rounded">47. toilet</span>
                  <span className="p-1.5 bg-gray-800 rounded">48. tv</span>
                  <span className="p-1.5 bg-gray-800 rounded">49. laptop</span>
                  <span className="p-1.5 bg-gray-800 rounded">50. mouse</span>
                  <span className="p-1.5 bg-gray-800 rounded">51. remote</span>
                  <span className="p-1.5 bg-gray-800 rounded">52. keyboard</span>
                  <span className="p-1.5 bg-amber-950/70 border border-amber-800 text-amber-300 rounded font-bold">53. cell phone</span>
                  <span className="p-1.5 bg-gray-800 rounded">54. microwave</span>
                  <span className="p-1.5 bg-gray-800 rounded">55. oven</span>
                  <span className="p-1.5 bg-gray-800 rounded">56. toaster</span>
                  <span className="p-1.5 bg-gray-800 rounded">57. sink</span>
                  <span className="p-1.5 bg-gray-800 rounded">58. refrigerator</span>
                  <span className="p-1.5 bg-gray-800 rounded">59. book</span>
                  <span className="p-1.5 bg-gray-800 rounded">60. clock</span>
                  <span className="p-1.5 bg-gray-800 rounded">61. vase</span>
                  <span className="p-1.5 bg-gray-800 rounded">62. scissors</span>
                  <span className="p-1.5 bg-gray-800 rounded">63. teddy bear</span>
                  <span className="p-1.5 bg-gray-800 rounded">64. hair drier</span>
                  <span className="p-1.5 bg-gray-800 rounded">65. toothbrush</span>
                  <span className="p-1.5 bg-gray-800 rounded">66. bird</span>
                  <span className="p-1.5 bg-gray-800 rounded">67. cat</span>
                  <span className="p-1.5 bg-amber-950/70 border border-amber-800 text-amber-300 rounded">68. dog</span>
                  <span className="p-1.5 bg-gray-800 rounded">69. horse</span>
                  <span className="p-1.5 bg-gray-800 rounded">70. sheep</span>
                  <span className="p-1.5 bg-gray-800 rounded">71. cow</span>
                  <span className="p-1.5 bg-gray-800 rounded">72. elephant</span>
                  <span className="p-1.5 bg-gray-800 rounded">73. bear</span>
                  <span className="p-1.5 bg-gray-800 rounded">74. zebra</span>
                  <span className="p-1.5 bg-gray-800 rounded">75. giraffe</span>
                  <span className="p-1.5 bg-gray-800 rounded">76. bicycle</span>
                  <span className="p-1.5 bg-gray-800 rounded">77. stop sign</span>
                  <span className="p-1.5 bg-gray-800 rounded">78. parking meter</span>
                  <span className="p-1.5 bg-gray-800 rounded">79. bench</span>
                  <span className="p-1.5 bg-gray-800 rounded">80. toaster</span>
                </div>
              </div>

              <div>
                <h4 className="font-bold text-emerald-400 text-sm mb-1">
                  3. Exporting to First Responders (ATAK / TAK Server)
                </h4>
                <p>
                  Clicking <strong>Export GeoJSON</strong> generates standard geospatial point features containing target ID, WGS84 GPS coordinates, confidence, and timestamp, compatible with QGroundControl, ATAK, and GIS dispatch platforms.
                </p>
              </div>
            </div>

            <div className="p-4 border-t border-gray-800 flex justify-end">
              <button
                onClick={() => setShowInfoModal(false)}
                className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-semibold transition"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="border-t border-gray-800/80 bg-gray-950 px-6 py-4 text-xs text-gray-500 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span>AeroVision SAR Mission Control</span>
          <span>•</span>
          <span>Open-Source Live Vision</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-gray-400 font-mono">
            Deployed on Vercel
          </span>
        </div>
      </footer>
    </div>
  );
};

export default App;
