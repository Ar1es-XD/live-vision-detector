import React, { useEffect, useState, useRef } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { initModel, reloadModel, FilterCategory, Detection } from './utils/detector';
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
  LifeBuoy,
  Volume2,
  VolumeX,
  Search,
  X,
  AlertTriangle
} from 'lucide-react';

const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
  'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
  'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

export const App: React.FC = () => {
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [modelStatus, setModelStatus] = useState<string>('Initializing Neural Net...');
  const [isModelReady, setIsModelReady] = useState<boolean>(false);
  const [showCpuPrompt, setShowCpuPrompt] = useState<boolean>(false);

  const [confidence, setConfidence] = useState<number>(0.35);
  const [filterCategory, setFilterCategory] = useState<FilterCategory>('all');
  const [sarMode, setSarMode] = useState<boolean>(false);
  const [videoSourceType, setVideoSourceType] = useState<'webcam' | 'simulation' | 'upload'>('webcam');
  const [soundEnabled, setSoundEnabled] = useState<boolean>(true);

  const [currentDetections, setCurrentDetections] = useState<Detection[]>([]);
  const [currentTelemetry, setCurrentTelemetry] = useState<DroneTelemetry>(getSimulatedTelemetry(0));
  const [showInfoModal, setShowInfoModal] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>('');

  const lastPingTime = useRef<number>(0);

  // Initialize TensorFlow.js and COCO-SSD
  useEffect(() => {
    initModel((status) => setModelStatus(status))
      .then((loadedModel) => {
        setModel(loadedModel);
        setIsModelReady(true);
      })
      .catch((err) => {
        console.error('Failed to load model:', err);
        setModelStatus('Failed to load model. Check network/WebGL support.');
      });
  }, []);

  // 3-Minute Timer: Prompt user to switch to CPU if model hasn't loaded after 3 minutes
  useEffect(() => {
    if (isModelReady) {
      setShowCpuPrompt(false);
      return;
    }

    const threeMinutesMs = 3 * 60 * 1000;
    const timer = setTimeout(() => {
      if (!isModelReady) {
        setShowCpuPrompt(true);
      }
    }, threeMinutesMs);

    return () => clearTimeout(timer);
  }, [isModelReady]);

  const handleSwitchToCpu = () => {
    setShowCpuPrompt(false);
    setModel(null);
    setIsModelReady(false);
    setModelStatus('Switching to CPU engine...');
    reloadModel(true, s => setModelStatus(s))
      .then(m => {
        setModel(m);
        setIsModelReady(true);
      })
      .catch(err => {
        console.error('CPU fallback error:', err);
        setModelStatus('Failed to load model on CPU.');
      });
  };

  // Multimodal Tactile Audio Alert for SAR Target Lock
  useEffect(() => {
    if (!sarMode || !soundEnabled) return;
    const hasVictim = currentDetections.some(d => d.class.toLowerCase() === 'person');
    const now = Date.now();

    if (hasVictim && now - lastPingTime.current > 2000) {
      lastPingTime.current = now;
      try {
        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
        if (AudioCtx) {
          const ctx = new AudioCtx();
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.type = 'sine';
          osc.frequency.setValueAtTime(587.33, ctx.currentTime);
          osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.08);
          gain.gain.setValueAtTime(0.05, ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15);
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start();
          osc.stop(ctx.currentTime + 0.15);
        }
      } catch {
        // AudioContext restricted before gesture
      }
    }
  }, [currentDetections, sarMode, soundEnabled]);

  const filteredClasses = COCO_CLASSES.filter(c => 
    c.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-[#07090e] text-white flex flex-col font-sans selection:bg-emerald-500/30 selection:text-emerald-200">
      {/* Top Apple-style Frosted Header */}
      <header className="glass-surface sticky top-0 z-40 px-6 py-3.5 flex items-center justify-between transition-all">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-emerald-500 via-teal-500 to-cyan-400 p-[1px] shadow-lg shadow-emerald-950/40">
            <div className="w-full h-full bg-[#0b0e14] rounded-2xl flex items-center justify-center">
              <Radio className="w-4.5 h-4.5 text-emerald-400 animate-pulse" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="display-title font-bold text-base tracking-tight text-white">
                AeroVision <span className="text-emerald-400">SAR</span>
              </h1>
              <span className="glass-pill px-2 py-0.5 text-[10px] mono-metric font-semibold text-emerald-300 rounded-full">
                UAV AI v2
              </span>
            </div>
            <p className="text-[11px] text-white/50 subheadline hidden sm:block">
              Tactical Real-Time Computer Vision & Monocular Geolocation
            </p>
          </div>
        </div>

        {/* Engine Status Capsule & Quick Actions */}
        <div className="flex items-center gap-2.5">
          {/* Dynamic Island Model Status */}
          <div className="flex items-center gap-2 px-3 py-1.5 glass-pill rounded-full text-xs">
            <Cpu className="w-3.5 h-3.5 text-teal-400 shrink-0" />
            <span className="text-white/80 mono-metric text-[11px] max-w-[170px] sm:max-w-none truncate">{modelStatus}</span>
            {isModelReady ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
            ) : (
              <button
                onClick={() => {
                  setModel(null);
                  setIsModelReady(false);
                  reloadModel(true, s => setModelStatus(s))
                    .then(m => { setModel(m); setIsModelReady(true); })
                    .catch(() => setModelStatus('CPU fallback failed'));
                }}
                className="tap-feedback px-2 py-0.5 rounded-md bg-amber-500/25 hover:bg-amber-500/35 text-amber-200 text-[10px] font-semibold border border-amber-400/30 shrink-0"
                title="Force CPU Fallback if WebGL hangs"
              >
                Force CPU
              </button>
            )}
          </div>

          {/* Sound Toggle */}
          <button
            onClick={() => setSoundEnabled(!soundEnabled)}
            className={`tap-feedback p-2 glass-pill rounded-xl text-xs font-semibold transition ${
              soundEnabled ? 'text-emerald-400' : 'text-white/40'
            }`}
            title={soundEnabled ? 'Mute Alert Audio' : 'Unmute Alert Audio'}
          >
            {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>

          {/* Docs / Classes Modal Trigger */}
          <button
            onClick={() => setShowInfoModal(!showInfoModal)}
            className="tap-feedback flex items-center gap-1.5 px-3.5 py-1.5 glass-pill hover:bg-white/10 text-white/90 rounded-xl text-xs font-semibold transition"
          >
            <Info className="w-3.5 h-3.5 text-emerald-400" />
            <span className="hidden sm:inline">Classes & Docs</span>
          </button>
        </div>
      </header>

      {/* Main Viewport Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 flex flex-col gap-5">
        {/* 3-Minute Model Loading Fallback Prompt */}
        {showCpuPrompt && !isModelReady && (
          <div className="glass-surface-elevated p-4 rounded-2xl border border-amber-500/40 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 shadow-xl">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-amber-500/20 border border-amber-400/30 flex items-center justify-center text-amber-400 shrink-0">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <div>
                <div className="font-bold text-xs text-amber-200">
                  Model Loading Has Taken Over 3 Minutes
                </div>
                <p className="text-[11px] text-white/70 leading-relaxed">
                  WebGL initialization or network download may be stalled by your browser. Would you like to switch to CPU Mode?
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0 w-full sm:w-auto">
              <button
                onClick={handleSwitchToCpu}
                className="tap-feedback flex-1 sm:flex-initial px-3.5 py-1.5 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-semibold text-xs transition shadow-md"
              >
                Switch to CPU Mode
              </button>
              <button
                onClick={() => setShowCpuPrompt(false)}
                className="tap-feedback px-3 py-1.5 rounded-xl glass-pill hover:bg-white/10 text-white/70 text-xs font-medium transition"
              >
                Keep Waiting
              </button>
            </div>
          </div>
        )}

        {/* Disaster Mode Tactile Alert Banner */}
        {sarMode && (
          <div className="glass-surface-elevated flex items-center justify-between p-4 rounded-2xl border border-red-500/30 text-white text-xs sm:text-sm shadow-lg shadow-red-950/20">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-xl bg-red-500/20 flex items-center justify-center text-red-400">
                <Flame className="w-4.5 h-4.5 animate-pulse" />
              </div>
              <div>
                <span className="font-bold tracking-wide text-red-400 mr-2">
                  TACTICAL SAR ACTIVE:
                </span>
                <span className="text-white/80 subheadline">
                  Survivors prioritized with monocular ray-casting GPS ground coordinates.
                </span>
              </div>
            </div>
            <span className="mono-metric text-[11px] font-bold text-red-400/90 hidden md:inline px-2.5 py-1 rounded-lg bg-red-950/60 border border-red-800/40">
              ATAK / CoT SIMULATED
            </span>
          </div>
        )}

        {/* Video Viewfinder Player & HUD */}
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

        {/* Tactical Glass Control Bar */}
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

        {/* Apple Bento-Grid Architecture Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          {/* Card 1: Live Hardware Acceleration */}
          <div className="glass-surface p-5 rounded-3xl flex flex-col gap-2.5 transition-all hover:border-white/15">
            <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs uppercase tracking-wider">
              <Activity className="w-4 h-4" /> Client-Side WebGL Inference
            </div>
            <p className="text-xs text-white/60 leading-relaxed subheadline">
              Inference runs on-device using WebGL GPU shaders with zero server latency and total visual privacy.
            </p>
          </div>

          {/* Card 2: SAR Geolocation */}
          <div className="glass-surface p-5 rounded-3xl flex flex-col gap-2.5 transition-all hover:border-white/15">
            <div className="flex items-center gap-2 text-amber-400 font-bold text-xs uppercase tracking-wider">
              <LifeBuoy className="w-4 h-4" /> Monocular Target Geolocation
            </div>
            <p className="text-xs text-white/60 leading-relaxed subheadline">
              Ray-casting projects pixel centroids into real-world WGS84 coordinates using gimbal pitch and UAV altitude.
            </p>
          </div>

          {/* Card 3: Physical Drone Companion Edge */}
          <div className="glass-surface p-5 rounded-3xl flex flex-col gap-2.5 transition-all hover:border-white/15">
            <div className="flex items-center gap-2 text-teal-400 font-bold text-xs uppercase tracking-wider">
              <Terminal className="w-4 h-4" /> Drone Edge Python Pipeline
            </div>
            <p className="text-xs text-white/60 leading-relaxed subheadline">
              Compatible with Jetson Orin and Raspberry Pi 5 + Hailo-8 running YOLO11 over MAVLink telemetry.
            </p>
          </div>
        </div>
      </main>

      {/* Apple-style Translucent Sheet Modal: Classes & Documentation */}
      {showInfoModal && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-md z-50 flex items-center justify-center p-4">
          <div className="glass-surface-elevated rounded-3xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl border border-white/12 overflow-hidden transition-all animate-in fade-in zoom-in-95">
            {/* Sheet Grab Bar */}
            <div className="w-full flex justify-center pt-3 pb-1">
              <div className="w-10 h-1 rounded-full bg-white/25" />
            </div>

            {/* Modal Header */}
            <div className="px-6 py-4 border-b border-white/8 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl bg-emerald-500/20 flex items-center justify-center text-emerald-400">
                  <Layers className="w-4.5 h-4.5" />
                </div>
                <div>
                  <h3 className="font-bold text-sm text-white subheadline">
                    Model Specifications & Detectable Classes
                  </h3>
                  <p className="text-[11px] text-white/50">COCO MobileNet V2 Benchmark</p>
                </div>
              </div>
              <button
                onClick={() => setShowInfoModal(false)}
                className="tap-feedback p-1.5 rounded-full hover:bg-white/10 text-white/60 hover:text-white transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto text-xs text-white/75 space-y-5 leading-relaxed">
              {/* Overview */}
              <div>
                <h4 className="font-bold text-emerald-400 text-xs uppercase tracking-wider mb-1.5">
                  1. Real-Time Geolocation Ray-Casting
                </h4>
                <p className="text-white/60 leading-relaxed subheadline">
                  In <strong>SAR Mode</strong>, pixel coordinates are mapped through focal length and sensor pitch against drone altitude AGL to calculate estimated WGS84 GPS latitude and longitude coordinates.
                </p>
              </div>

              {/* Class Explorer with Quick Search */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-bold text-emerald-400 text-xs uppercase tracking-wider">
                    2. Detectable Objects ({filteredClasses.length} shown)
                  </h4>
                  <div className="relative">
                    <Search className="w-3.5 h-3.5 absolute left-2.5 top-2 text-white/40" />
                    <input
                      type="text"
                      placeholder="Search class..."
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      className="pl-8 pr-3 py-1 text-xs rounded-xl bg-black/40 border border-white/10 text-white placeholder-white/30 focus:outline-none focus:border-emerald-400/50"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5 max-h-56 overflow-y-auto p-1 text-[11px] mono-metric">
                  {filteredClasses.map((cls, idx) => {
                    const isSAR = ['person', 'boat', 'backpack', 'cell phone', 'dog'].includes(cls);
                    return (
                      <span
                        key={cls}
                        className={`p-1.5 rounded-lg flex items-center justify-between ${
                          isSAR
                            ? 'bg-amber-500/20 text-amber-200 border border-amber-400/30 font-semibold'
                            : 'glass-pill text-white/70'
                        }`}
                      >
                        <span>{idx + 1}. {cls}</span>
                        {isSAR && <span className="text-[9px] px-1 rounded bg-amber-500/40 text-amber-100">SAR</span>}
                      </span>
                    );
                  })}
                </div>
              </div>

              {/* Export Info */}
              <div>
                <h4 className="font-bold text-emerald-400 text-xs uppercase tracking-wider mb-1.5">
                  3. Exporting to ATAK / TAK Server
                </h4>
                <p className="text-white/60 leading-relaxed subheadline">
                  The <strong>Export GeoJSON</strong> button packages all verified targets into standard GIS point geometries ready for import into ATAK, QGroundControl, or ESRI ArcGIS.
                </p>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-white/8 flex justify-end bg-black/20">
              <button
                onClick={() => setShowInfoModal(false)}
                className="tap-feedback px-5 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-semibold shadow-md transition"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Apple-style Translucent Footer */}
      <footer className="border-t border-white/8 px-6 py-4 text-xs text-white/40 flex flex-wrap items-center justify-between gap-3 mt-auto">
        <div className="flex items-center gap-2">
          <span>AeroVision SAR Mission Control</span>
          <span>•</span>
          <span className="text-emerald-400/80">Fluid Apple Design</span>
        </div>
        <div className="flex items-center gap-4 mono-metric text-[11px]">
          <span>Privacy Verified (Zero Server Upload)</span>
        </div>
      </footer>
    </div>
  );
};

export default App;

