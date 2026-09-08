import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Detection, FilterCategory, detectFrame } from '../utils/detector';
import { TacticalHUD } from './TacticalHUD';
import { DroneTelemetry, getSimulatedTelemetry } from '../utils/sarTelemetry';
import { Camera, Video, Upload, AlertCircle, RefreshCw, Radio } from 'lucide-react';

interface VideoPlayerProps {
  model: cocoSsd.ObjectDetection | null;
  confidenceThreshold: number;
  filterCategory: FilterCategory;
  sarMode: boolean;
  onDetectionsUpdate?: (detections: Detection[], telemetry: DroneTelemetry) => void;
  videoSourceType: 'webcam' | 'simulation' | 'upload';
  onSourceChange: (source: 'webcam' | 'simulation' | 'upload') => void;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  model,
  confidenceThreshold,
  filterCategory,
  sarMode,
  onDetectionsUpdate,
  videoSourceType,
  onSourceChange,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const simCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [detections, setDetections] = useState<Detection[]>([]);
  const [telemetry, setTelemetry] = useState<DroneTelemetry>(getSimulatedTelemetry(0));
  const [fps, setFps] = useState<number>(0);
  const [streamDimensions, setStreamDimensions] = useState({ width: 1280, height: 720 });
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Performance FPS tracking
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(performance.now());
  const tickRef = useRef(0);
  const animationFrameId = useRef<number | null>(null);

  // 1. Initialize Webcam Stream
  const startWebcam = useCallback(async () => {
    setCameraError(null);
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera API not supported in this browser environment.');
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play();
            setStreamDimensions({
              width: videoRef.current.videoWidth || 1280,
              height: videoRef.current.videoHeight || 720,
            });
          }
        };
      }
    } catch (err: any) {
      console.error('Camera access error:', err);
      let msg = 'Unable to access camera. Please allow camera permissions in your browser settings.';
      if (err.name === 'NotAllowedError') {
        msg = 'Camera permission was denied. Please allow camera access in browser URL bar.';
      } else if (err.name === 'NotFoundError') {
        msg = 'No camera device found on this system. Switch to Drone Simulation mode below.';
      }
      setCameraError(msg);
    }
  }, []);

  // Stop Webcam
  const stopWebcam = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  }, []);

  // 2. Handle Source Switching
  useEffect(() => {
    if (videoSourceType === 'webcam') {
      startWebcam();
    } else {
      stopWebcam();
    }

    return () => {
      stopWebcam();
    };
  }, [videoSourceType, startWebcam, stopWebcam]);

  // 3. Procedural Aerial Drone Simulation Loop
  useEffect(() => {
    if (videoSourceType !== 'simulation') return;

    const canvas = simCanvasRef.current;
    if (!canvas) return;
    canvas.width = 1280;
    canvas.height = 720;
    setStreamDimensions({ width: 1280, height: 720 });
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let simTick = 0;
    let simLoopId: number;

    const renderSim = () => {
      simTick += 1;

      // Synthetic satellite wilderness flood terrain
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Flooded lake / river basin
      ctx.fillStyle = '#092537';
      ctx.beginPath();
      ctx.ellipse(640, 360, 480, 260, 0.2, 0, Math.PI * 2);
      ctx.fill();

      // Sonar / sensor scan sweeps
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.15)';
      ctx.lineWidth = 1.5;
      for (let i = 0; i < 4; i++) {
        const r = ((simTick * 2 + i * 80) % 320) + 40;
        ctx.beginPath();
        ctx.arc(640, 360, r, 0, Math.PI * 2);
        ctx.stroke();
      }

      // 1. Stranded civilian survivor
      const personX = 580 + Math.sin(simTick * 0.02) * 15;
      const personY = 320 + Math.cos(simTick * 0.02) * 10;
      // Jacket
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(personX, personY, 18, 0, Math.PI * 2);
      ctx.fill();
      // Head
      ctx.fillStyle = '#fde047';
      ctx.beginPath();
      ctx.arc(personX, personY - 8, 8, 0, Math.PI * 2);
      ctx.fill();

      // 2. Rescue response boat
      const boatX = 350 + (simTick * 1.2) % 600;
      const boatY = 420 + Math.sin(simTick * 0.05) * 12;
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.ellipse(boatX, boatY, 35, 14, -0.1, 0, Math.PI * 2);
      ctx.fill();

      // 3. Transport vehicle
      ctx.fillStyle = '#475569';
      ctx.fillRect(850, 180, 70, 35);
      ctx.fillStyle = '#94a3b8';
      ctx.fillRect(865, 185, 40, 25);

      simLoopId = requestAnimationFrame(renderSim);
    };

    renderSim();
    return () => cancelAnimationFrame(simLoopId);
  }, [videoSourceType]);

  // 4. Main AI Inference Loop
  useEffect(() => {
    let isRunning = true;

    const runInference = async () => {
      if (!isRunning) return;

      const sourceElement =
        videoSourceType === 'simulation'
          ? simCanvasRef.current
          : videoRef.current;

      if (model && sourceElement) {
        // Calculate FPS
        frameCountRef.current += 1;
        const now = performance.now();
        const elapsed = now - lastTimeRef.current;
        if (elapsed >= 500) {
          setFps((frameCountRef.current * 1000) / elapsed);
          frameCountRef.current = 0;
          lastTimeRef.current = now;
        }

        // Update simulated telemetry
        tickRef.current += 1;
        const currentTelemetry = getSimulatedTelemetry(tickRef.current);
        setTelemetry(currentTelemetry);

        // Run detection
        const results = await detectFrame(model, sourceElement, confidenceThreshold, filterCategory);
        setDetections(results);
        onDetectionsUpdate?.(results, currentTelemetry);
      }

      animationFrameId.current = requestAnimationFrame(runInference);
    };

    runInference();

    return () => {
      isRunning = false;
      if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
    };
  }, [model, confidenceThreshold, filterCategory, videoSourceType, onDetectionsUpdate]);

  // File Upload Handler
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      onSourceChange('upload');
      if (videoRef.current) {
        videoRef.current.srcObject = null;
        videoRef.current.src = url;
        videoRef.current.loop = true;
        videoRef.current.play();
      }
    }
  };

  return (
    <div className="relative w-full rounded-3xl overflow-hidden glass-surface-elevated shadow-2xl transition-all">
      {/* Video Viewfinder Viewport */}
      <div ref={containerRef} className="relative aspect-video w-full bg-black flex items-center justify-center overflow-hidden">
        {videoSourceType === 'simulation' ? (
          <canvas
            ref={simCanvasRef}
            className="w-full h-full object-contain"
          />
        ) : (
          <video
            ref={videoRef}
            playsInline
            muted
            autoPlay
            className="w-full h-full object-contain"
          />
        )}

        {/* Floating Top Viewfinder Pill Indicators */}
        <div className="absolute top-4 left-4 z-20 flex items-center gap-2 pointer-events-none">
          <div className="glass-pill px-3 py-1 rounded-full flex items-center gap-2 text-[11px] font-semibold text-white/90 shadow-md">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
            </span>
            <span className="mono-metric tracking-wider uppercase">
              {videoSourceType === 'webcam' ? 'OPTICAL LIVE' : videoSourceType === 'simulation' ? 'SYNTHETIC SIM' : 'STREAM FILE'}
            </span>
          </div>

          <div className="glass-pill px-2.5 py-1 rounded-full text-[11px] mono-metric font-medium text-emerald-400 shadow-md hidden sm:flex items-center gap-1.5">
            <Radio className="w-3 h-3 animate-pulse" />
            <span>{Math.round(fps)} FPS</span>
          </div>
        </div>

        <div className="absolute top-4 right-4 z-20 flex items-center gap-2 pointer-events-none">
          <div className="glass-pill px-3 py-1 rounded-full text-[11px] mono-metric font-medium text-white/75 shadow-md">
            <span>{streamDimensions.width}×{streamDimensions.height}</span>
          </div>
        </div>

        {/* Tactical HUD Overlay Layer */}
        <TacticalHUD
          detections={detections}
          telemetry={telemetry}
          sarMode={sarMode}
          videoWidth={streamDimensions.width}
          videoHeight={streamDimensions.height}
          fps={fps}
        />

        {/* Camera Permission Alert (Apple Modal Style) */}
        {cameraError && videoSourceType === 'webcam' && (
          <div className="absolute inset-0 bg-black/85 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center z-30">
            <div className="glass-surface-elevated max-w-md p-6 rounded-3xl flex flex-col items-center border border-white/10 shadow-2xl">
              <div className="w-12 h-12 rounded-2xl bg-amber-500/20 border border-amber-400/30 flex items-center justify-center mb-3">
                <AlertCircle className="w-6 h-6 text-amber-400" />
              </div>
              <h3 className="text-base font-bold text-white mb-1.5 subheadline">Camera Access Required</h3>
              <p className="text-white/60 text-xs mb-5 leading-relaxed">{cameraError}</p>
              <div className="flex flex-col sm:flex-row gap-2.5 w-full">
                <button
                  onClick={startWebcam}
                  className="tap-feedback flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-semibold shadow-md transition"
                >
                  <RefreshCw className="w-3.5 h-3.5" /> Retry Camera
                </button>
                <button
                  onClick={() => onSourceChange('simulation')}
                  className="tap-feedback flex-1 flex items-center justify-center gap-2 px-4 py-2.5 glass-pill hover:bg-white/10 text-white rounded-xl text-xs font-semibold transition"
                >
                  <Video className="w-3.5 h-3.5 text-emerald-400" /> Switch to Sim
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Floating Apple Camera Control Strip */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3.5 bg-black/40 border-t border-white/8 backdrop-blur-md">
        <div className="flex items-center gap-2.5 text-xs text-white/60 font-medium">
          <span className="mono-metric text-[11px] tracking-wider text-white/40">FEED:</span>
          {/* Apple Segmented Switcher */}
          <div className="inline-flex rounded-2xl bg-black/50 p-1 border border-white/8 shadow-inner">
            <button
              onClick={() => onSourceChange('webcam')}
              className={`tap-feedback flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-medium transition-all ${
                videoSourceType === 'webcam'
                  ? 'bg-white/20 text-white shadow-sm border-t border-white/20 font-semibold'
                  : 'text-white/60 hover:text-white hover:bg-white/5'
              }`}
            >
              <Camera className="w-3.5 h-3.5" />
              <span>Webcam</span>
            </button>
            <button
              onClick={() => onSourceChange('simulation')}
              className={`tap-feedback flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-medium transition-all ${
                videoSourceType === 'simulation'
                  ? 'bg-white/20 text-white shadow-sm border-t border-white/20 font-semibold'
                  : 'text-white/60 hover:text-white hover:bg-white/5'
              }`}
            >
              <Video className="w-3.5 h-3.5" />
              <span>Drone Sim</span>
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          <label className="tap-feedback glass-pill hover:bg-white/10 text-white/90 px-3.5 py-1.5 rounded-2xl text-xs font-medium cursor-pointer transition flex items-center gap-1.5 shadow-sm">
            <Upload className="w-3.5 h-3.5 text-emerald-400" />
            <span>Upload Flight Video</span>
            <input
              type="file"
              accept="video/*,image/*"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>
        </div>
      </div>
    </div>
  );
};

