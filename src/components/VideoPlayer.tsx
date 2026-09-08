import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Detection, FilterCategory, detectFrame } from '../utils/detector';
import { TacticalHUD } from './TacticalHUD';
import { DroneTelemetry, getSimulatedTelemetry } from '../utils/sarTelemetry';
import { Camera, Video, Upload, AlertCircle, RefreshCw } from 'lucide-react';

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

  // 3. Procedural Aerial Drone Simulation Loop (when camera is disabled/in simulation mode)
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

      // Draw synthetic disaster flood / wilderness terrain
      ctx.fillStyle = '#1e293b'; // Muddy terrain
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Floodwater body
      ctx.fillStyle = '#0f3a53';
      ctx.beginPath();
      ctx.ellipse(640, 360, 480, 260, 0.2, 0, Math.PI * 2);
      ctx.fill();

      // Water ripples
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)';
      ctx.lineWidth = 2;
      for (let i = 0; i < 4; i++) {
        const r = ((simTick * 2 + i * 80) % 300) + 50;
        ctx.beginPath();
        ctx.arc(640, 360, r, 0, Math.PI * 2);
        ctx.stroke();
      }

      // 1. Simulated stranded civilian on rooftop/island (Person)
      const personX = 580 + Math.sin(simTick * 0.02) * 15;
      const personY = 320 + Math.cos(simTick * 0.02) * 10;
      // Body
      ctx.fillStyle = '#ef4444'; // Orange/Red jacket
      ctx.beginPath();
      ctx.arc(personX, personY, 18, 0, Math.PI * 2);
      ctx.fill();
      // Head
      ctx.fillStyle = '#fde047';
      ctx.beginPath();
      ctx.arc(personX, personY - 8, 8, 0, Math.PI * 2);
      ctx.fill();

      // 2. Simulated rescue boat
      const boatX = 350 + (simTick * 1.2) % 600;
      const boatY = 420 + Math.sin(simTick * 0.05) * 12;
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.ellipse(boatX, boatY, 35, 14, -0.1, 0, Math.PI * 2);
      ctx.fill();

      // 3. Simulated vehicle on embankment (Car)
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
    <div className="relative w-full rounded-2xl overflow-hidden border border-gray-800 bg-black shadow-2xl">
      {/* Video / Canvas Ingestion */}
      <div ref={containerRef} className="relative aspect-video w-full bg-black flex items-center justify-center">
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

        {/* Tactical HUD Overlay Layer */}
        <TacticalHUD
          detections={detections}
          telemetry={telemetry}
          sarMode={sarMode}
          videoWidth={streamDimensions.width}
          videoHeight={streamDimensions.height}
          fps={fps}
        />

        {/* Camera Permission Warning Banner */}
        {cameraError && videoSourceType === 'webcam' && (
          <div className="absolute inset-0 bg-gray-950/90 flex flex-col items-center justify-center p-6 text-center z-30">
            <AlertCircle className="w-14 h-14 text-amber-500 mb-3 animate-pulse" />
            <h3 className="text-xl font-bold text-white mb-2">Camera Access Notice</h3>
            <p className="text-gray-400 max-w-md mb-5 text-sm">{cameraError}</p>
            <div className="flex gap-3">
              <button
                onClick={startWebcam}
                className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-sm font-semibold transition"
              >
                <RefreshCw className="w-4 h-4" /> Retry Camera
              </button>
              <button
                onClick={() => onSourceChange('simulation')}
                className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-200 rounded-lg text-sm font-semibold transition border border-gray-700"
              >
                <Video className="w-4 h-4 text-emerald-400" /> Switch to Drone Simulation
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Source Switcher Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-gray-900 border-t border-gray-800">
        <div className="flex items-center gap-2 text-xs font-medium text-gray-400">
          <span>INPUT FEED:</span>
          <div className="inline-flex rounded-lg bg-gray-950 p-1 border border-gray-800">
            <button
              onClick={() => onSourceChange('webcam')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition ${
                videoSourceType === 'webcam'
                  ? 'bg-emerald-600 text-white shadow-sm'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Camera className="w-3.5 h-3.5" /> Live Webcam
            </button>
            <button
              onClick={() => onSourceChange('simulation')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition ${
                videoSourceType === 'simulation'
                  ? 'bg-emerald-600 text-white shadow-sm'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Video className="w-3.5 h-3.5" /> Aerial Drone Sim
            </button>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-xs font-medium cursor-pointer transition border border-gray-700">
            <Upload className="w-3.5 h-3.5 text-emerald-400" />
            <span>Upload Drone Clip</span>
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
