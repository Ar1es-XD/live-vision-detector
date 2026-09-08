import React, { useEffect, useRef } from 'react';
import { Detection } from '../utils/detector';
import { DroneTelemetry, calculateTargetGPS } from '../utils/sarTelemetry';

interface TacticalHUDProps {
  detections: Detection[];
  telemetry: DroneTelemetry;
  sarMode: boolean;
  videoWidth: number;
  videoHeight: number;
  fps: number;
}

export const TacticalHUD: React.FC<TacticalHUDProps> = ({
  detections,
  telemetry,
  sarMode,
  videoWidth,
  videoHeight,
  fps,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || videoWidth === 0 || videoHeight === 0) return;

    canvas.width = videoWidth;
    canvas.height = videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw Grid Overlay & Crosshairs in SAR Mode
    if (sarMode) {
      // Vignette & Scanlines
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.15)';
      ctx.lineWidth = 1;

      // Center reticle
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 40, 0, 2 * Math.PI);
      ctx.moveTo(cx - 60, cy);
      ctx.lineTo(cx + 60, cy);
      ctx.moveTo(cx, cy - 60);
      ctx.lineTo(cx, cy + 60);
      ctx.stroke();

      // Corner flight telemetry brackets
      const bracketSize = 30;
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
      ctx.lineWidth = 2;
      // Top-Left
      ctx.strokeRect(20, 20, bracketSize, bracketSize);
      // Top-Right
      ctx.strokeRect(canvas.width - 20 - bracketSize, 20, bracketSize, bracketSize);
      // Bottom-Left
      ctx.strokeRect(20, canvas.height - 20 - bracketSize, bracketSize, bracketSize);
      // Bottom-Right
      ctx.strokeRect(canvas.width - 20 - bracketSize, canvas.height - 20 - bracketSize, bracketSize, bracketSize);
    }

    // 2. Render Detections
    detections.forEach((det, idx) => {
      const [x, y, w, h] = det.bbox;
      const isPerson = det.class.toLowerCase() === 'person';
      
      // Determine colors
      let boxColor = '#3b82f6'; // blue default
      let labelBg = 'rgba(59, 130, 246, 0.85)';
      
      if (sarMode) {
        if (isPerson) {
          boxColor = '#ef4444'; // Red alert for human in distress
          labelBg = 'rgba(239, 68, 68, 0.9)';
        } else {
          boxColor = '#f59e0b'; // Amber for vehicles/assets
          labelBg = 'rgba(245, 158, 11, 0.85)';
        }
      } else {
        if (isPerson) boxColor = '#10b981';
      }

      // Draw Main Bounding Box
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = sarMode && isPerson ? 3 : 2;
      ctx.strokeRect(x, y, w, h);

      // Corner accents for high-tech HUD look
      const cornerLen = Math.min(15, w / 4, h / 4);
      ctx.lineWidth = 3;
      ctx.beginPath();
      // Top-Left corner
      ctx.moveTo(x, y + cornerLen); ctx.lineTo(x, y); ctx.lineTo(x + cornerLen, y);
      // Top-Right corner
      ctx.moveTo(x + w - cornerLen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cornerLen);
      // Bottom-Left corner
      ctx.moveTo(x, y + h - cornerLen); ctx.lineTo(x, y + h); ctx.lineTo(x + cornerLen, y + h);
      // Bottom-Right corner
      ctx.moveTo(x + w - cornerLen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cornerLen);
      ctx.stroke();

      // Calculate Target Telemetry (GPS & Distance)
      const targetLoc = calculateTargetGPS([x, y, w, h], canvas.width, canvas.height, telemetry);

      // Label Formatting
      const percentStr = `${Math.round(det.score * 100)}%`;
      const titleText = isPerson && sarMode 
        ? `🚨 VICTIM #${idx + 1} (${percentStr})` 
        : `${det.class.toUpperCase()} ${percentStr}`;
      
      ctx.font = 'bold 12px "SF Pro Display", monospace, sans-serif';
      const textMetrics = ctx.measureText(titleText);
      const bgWidth = textMetrics.width + 12;
      const bgHeight = 22;

      // Draw Label Background
      ctx.fillStyle = labelBg;
      ctx.fillRect(x, Math.max(0, y - bgHeight), bgWidth, bgHeight);

      // Draw Text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(titleText, x + 6, Math.max(16, y - 6));

      // SAR Mode Target Crosshair & Geolocation Badge
      if (sarMode && isPerson) {
        const targetCenterX = x + w / 2;
        const targetCenterY = y + h / 2;

        // Target Reticle
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(targetCenterX, targetCenterY, 14, 0, 2 * Math.PI);
        ctx.moveTo(targetCenterX - 20, targetCenterY);
        ctx.lineTo(targetCenterX + 20, targetCenterY);
        ctx.moveTo(targetCenterX, targetCenterY - 20);
        ctx.lineTo(targetCenterX, targetCenterY + 20);
        ctx.stroke();

        // Target GPS Tag below box
        const gpsTag = `GPS: ${targetLoc.lat.toFixed(5)}°, ${targetLoc.lon.toFixed(5)}° | ~${targetLoc.distanceMeters}m`;
        ctx.font = '11px monospace';
        const gpsMetrics = ctx.measureText(gpsTag);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
        ctx.fillRect(x, y + h + 4, gpsMetrics.width + 10, 18);
        ctx.fillStyle = '#fca5a5';
        ctx.fillText(gpsTag, x + 5, y + h + 17);
      }
    });

    // 3. Top Tactical Telemetry HUD Strip
    if (sarMode) {
      ctx.fillStyle = 'rgba(10, 14, 20, 0.85)';
      ctx.fillRect(0, 0, canvas.width, 36);
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.3)';
      ctx.beginPath();
      ctx.moveTo(0, 36);
      ctx.lineTo(canvas.width, 36);
      ctx.stroke();

      ctx.fillStyle = '#10b981';
      ctx.font = 'bold 12px monospace';
      ctx.fillText(`UAV ALT: ${telemetry.altitudeAGL.toFixed(1)}m AGL`, 16, 23);
      ctx.fillText(`HDG: ${telemetry.headingDeg}°`, 170, 23);
      ctx.fillText(`DRONE GPS: ${telemetry.droneLat.toFixed(5)}°, ${telemetry.droneLon.toFixed(5)}°`, 280, 23);
      ctx.fillText(`FPS: ${Math.round(fps)}`, canvas.width - 150, 23);
      ctx.fillText(`BATT: ${telemetry.batteryPercent}%`, canvas.width - 70, 23);
    }
  }, [detections, telemetry, sarMode, videoWidth, videoHeight, fps]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none z-10"
    />
  );
};
