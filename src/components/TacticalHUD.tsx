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

    const dpr = window.devicePixelRatio || 1;
    // Set internal canvas resolution scaled by DPR for sharp Retina rendering
    canvas.width = videoWidth * dpr;
    canvas.height = videoHeight * dpr;
    canvas.style.width = '100%';
    canvas.style.height = '100%';

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, videoWidth, videoHeight);

    // 1. Apple-style Viewfinder Brackets & Center Crosshair in SAR Mode
    if (sarMode) {
      const cx = videoWidth / 2;
      const cy = videoHeight / 2;

      // Soft Optical Center Reticle
      ctx.strokeStyle = 'rgba(52, 211, 153, 0.28)'; // Emerald tint
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, 32, 0, 2 * Math.PI);
      ctx.stroke();

      ctx.strokeStyle = 'rgba(52, 211, 153, 0.4)';
      ctx.beginPath();
      // Tick marks
      ctx.moveTo(cx - 48, cy); ctx.lineTo(cx - 36, cy);
      ctx.moveTo(cx + 36, cy); ctx.lineTo(cx + 48, cy);
      ctx.moveTo(cx, cy - 48); ctx.lineTo(cx, cy - 36);
      ctx.moveTo(cx, cy + 36); ctx.lineTo(cx, cy + 48);
      // Tiny center dot
      ctx.arc(cx, cy, 1.5, 0, 2 * Math.PI);
      ctx.stroke();

      // Viewfinder 4-corner brackets
      const bLen = 28;
      const m = 20; // margin
      ctx.strokeStyle = 'rgba(52, 211, 153, 0.5)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      // Top-Left
      ctx.moveTo(m, m + bLen); ctx.lineTo(m, m); ctx.lineTo(m + bLen, m);
      // Top-Right
      ctx.moveTo(videoWidth - m - bLen, m); ctx.lineTo(videoWidth - m, m); ctx.lineTo(videoWidth - m, m + bLen);
      // Bottom-Left
      ctx.moveTo(m, videoHeight - m - bLen); ctx.lineTo(m, videoHeight - m); ctx.lineTo(m + bLen, videoHeight - m);
      // Bottom-Right
      ctx.moveTo(videoWidth - m - bLen, videoHeight - m); ctx.lineTo(videoWidth - m, videoHeight - m); ctx.lineTo(videoWidth - m, videoHeight - m - bLen);
      ctx.stroke();
    }

    // 2. Render Optical Detections
    detections.forEach((det, idx) => {
      const [x, y, w, h] = det.bbox;
      const isPerson = det.class.toLowerCase() === 'person';

      // Accent palette: Apple Human Interface tone mapping
      let strokeColor = 'rgba(96, 165, 250, 0.9)'; // Blue
      let badgeBg = 'rgba(30, 58, 138, 0.85)';
      let glowColor = 'rgba(59, 130, 246, 0.4)';

      if (sarMode) {
        if (isPerson) {
          strokeColor = 'rgba(239, 68, 68, 0.95)'; // Vivid Red for distress victim
          badgeBg = 'rgba(153, 27, 27, 0.92)';
          glowColor = 'rgba(239, 68, 68, 0.5)';
        } else {
          strokeColor = 'rgba(251, 191, 36, 0.9)'; // Amber for support assets
          badgeBg = 'rgba(180, 83, 9, 0.88)';
          glowColor = 'rgba(245, 158, 11, 0.4)';
        }
      } else if (isPerson) {
        strokeColor = 'rgba(52, 211, 153, 0.95)'; // Emerald
        badgeBg = 'rgba(6, 95, 70, 0.88)';
        glowColor = 'rgba(16, 185, 129, 0.4)';
      }

      // Smooth Rounded Bounding Box
      ctx.save();
      ctx.shadowColor = glowColor;
      ctx.shadowBlur = sarMode && isPerson ? 10 : 4;
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = sarMode && isPerson ? 2.5 : 1.75;

      const cornerRadius = 6;
      if (typeof ctx.roundRect === 'function') {
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, cornerRadius);
        ctx.stroke();
      } else {
        ctx.strokeRect(x, y, w, h);
      }
      ctx.restore();

      // Precision Corner Reticles
      const cSize = Math.min(14, w / 4, h / 4);
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 3;
      ctx.beginPath();
      // Top-Left
      ctx.moveTo(x, y + cSize); ctx.lineTo(x, y); ctx.lineTo(x + cSize, y);
      // Top-Right
      ctx.moveTo(x + w - cSize, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cSize);
      // Bottom-Left
      ctx.moveTo(x, y + h - cSize); ctx.lineTo(x, y + h); ctx.lineTo(x + cSize, y + h);
      // Bottom-Right
      ctx.moveTo(x + w - cSize, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cSize);
      ctx.stroke();

      // Monocular Telemetry Calculation
      const targetLoc = calculateTargetGPS([x, y, w, h], videoWidth, videoHeight, telemetry);

      // Label Badge with Apple-style Typography
      const percentStr = `${Math.round(det.score * 100)}%`;
      const labelText = isPerson && sarMode
        ? `🚨 VICTIM #${idx + 1} • ${percentStr}`
        : `${det.class.toUpperCase()} • ${percentStr}`;

      ctx.font = '600 11px -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif';
      const labelWidth = ctx.measureText(labelText).width + 16;
      const labelHeight = 22;
      const labelY = Math.max(4, y - labelHeight - 2);

      // Translucent Glass Tag
      ctx.fillStyle = badgeBg;
      if (typeof ctx.roundRect === 'function') {
        ctx.beginPath();
        ctx.roundRect(x, labelY, labelWidth, labelHeight, 5);
        ctx.fill();
      } else {
        ctx.fillRect(x, labelY, labelWidth, labelHeight);
      }

      // Top light edge highlight
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + 2, labelY + 1);
      ctx.lineTo(x + labelWidth - 2, labelY + 1);
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x + 8, labelY + 15);

      // SAR Victim Pin & GPS Telemetry Plaque
      if (sarMode && isPerson) {
        const tX = x + w / 2;
        const tY = y + h / 2;

        // Pulsing Target Lock Ring
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.9)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(tX, tY, 16, 0, 2 * Math.PI);
        ctx.moveTo(tX - 22, tY); ctx.lineTo(tX + 22, tY);
        ctx.moveTo(tX, tY - 22); ctx.lineTo(tX, tY + 22);
        ctx.stroke();

        // Target GPS Tag below box
        const geoTag = `GPS ${targetLoc.lat.toFixed(5)}°, ${targetLoc.lon.toFixed(5)}° • ~${targetLoc.distanceMeters}m`;
        ctx.font = '500 10px "SF Mono", Menlo, Consolas, monospace';
        const geoWidth = ctx.measureText(geoTag).width + 14;
        const geoHeight = 18;
        const geoY = y + h + 6;

        ctx.fillStyle = 'rgba(15, 17, 23, 0.88)';
        if (typeof ctx.roundRect === 'function') {
          ctx.beginPath();
          ctx.roundRect(x, geoY, geoWidth, geoHeight, 4);
          ctx.fill();
        } else {
          ctx.fillRect(x, geoY, geoWidth, geoHeight);
        }

        ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = '#fca5a5';
        ctx.fillText(geoTag, x + 7, geoY + 13);
      }
    });

    // 3. Apple-style Top Telemetry Glass Strip (SAR Mode)
    if (sarMode) {
      const stripHeight = 34;
      ctx.fillStyle = 'rgba(10, 14, 20, 0.75)';
      ctx.fillRect(0, 0, videoWidth, stripHeight);

      // Bottom light border
      ctx.strokeStyle = 'rgba(52, 211, 153, 0.25)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, stripHeight);
      ctx.lineTo(videoWidth, stripHeight);
      ctx.stroke();

      ctx.font = '600 11px "SF Mono", Menlo, Consolas, monospace';
      ctx.fillStyle = 'rgba(52, 211, 153, 0.95)';
      ctx.fillText(`ALT ${telemetry.altitudeAGL.toFixed(1)}m AGL`, 16, 21);
      ctx.fillText(`HDG ${telemetry.headingDeg}°`, 140, 21);
      ctx.fillText(`UAV ${telemetry.droneLat.toFixed(5)}°, ${telemetry.droneLon.toFixed(5)}°`, 230, 21);

      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.fillText(`FPS ${Math.round(fps)}`, videoWidth - 145, 21);
      ctx.fillText(`BAT ${telemetry.batteryPercent}%`, videoWidth - 75, 21);
    }

    ctx.restore();
  }, [detections, telemetry, sarMode, videoWidth, videoHeight, fps]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none z-10"
    />
  );
};

