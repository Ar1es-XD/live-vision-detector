import React from 'react';
import { FilterCategory } from '../utils/detector';
import { Detection } from '../utils/detector';
import { DroneTelemetry, exportGeoJSON, GeotaggedTarget, calculateTargetGPS } from '../utils/sarTelemetry';
import { ShieldAlert, Crosshair, Users, Sliders, Download, Eye, Sparkles } from 'lucide-react';

interface ControlBarProps {
  confidence: number;
  onConfidenceChange: (conf: number) => void;
  filterCategory: FilterCategory;
  onFilterChange: (filter: FilterCategory) => void;
  sarMode: boolean;
  onSarModeToggle: () => void;
  currentDetections: Detection[];
  currentTelemetry: DroneTelemetry;
}

export const ControlBar: React.FC<ControlBarProps> = ({
  confidence,
  onConfidenceChange,
  filterCategory,
  onFilterChange,
  sarMode,
  onSarModeToggle,
  currentDetections,
  currentTelemetry,
}) => {
  // Export Incident Report
  const handleExportIncident = () => {
    const targets: GeotaggedTarget[] = currentDetections.map((d, i) => {
      const loc = calculateTargetGPS(d.bbox, 1280, 720, currentTelemetry);
      return {
        id: `TARGET-${Date.now()}-${i + 1}`,
        className: d.class,
        confidence: d.score,
        latitude: loc.lat,
        longitude: loc.lon,
        estimatedDistanceMeters: loc.distanceMeters,
        timestamp: new Date().toISOString(),
      };
    });

    const geojsonStr = exportGeoJSON(targets);
    const blob = new Blob([geojsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sar_incident_report_${Date.now()}.geojson`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Group current detections by class name
  const classCounts = currentDetections.reduce((acc, d) => {
    acc[d.class] = (acc[d.class] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="glass-surface-elevated rounded-3xl p-5 flex flex-col gap-4.5 transition-all">
      {/* Primary Tactical Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3.5">
        {/* SAR Drone Mode Toggle */}
        <button
          onClick={onSarModeToggle}
          className={`tap-feedback relative flex items-center gap-2.5 px-5 py-2.5 rounded-2xl font-semibold text-xs tracking-wide transition-all shadow-md ${
            sarMode
              ? 'bg-gradient-to-r from-red-600 to-rose-600 text-white shadow-red-950/50 ring-1 ring-white/30'
              : 'glass-pill hover:bg-white/10 text-white/90'
          }`}
        >
          <span className={`w-2 h-2 rounded-full ${sarMode ? 'bg-white animate-ping' : 'bg-red-400'}`} />
          <ShieldAlert className="w-4 h-4" />
          <span>{sarMode ? 'SAR TACTICAL MODE: ACTIVE' : 'ACTIVATE SAR DRONE MODE'}</span>
        </button>

        {/* Apple Segmented Control - Filter Mode */}
        <div className="flex items-center p-1 rounded-2xl bg-black/40 border border-white/8 backdrop-blur-md shadow-inner">
          <button
            onClick={() => onFilterChange('all')}
            className={`tap-feedback flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-medium transition-all ${
              filterCategory === 'all'
                ? 'bg-white/20 text-white shadow-sm backdrop-blur-md border-t border-white/25 font-semibold'
                : 'text-white/60 hover:text-white hover:bg-white/5'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>All 80 Classes</span>
          </button>
          <button
            onClick={() => onFilterChange('sar_essentials')}
            className={`tap-feedback flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-medium transition-all ${
              filterCategory === 'sar_essentials'
                ? 'bg-amber-500/25 text-amber-200 border border-amber-400/30 shadow-sm font-semibold'
                : 'text-white/60 hover:text-white hover:bg-white/5'
            }`}
          >
            <Crosshair className="w-3.5 h-3.5 text-amber-400" />
            <span>SAR Essentials</span>
          </button>
          <button
            onClick={() => onFilterChange('person_only')}
            className={`tap-feedback flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-medium transition-all ${
              filterCategory === 'person_only'
                ? 'bg-red-500/25 text-red-200 border border-red-400/30 shadow-sm font-semibold'
                : 'text-white/60 hover:text-white hover:bg-white/5'
            }`}
          >
            <Users className="w-3.5 h-3.5 text-red-400" />
            <span>Survivors Only</span>
          </button>
        </div>

        {/* Export GeoJSON Incident Report */}
        <button
          onClick={handleExportIncident}
          disabled={currentDetections.length === 0}
          className="tap-feedback glass-pill hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed text-white/90 px-4 py-2 rounded-2xl text-xs font-semibold flex items-center gap-2 transition"
          title="Download GeoJSON Incident Report for ATAK / GIS Mapping"
        >
          <Download className="w-3.5 h-3.5 text-emerald-400" />
          <span>Export GeoJSON</span>
        </button>
      </div>

      {/* Secondary Controls: Scrub Confidence & Live Telemetry Pills */}
      <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-white/8">
        {/* Apple Style Precision Confidence Slider */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs text-white/70 font-medium">
            <Sliders className="w-3.5 h-3.5 text-emerald-400" />
            <span>CONFIDENCE</span>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min="0.2"
              max="0.85"
              step="0.05"
              value={confidence}
              onChange={e => onConfidenceChange(parseFloat(e.target.value))}
              className="apple-slider w-28 sm:w-36"
            />
            <span className="mono-metric text-xs font-semibold text-emerald-400 min-w-[38px]">
              {Math.round(confidence * 100)}%
            </span>
          </div>
        </div>

        {/* Live Targets In FOV Chips */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="mono-metric text-[11px] text-white/40 tracking-wider">
            ACQUIRED TARGETS:
          </span>
          {Object.keys(classCounts).length === 0 ? (
            <span className="text-xs text-white/35 italic flex items-center gap-1">
              <Sparkles className="w-3 h-3 text-white/20" /> No target in view
            </span>
          ) : (
            Object.entries(classCounts).map(([cls, count]) => {
              const isPerson = cls.toLowerCase() === 'person';
              return (
                <span
                  key={cls}
                  className={`mono-metric px-2.5 py-1 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-all ${
                    isPerson
                      ? 'bg-red-500/20 text-red-200 border border-red-500/40 shadow-sm shadow-red-950/40'
                      : 'glass-pill text-white/90'
                  }`}
                >
                  <span className={`w-1.5 h-1.5 rounded-full ${isPerson ? 'bg-red-400 animate-ping' : 'bg-emerald-400'}`} />
                  {cls.toUpperCase()} <span className="text-white/60">•</span> {count}
                </span>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};

