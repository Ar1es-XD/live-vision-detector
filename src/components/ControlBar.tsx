import React from 'react';
import { FilterCategory } from '../utils/detector';
import { Detection } from '../utils/detector';
import { DroneTelemetry, exportGeoJSON, GeotaggedTarget, calculateTargetGPS } from '../utils/sarTelemetry';
import { ShieldAlert, Crosshair, Users, Sliders, Download, Eye } from 'lucide-react';

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
    <div className="flex flex-col gap-4 p-5 bg-gray-900 border border-gray-800 rounded-2xl shadow-xl">
      {/* Top Row: Primary Mode Toggles */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* SAR Drone Mode Switch */}
        <button
          onClick={onSarModeToggle}
          className={`flex items-center gap-2.5 px-5 py-2.5 rounded-xl font-bold text-sm tracking-wide transition shadow-lg ${
            sarMode
              ? 'bg-red-600 hover:bg-red-500 text-white shadow-red-900/30 ring-2 ring-red-400'
              : 'bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700'
          }`}
        >
          <ShieldAlert className={`w-5 h-5 ${sarMode ? 'animate-bounce' : 'text-gray-400'}`} />
          <span>{sarMode ? 'SAR DRONE MODE: ACTIVE' : 'ACTIVATE SAR DRONE MODE'}</span>
        </button>

        {/* Category Filter Selector */}
        <div className="flex items-center gap-1.5 p-1 bg-gray-950 border border-gray-800 rounded-xl">
          <button
            onClick={() => onFilterChange('all')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              filterCategory === 'all'
                ? 'bg-emerald-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Eye className="w-3.5 h-3.5" /> All 80 Objects
          </button>
          <button
            onClick={() => onFilterChange('sar_essentials')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              filterCategory === 'sar_essentials'
                ? 'bg-amber-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Crosshair className="w-3.5 h-3.5" /> SAR Essentials
          </button>
          <button
            onClick={() => onFilterChange('person_only')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              filterCategory === 'person_only'
                ? 'bg-red-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Users className="w-3.5 h-3.5" /> Humans Only
          </button>
        </div>

        {/* Export Button */}
        <button
          onClick={handleExportIncident}
          disabled={currentDetections.length === 0}
          className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed text-gray-200 border border-gray-700 rounded-xl text-xs font-semibold transition"
          title="Download GeoJSON Incident Report for ATAK / GIS Mapping"
        >
          <Download className="w-4 h-4 text-emerald-400" />
          <span>Export GeoJSON</span>
        </button>
      </div>

      {/* Middle Row: Confidence Slider & Live Inventory */}
      <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-gray-800">
        {/* Confidence Threshold */}
        <div className="flex items-center gap-3 min-w-[260px]">
          <Sliders className="w-4 h-4 text-emerald-400" />
          <span className="text-xs font-medium text-gray-300">
            CONFIDENCE THRESHOLD:
          </span>
          <input
            type="range"
            min="0.2"
            max="0.85"
            step="0.05"
            value={confidence}
            onChange={e => onConfidenceChange(parseFloat(e.target.value))}
            className="w-28 accent-emerald-500 cursor-pointer"
          />
          <span className="text-xs font-bold font-mono text-emerald-400">
            {Math.round(confidence * 100)}%
          </span>
        </div>

        {/* Real-time Detections Count Pills */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-gray-500 font-mono">TARGETS IN FOV:</span>
          {Object.keys(classCounts).length === 0 ? (
            <span className="text-xs text-gray-600 italic">No targets acquired</span>
          ) : (
            Object.entries(classCounts).map(([cls, count]) => (
              <span
                key={cls}
                className={`px-2.5 py-1 rounded-md text-xs font-bold font-mono ${
                  cls.toLowerCase() === 'person'
                    ? 'bg-red-950 text-red-400 border border-red-800'
                    : 'bg-gray-800 text-gray-300 border border-gray-700'
                }`}
              >
                {cls.toUpperCase()}: {count}
              </span>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
