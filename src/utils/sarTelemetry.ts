/**
 * Drone Search & Rescue (SAR) Telemetry & Geolocation Ray-Casting
 */

export interface DroneTelemetry {
  altitudeAGL: number; // meters above ground level
  headingDeg: number;   // compass heading (0-360)
  pitchDeg: number;     // camera pitch (-90 nadir, 0 horizon)
  droneLat: number;
  droneLon: number;
  batteryPercent: number;
}

export interface GeotaggedTarget {
  id: string;
  className: string;
  confidence: number;
  latitude: number;
  longitude: number;
  estimatedDistanceMeters: number;
  timestamp: string;
}

// Default reference position (e.g., Disaster SAR staging area)
const BASE_LAT = 34.052234;
const BASE_LON = -118.243685;

export function getSimulatedTelemetry(tick: number): DroneTelemetry {
  // Slight oscillation to mimic drone wind drift and hovering
  const altJitter = Math.sin(tick * 0.05) * 1.5;
  const headingJitter = Math.cos(tick * 0.03) * 3;

  return {
    altitudeAGL: 45.0 + altJitter,
    headingDeg: Math.round((135 + headingJitter + 360) % 360),
    pitchDeg: -45, // 45 degree oblique camera angle
    droneLat: BASE_LAT + Math.sin(tick * 0.01) * 0.0005,
    droneLon: BASE_LON + Math.cos(tick * 0.01) * 0.0005,
    batteryPercent: Math.max(15, Math.round(98 - tick * 0.02)),
  };
}

/**
 * Monocular Ray-Casting Projection:
 * Maps 2D bounding box center [x, y] to estimated ground GPS coordinates.
 */
export function calculateTargetGPS(
  bbox: [number, number, number, number],
  frameWidth: number,
  frameHeight: number,
  telemetry: DroneTelemetry
): { lat: number; lon: number; distanceMeters: number } {
  const [x, y, w, h] = bbox;
  const cx = x + w / 2;
  const cy = y + h / 2;

  // Normalized coordinates from optical center (-1 to +1)
  const normX = (cx - frameWidth / 2) / (frameWidth / 2);
  const normY = (cy - frameHeight / 2) / (frameHeight / 2);

  // Field of view assumptions (Standard drone 84 degree HFOV)
  const hfovRad = (84 * Math.PI) / 180;
  const vfovRad = hfovRad * (frameHeight / frameWidth);

  const angleX = normX * (hfovRad / 2);
  const angleY = normY * (vfovRad / 2);

  // Approximate ground intersection distance using pitch and altitude
  const totalPitchRad = Math.abs((telemetry.pitchDeg * Math.PI) / 180) - angleY;
  const groundDistance = telemetry.altitudeAGL / Math.tan(Math.max(0.1, totalPitchRad));

  // Compute bearing to target
  const targetBearingRad = ((telemetry.headingDeg * Math.PI) / 180) + angleX;

  // Earth radius in meters
  const R = 6378137.0;
  const dNorth = groundDistance * Math.cos(targetBearingRad);
  const dEast = groundDistance * Math.sin(targetBearingRad);

  const deltaLat = (dNorth / R) * (180 / Math.PI);
  const deltaLon = (dEast / (R * Math.cos((telemetry.droneLat * Math.PI) / 180))) * (180 / Math.PI);

  return {
    lat: Number((telemetry.droneLat + deltaLat).toFixed(6)),
    lon: Number((telemetry.droneLon + deltaLon).toFixed(6)),
    distanceMeters: Math.round(groundDistance),
  };
}

export function exportGeoJSON(targets: GeotaggedTarget[]): string {
  const geojson = {
    type: 'FeatureCollection',
    metadata: {
      generatedBy: 'AeroVision SAR Drone Detector',
      timestamp: new Date().toISOString(),
      totalTargets: targets.length,
    },
    features: targets.map(t => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [t.longitude, t.latitude],
      },
      properties: {
        id: t.id,
        classification: t.className,
        confidence: `${(t.confidence * 100).toFixed(1)}%`,
        distanceMeters: t.estimatedDistanceMeters,
        detectedAt: t.timestamp,
        status: 'VERIFICATION_REQUIRED',
      },
    })),
  };
  return JSON.stringify(geojson, null, 2);
}
