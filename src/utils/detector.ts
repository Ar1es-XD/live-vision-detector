import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';

export interface Detection {
  bbox: [number, number, number, number]; // [x, y, width, height]
  class: string;
  score: number;
}

export type FilterCategory = 'all' | 'sar_essentials' | 'person_only';

export const SAR_ESSENTIAL_CLASSES = new Set([
  'person',
  'boat',
  'car',
  'truck',
  'bus',
  'backpack',
  'suitcase',
  'cell phone',
  'dog',
  'horse'
]);

let modelInstance: cocoSsd.ObjectDetection | null = null;
let isLoading = false;

export async function initModel(onStatus?: (status: string) => void): Promise<cocoSsd.ObjectDetection> {
  if (modelInstance) return modelInstance;
  if (isLoading) {
    while (isLoading) {
      await new Promise(r => setTimeout(r, 100));
    }
    if (modelInstance) return modelInstance;
  }

  isLoading = true;
  try {
    onStatus?.('Initializing WebGL GPU accelerator...');
    await tf.ready();
    
    // Attempt WebGL, fallback gracefully to CPU if not supported
    try {
      await tf.setBackend('webgl');
      console.log('[Vision Engine] Using WebGL hardware acceleration');
    } catch {
      await tf.setBackend('cpu');
      console.warn('[Vision Engine] WebGL unavailable, falling back to CPU');
    }

    onStatus?.('Loading COCO-SSD Neural Network (MobileNet V2)...');
    modelInstance = await cocoSsd.load({
      base: 'mobilenet_v2',
    });
    
    onStatus?.('Vision AI Ready');
    return modelInstance;
  } finally {
    isLoading = false;
  }
}

export async function detectFrame(
  model: cocoSsd.ObjectDetection,
  source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement,
  minScore: number = 0.35,
  filter: FilterCategory = 'all'
): Promise<Detection[]> {
  if (!model || !source) return [];

  try {
    const rawDetections = await model.detect(source, 20, minScore);

    return rawDetections
      .map(d => ({
        bbox: d.bbox as [number, number, number, number],
        class: d.class,
        score: d.score,
      }))
      .filter(d => {
        if (filter === 'person_only') {
          return d.class.toLowerCase() === 'person';
        }
        if (filter === 'sar_essentials') {
          return SAR_ESSENTIAL_CLASSES.has(d.class.toLowerCase());
        }
        return true;
      });
  } catch (err) {
    console.error('Detection error:', err);
    return [];
  }
}
