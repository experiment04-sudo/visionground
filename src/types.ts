export interface Detection {
  label: string;
  box_2d: [number, number, number, number]; // [ymin, xmin, ymax, xmax] 0-1000
  confidence?: number;
  color: string;
}

export interface CameraDeviceInfo {
  deviceId: string;
  label: string;
}

export interface PythonFilesMap {
  [filename: string]: string;
}
