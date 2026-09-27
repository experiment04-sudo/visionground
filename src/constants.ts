export const PALETTE = [
  "#00d9ff", // Cyan
  "#10b981", // Emerald
  "#f59e0b", // Amber
  "#ef4444", // Crimson
  "#a855f7", // Purple
  "#ec4899", // Pink
  "#3b82f6", // Electric Blue
  "#84cc16", // Lime
  "#14b8a6", // Teal
  "#f97316", // Orange
];

export const PERMISSION_TROUBLESHOOTING = `Camera Access Denied or Device Unavailable

Please follow these operating system & browser steps to resolve:

1. Browser Camera Permissions:
   - Click the camera / lock icon in your browser address bar.
   - Set Camera to "Allow" and refresh this page.

2. Operating System Settings:
   - macOS: System Settings -> Privacy & Security -> Camera -> Allow Browser.
   - Windows: Settings -> Privacy & Security -> Camera -> Enable "Allow desktop apps to access your camera".
   - Linux: Ensure user has permissions for /dev/video0 (e.g. 'sudo usermod -a -G video $USER').

3. Hardware & Exclusivity:
   - Check if another application (Zoom, Microsoft Teams, OBS, Google Meet) is actively using the webcam.
   - Unplug and reconnect external USB webcams.`;
