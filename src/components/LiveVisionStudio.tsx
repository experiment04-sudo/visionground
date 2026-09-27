import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  Camera,
  Play,
  Square,
  Sparkles,
  Download,
  Trash2,
  RefreshCw,
  Sliders,
  AlertTriangle,
  Layers,
  Activity,
  Zap,
  Clock,
  Gauge,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  XCircle,
  Cpu,
  BarChart3,
} from "lucide-react";
import { Detection, CameraDeviceInfo } from "../types";
import { PALETTE } from "../constants";
import { TroubleshootModal } from "./TroubleshootModal";

export const LiveVisionStudio: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameId = useRef<number | null>(null);

  // States
  const [devices, setDevices] = useState<CameraDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [isContinuous, setIsContinuous] = useState<boolean>(true);
  const [isInferring, setIsInferring] = useState<boolean>(false);
  const [serverBusyNotice, setServerBusyNotice] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>("Local YOLO11 Standby");
  const [aiErrorReasoning, setAiErrorReasoning] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [showTroubleshoot, setShowTroubleshoot] = useState<boolean>(false);

  // Diagnostic & Performance Dashboard states
  const [showPerfDashboard, setShowPerfDashboard] = useState<boolean>(false);
  const [activeModel, setActiveModel] = useState<string>("YOLO11 Local (PyTorch)");
  const [apiLatencyMs, setApiLatencyMs] = useState<number | null>(null);
  const [payloadBytes, setPayloadBytes] = useState<number | null>(null);
  const [dropErrorCount, setDropErrorCount] = useState<number>(0);
  const [successCount, setSuccessCount] = useState<number>(0);
  const [inferenceFrequency, setInferenceFrequency] = useState<number>(0);

  // FPS calculation and loop control refs
  const frameCountRef = useRef<number>(0);
  const lastFpsTimeRef = useRef<number>(performance.now());
  const detectionsRef = useRef<Detection[]>([]);
  const isInferringRef = useRef<boolean>(false);
  const isAnalyzingRef = useRef<boolean>(false);
  const diagnosisCooldownRef = useRef<number>(0);
  const busyUntilRef = useRef<number>(0);
  const busyTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const inferenceTimestampsRef = useRef<number[]>([]);

  // Sync ref with state
  useEffect(() => {
    detectionsRef.current = detections;
  }, [detections]);

  useEffect(() => {
    isInferringRef.current = isInferring;
  }, [isInferring]);

  // Enumerate video devices
  const enumerateDevices = useCallback(async () => {
    try {
      const mediaDevices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = mediaDevices
        .filter((d) => d.kind === "videoinput")
        .map((d, idx) => ({
          deviceId: d.deviceId,
          label: d.label || `Camera ${idx + 1}`,
        }));
      setDevices(videoInputs);
      if (videoInputs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoInputs[0].deviceId);
      }
    } catch {
      // ignore enumeration error
    }
  }, [selectedDeviceId]);

  // 1. FIX WEBCAM MOUNT RACE CONDITION: Camera initialization with cancelled boolean flag
  useEffect(() => {
    let cancelled = false;
    let activeStream: MediaStream | null = null;

    async function initCamera() {
      try {
        setErrorMessage(null);
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        activeStream = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch((err: Error) => {
            if (err.name !== "AbortError") console.error("Play error:", err);
          });
          setIsStreaming(true);
        }

        await enumerateDevices();
      } catch (err) {
        console.error("Camera access denied or failed:", err);
        const msg =
          err instanceof Error
            ? err.message
            : "Camera permission denied or camera device in use by another app.";
        setErrorMessage(msg);
        setIsStreaming(false);
        setShowTroubleshoot(true);
      }
    }

    initCamera();

    return () => {
      cancelled = true;
      activeStream?.getTracks().forEach((t) => t.stop());
      setIsStreaming(false);
      setIsContinuous(false);
    };
  }, []); // Empty array ensures single execution on mount

  // Switch camera when user selects different device
  const handleDeviceChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newId = e.target.value;
    setSelectedDeviceId(newId);
    try {
      if (videoRef.current && videoRef.current.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: newId } },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch((err: Error) => {
          if (err.name !== "AbortError") console.error("Play error:", err);
        });
        setIsStreaming(true);
      }
    } catch (err) {
      console.error("Failed to switch camera:", err);
    }
  };

  const startCamera = async (deviceId?: string) => {
    try {
      setErrorMessage(null);
      if (videoRef.current && videoRef.current.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      }
      const constraints: MediaStreamConstraints = {
        video: { deviceId: deviceId ? { exact: deviceId } : undefined },
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch((err: Error) => {
          if (err.name !== "AbortError") console.error("Play error:", err);
        });
        setIsStreaming(true);
      }
    } catch (err) {
      console.error("Retry camera error:", err);
    }
  };

  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setIsContinuous(false);
  }, []);

  // 3. Render Canvas Loop (Aspect Ratio Preserved + Bounding Boxes)
  useEffect(() => {
    let active = true;

    const render = () => {
      if (!active) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (video && canvas && video.readyState >= 2) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          // Canvas dimensions
          const cw = canvas.width;
          const ch = canvas.height;

          // Clear canvas with dark base
          ctx.fillStyle = "#090d16";
          ctx.fillRect(0, 0, cw, ch);

          const vw = video.videoWidth || 1280;
          const vh = video.videoHeight || 720;

          // Edge Case 2: Exact Letterbox / Pillarbox Scale Calculation
          const scale = Math.min(cw / vw, ch / vh);
          const rw = vw * scale;
          const rh = vh * scale;
          const ox = (cw - rw) / 2;
          const oy = (ch - rh) / 2;

          // Draw video frame
          ctx.drawImage(video, 0, 0, vw, vh, ox, oy, rw, rh);

          // Draw active bounding boxes
          const activeDetections = detectionsRef.current;
          for (let i = 0; i < activeDetections.length; i++) {
            const det = activeDetections[i];
            const [ymin, xmin, ymax, xmax] = det.box_2d;

            // Map 0-1000 coordinates to actual rendered rectangle
            const bx = ox + (xmin / 1000) * rw;
            const by = oy + (ymin / 1000) * rh;
            const bw = Math.max(2, ((xmax - xmin) / 1000) * rw);
            const bh = Math.max(2, ((ymax - ymin) / 1000) * rh);

            const color = det.color || PALETTE[i % PALETTE.length];

            // 1. Semi-transparent fill
            ctx.fillStyle = `${color}28`;
            ctx.fillRect(bx, by, bw, bh);

            // 2. Crisp boundary stroke
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.strokeRect(bx, by, bw, bh);

            // 3. Label Badge
            const labelText = det.confidence
              ? `${det.label} (${Math.round(det.confidence * 100)}%)`
              : det.label;

            ctx.font = "bold 12px 'Segoe UI', system-ui, sans-serif";
            const textMetrics = ctx.measureText(labelText);
            const padX = 7;
            const padY = 4;
            const badgeW = textMetrics.width + padX * 2;
            const badgeH = 20;

            let badgeY = by - badgeH - 2;
            if (badgeY < oy) {
              badgeY = by + 2;
            }

            // Badge background
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.roundRect(bx, badgeY, badgeW, badgeH, 4);
            ctx.fill();

            // Badge text (high-contrast black text on bright vibrant background)
            ctx.fillStyle = "#000000";
            ctx.textBaseline = "middle";
            ctx.fillText(labelText, bx + padX, badgeY + badgeH / 2);
          }

          // Frame count for FPS
          frameCountRef.current += 1;
        }
      }

      // FPS update every second
      const now = performance.now();
      if (now - lastFpsTimeRef.current >= 1000) {
        setFps(Math.round((frameCountRef.current * 1000) / (now - lastFpsTimeRef.current)));
        frameCountRef.current = 0;
        lastFpsTimeRef.current = now;
      }

      animationFrameId.current = requestAnimationFrame(render);
    };

    animationFrameId.current = requestAnimationFrame(render);

    return () => {
      active = false;
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, []);

  // Canvas renderer multiplies normalized [0..1] by canvas width/height
  const renderBoundingBoxes = useCallback((boxes: any) => { // eslint-disable-line @typescript-eslint/no-explicit-any
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cw = canvas.width || 640;
    const ch = canvas.height || 480;

    const rawList: any[] = Array.isArray(boxes) // eslint-disable-line @typescript-eslint/no-explicit-any
      ? boxes
      : boxes && typeof boxes === "object"
      ? boxes.boxes || boxes.detections || []
      : [];

    const parsedDetections: Detection[] = [];

    rawList.forEach((item, idx) => {
      if (!item) return;

      let ymin = 0;
      let xmin = 0;
      let ymax = 0;
      let xmax = 0;
      let label = "object";
      let confidence: number | undefined = undefined;

      if (Array.isArray(item)) {
        // [ymin, xmin, ymax, xmax, label?, confidence?]
        [ymin, xmin, ymax, xmax] = item;
        if (typeof item[4] === "string") label = item[4];
        if (typeof item[5] === "number") confidence = item[5];
      } else if (typeof item === "object") {
        const box =
          item.box_2d ||
          item.box ||
          item.bbox ||
          (item.ymin !== undefined ? [item.ymin, item.xmin, item.ymax, item.xmax] : null);
        if (Array.isArray(box)) {
          [ymin, xmin, ymax, xmax] = box;
        }
        label = item.label || item.name || item.class_name || "object";
        confidence =
          typeof item.confidence === "number"
            ? item.confidence
            : typeof item.conf === "number"
            ? item.conf
            : undefined;
      }

      // Normalize if in 0..1000 scale
      if (ymax > 1.0 || xmax > 1.0) {
        ymin = ymin / 1000;
        xmin = xmin / 1000;
        ymax = ymax / 1000;
        xmax = xmax / 1000;
      }

      // Clamp ratios [0..1]
      ymin = Math.max(0, Math.min(1, ymin));
      xmin = Math.max(0, Math.min(1, xmin));
      ymax = Math.max(0, Math.min(1, ymax));
      xmax = Math.max(0, Math.min(1, xmax));

      // Multiply normalized [ymin, xmin, ymax, xmax] ratios by canvas pixel dimensions
      const pixelXmin = xmin * cw;
      const pixelYmin = ymin * ch;
      const pixelXmax = xmax * cw;
      const pixelYmax = ymax * ch;

      const bx = pixelXmin;
      const by = pixelYmin;
      const bw = Math.max(2, pixelXmax - pixelXmin);
      const bh = Math.max(2, pixelYmax - pixelYmin);

      const color = PALETTE[idx % PALETTE.length];

      // Draw bounding box on canvas
      ctx.fillStyle = `${color}28`;
      ctx.fillRect(bx, by, bw, bh);

      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(bx, by, bw, bh);

      // Label badge
      const labelText =
        confidence !== undefined
          ? `${label} (${Math.round(confidence * 100)}%)`
          : label;

      ctx.font = "bold 12px 'Segoe UI', system-ui, sans-serif";
      const textMetrics = ctx.measureText(labelText);
      const padX = 6;
      const badgeW = textMetrics.width + padX * 2;
      const badgeH = 18;
      let badgeY = by - badgeH - 2;
      if (badgeY < 0) {
        badgeY = by + 2;
      }

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(bx, badgeY, badgeW, badgeH, 4);
      ctx.fill();

      ctx.fillStyle = "#000000";
      ctx.textBaseline = "middle";
      ctx.fillText(labelText, bx + padX, badgeY + badgeH / 2);

      parsedDetections.push({
        label,
        box_2d: [ymin * 1000, xmin * 1000, ymax * 1000, xmax * 1000],
        confidence,
        color,
      });
    });

    setDetections(parsedDetections);
    detectionsRef.current = parsedDetections;
  }, []);

  // Frame Capture & Local YOLO11 Inference Loop
  async function captureAndSendFrame() {
    if (isAnalyzingRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState !== 4) return;

    isAnalyzingRef.current = true;
    setIsInferring(true);
    const startTime = performance.now();

    try {
      const ctx = canvas.getContext("2d");
      canvas.width = 640;
      canvas.height = 480;
      ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);

      const base64Image = canvas.toDataURL("image/jpeg", 0.6);

      const response = await fetch("http://localhost:8000/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: base64Image }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();
      renderBoundingBoxes(data.boxes); // Canvas renderer multiplies normalized [0..1] by canvas width/height
      setStatusMessage("Local YOLO11 Active (30+ FPS)");

      const elapsed = Math.round(performance.now() - startTime);
      setLatencyMs(elapsed);
      setSuccessCount((prev) => prev + 1);
      if (data.model) setActiveModel(data.model);

      const now = Date.now();
      inferenceTimestampsRef.current.push(now);
      inferenceTimestampsRef.current = inferenceTimestampsRef.current.filter(
        (t) => now - t <= 60000
      );
      setInferenceFrequency(inferenceTimestampsRef.current.length);
    } catch (err: any) { // eslint-disable-line @typescript-eslint/no-explicit-any
      setDropErrorCount((prev) => prev + 1);
      setStatusMessage("Local server disconnected. Diagnosing...");

      // Trigger Gemini error reasoning ONLY if 10 seconds have elapsed
      if (Date.now() >= diagnosisCooldownRef.current) {
        diagnosisCooldownRef.current = Date.now() + 10000; // 10s cooldown

        fetch("http://localhost:8000/diagnose-error", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ error_message: err.message, stack_trace: err.stack }),
        })
          .then((res) => res.json())
          .then((data) => setAiErrorReasoning(data.reasoning || "Could not analyze error."))
          .catch(() => setAiErrorReasoning("Local diagnostic backend offline."));
      }
    } finally {
      isAnalyzingRef.current = false;
      setIsInferring(false);
    }
  }

  // Manual trigger button alias
  const runDetection = captureAndSendFrame;

  // 5. Continuous YOLO11 Camera Loop (~30 FPS cadence)
  useEffect(() => {
    let isMounted = true;

    async function runDetectionLoop() {
      while (isMounted) {
        if (isContinuous && isStreaming) {
          await captureAndSendFrame();
        }
        // Continuous 30+ FPS tick rate
        await new Promise((r) => setTimeout(r, 33));
      }
    }

    runDetectionLoop();

    return () => {
      isMounted = false;
    };
  }, [isContinuous, isStreaming, renderBoundingBoxes]);

  // 6. Snapshot & Export (Downloads annotated image matrix as PNG)
  const handleExportSnapshot = () => {
    const video = videoRef.current;
    if (!video || !isStreaming) return;

    const exportCanvas = document.createElement("canvas");
    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;
    exportCanvas.width = vw;
    exportCanvas.height = vh;
    const ctx = exportCanvas.getContext("2d");
    if (!ctx) return;

    // 1. Draw raw high-resolution video frame
    ctx.drawImage(video, 0, 0, vw, vh);

    // 2. Draw active bounding boxes scaled to full resolution
    const currentDetections = detectionsRef.current;
    for (let i = 0; i < currentDetections.length; i++) {
      const det = currentDetections[i];
      const [ymin, xmin, ymax, xmax] = det.box_2d;

      const bx = (xmin / 1000) * vw;
      const by = (ymin / 1000) * vh;
      const bw = ((xmax - xmin) / 1000) * vw;
      const bh = ((ymax - ymin) / 1000) * vh;
      const color = det.color || PALETTE[i % PALETTE.length];

      ctx.fillStyle = `${color}30`;
      ctx.fillRect(bx, by, bw, bh);

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(bx, by, bw, bh);

      const labelText = det.confidence
        ? `${det.label} (${Math.round(det.confidence * 100)}%)`
        : det.label;

      ctx.font = "bold 14px 'Segoe UI', system-ui, sans-serif";
      const metrics = ctx.measureText(labelText);
      const badgeW = metrics.width + 16;
      const badgeH = 24;
      const badgeY = Math.max(0, by - badgeH - 2);

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(bx, badgeY, badgeW, badgeH, 4);
      ctx.fill();

      ctx.fillStyle = "#000000";
      ctx.textBaseline = "middle";
      ctx.fillText(labelText, bx + 8, badgeY + badgeH / 2);
    }

    // Trigger download
    const dataUrl = exportCanvas.toDataURL("image/png");
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const link = document.createElement("a");
    link.download = `visionground_snapshot_${timestamp}.png`;
    link.href = dataUrl;
    link.click();
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-100 rounded-xl overflow-hidden border border-slate-800">
      {/* Hidden raw video element */}
      <video
        ref={videoRef}
        playsInline
        muted
        className="hidden"
        onLoadedMetadata={() => {
          if (canvasRef.current && videoRef.current) {
            canvasRef.current.width = 1280;
            canvasRef.current.height = 720;
          }
        }}
      />

      {/* Top Action & Device Bar */}
      <div className="px-5 py-3.5 bg-slate-900 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/80 rounded-lg border border-slate-700/80 text-xs">
            <Camera className="w-4 h-4 text-sky-400" />
            <span className="text-slate-400 font-medium">Device:</span>
            <select
              id="camera-device-select"
              value={selectedDeviceId}
              onChange={handleDeviceChange}
              className="bg-transparent text-slate-200 outline-none cursor-pointer font-medium"
            >
              {devices.length === 0 ? (
                <option value="">Default Webcam</option>
              ) : (
                devices.map((d) => (
                  <option key={d.deviceId} value={d.deviceId} className="bg-slate-900">
                    {d.label}
                  </option>
                ))
              )}
            </select>
          </div>

          <button
            id="rescan-cameras-btn"
            onClick={() => {
              enumerateDevices();
              startCamera(selectedDeviceId);
            }}
            title="Rescan camera devices"
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>

        {/* Local YOLO11 Status Badge */}
        <div
          id="yolo11-status-pill"
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-mono font-semibold transition ${
            statusMessage.includes("Active")
              ? "bg-emerald-950/80 border-emerald-500/60 text-emerald-300 shadow-sm"
              : statusMessage.includes("disconnected") || statusMessage.includes("Diagnosing")
              ? "bg-amber-950/80 border-amber-500/60 text-amber-300"
              : "bg-slate-800/80 border-slate-700 text-slate-300"
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              statusMessage.includes("Active")
                ? "bg-emerald-400 animate-ping"
                : statusMessage.includes("disconnected")
                ? "bg-amber-400 animate-pulse"
                : "bg-sky-400"
            }`}
          />
          <span>{statusMessage}</span>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-800/70 rounded-md border border-slate-700/50">
            <Activity className="w-3.5 h-3.5 text-sky-400" />
            <span className="text-slate-400">FPS:</span>
            <span className="text-sky-300 font-bold">{fps}</span>
          </div>

          {latencyMs !== null && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-800/70 rounded-md border border-slate-700/50">
              <Zap className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-slate-400">Latency:</span>
              <span className="text-emerald-300 font-bold">{latencyMs} ms</span>
            </div>
          )}

          <div className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-800/70 rounded-md border border-slate-700/50">
            <Layers className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-slate-400">Objects:</span>
            <span className="text-amber-300 font-bold">{detections.length}</span>
          </div>

          {/* Performance Dashboard Collapsible Toggle */}
          <button
            id="toggle-performance-dashboard-btn"
            onClick={() => setShowPerfDashboard((prev) => !prev)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs transition cursor-pointer ${
              showPerfDashboard
                ? "bg-indigo-950/80 border-indigo-500/70 text-indigo-200 shadow-sm"
                : "bg-slate-800/70 border-slate-700/50 hover:bg-slate-800 text-slate-300"
            }`}
            title="Toggle Performance & Network Diagnostics Dashboard"
          >
            <Gauge className="w-3.5 h-3.5 text-indigo-400" />
            <span className="font-sans font-medium">Diagnostics</span>
            {dropErrorCount > 0 && (
              <span className="bg-rose-500/20 text-rose-300 text-[10px] px-1.5 py-0.2 rounded-full font-bold">
                {dropErrorCount}
              </span>
            )}
            {showPerfDashboard ? (
              <ChevronUp className="w-3.5 h-3.5 text-slate-400" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
            )}
          </button>
        </div>
      </div>

      {/* AI Error Diagnosis Card with 10s Cooldown Guard */}
      {aiErrorReasoning && (
        <div
          id="ai-error-reasoning-card"
          className="mx-5 my-2.5 p-3.5 bg-slate-900/95 border border-amber-500/50 rounded-xl shadow-xl flex items-start justify-between gap-3 text-xs backdrop-blur shrink-0"
        >
          <div className="flex items-start gap-3">
            <div className="p-2 bg-amber-500/10 rounded-lg text-amber-400 shrink-0">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="font-bold text-amber-300 text-sm">
                  Gemini AI Error Reasoning
                </span>
                <span className="text-[10px] font-mono bg-amber-500/20 text-amber-300 px-2 py-0.5 rounded-full">
                  10s Cooldown Active
                </span>
              </div>
              <p className="text-slate-200 font-mono text-[11px] leading-relaxed">
                {aiErrorReasoning}
              </p>
            </div>
          </div>
          <button
            id="dismiss-ai-diagnosis-btn"
            onClick={() => setAiErrorReasoning(null)}
            className="text-slate-400 hover:text-slate-200 text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 transition shrink-0"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Collapsible Performance Dashboard Widget */}
      {showPerfDashboard && (
        <div
          id="performance-dashboard-drawer"
          className="bg-slate-900/95 border-b border-indigo-500/30 px-5 py-3 text-xs backdrop-blur-md transition-all duration-200"
        >
          <div className="flex items-center justify-between mb-2 pb-1.5 border-b border-slate-800/80">
            <div className="flex items-center gap-2 flex-wrap">
              <BarChart3 className="w-4 h-4 text-indigo-400" />
              <span className="font-bold text-slate-200 tracking-wide">
                Live Backend Diagnostics & Telemetry
              </span>
              <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded-full font-mono">
                Inference: http://localhost:8000/detect
              </span>
              <span className="text-[10px] bg-amber-500/20 text-amber-300 px-2 py-0.5 rounded-full font-mono">
                Diagnostics: http://localhost:8000/diagnose-error (10s guard)
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                id="reset-perf-metrics-btn"
                onClick={() => {
                  setDropErrorCount(0);
                  setSuccessCount(0);
                  inferenceTimestampsRef.current = [];
                  setInferenceFrequency(0);
                }}
                className="text-[11px] text-slate-400 hover:text-slate-200 px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 transition"
              >
                Reset Counters
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono">
            {/* 1. Latency Metric */}
            <div className="bg-slate-950/70 p-2.5 rounded-lg border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
                <span>Last Latency</span>
                <Zap className="w-3 h-3 text-emerald-400" />
              </div>
              <div className="text-base font-bold text-emerald-300">
                {latencyMs !== null ? `${latencyMs} ms` : "--"}
              </div>
              <div className="text-[10px] text-slate-400 mt-0.5 truncate">
                API: {apiLatencyMs !== null ? `${apiLatencyMs}ms` : "--"} | Transfer:{" "}
                {latencyMs !== null && apiLatencyMs !== null
                  ? `${Math.max(0, latencyMs - apiLatencyMs)}ms`
                  : "--"}
              </div>
            </div>

            {/* 2. FPS & Inference Frequency */}
            <div className="bg-slate-950/70 p-2.5 rounded-lg border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
                <span>FPS / Frequency</span>
                <Activity className="w-3 h-3 text-sky-400" />
              </div>
              <div className="text-base font-bold text-sky-300">
                {fps} FPS
              </div>
              <div className="text-[10px] text-slate-400 mt-0.5">
                {inferenceFrequency} inferences / min
              </div>
            </div>

            {/* 3. Active Target Model */}
            <div className="bg-slate-950/70 p-2.5 rounded-lg border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
                <span>Target Model</span>
                <Cpu className="w-3 h-3 text-purple-400" />
              </div>
              <div className="text-sm font-bold text-purple-300 truncate">
                {activeModel}
              </div>
              <div className="text-[10px] text-slate-400 mt-0.5 flex items-center gap-1">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    activeModel.includes("3.6") ? "bg-emerald-400" : "bg-amber-400"
                  }`}
                />
                <span>{activeModel.includes("3.6") ? "Primary Model" : "Fallback Model"}</span>
              </div>
            </div>

            {/* 4. Drops & Success Rate */}
            <div className="bg-slate-950/70 p-2.5 rounded-lg border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
                <span>Drops / Health</span>
                {dropErrorCount === 0 ? (
                  <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                ) : (
                  <XCircle className="w-3 h-3 text-rose-400" />
                )}
              </div>
              <div className="flex items-baseline gap-1.5">
                <span
                  className={`text-base font-bold ${
                    dropErrorCount === 0 ? "text-emerald-300" : "text-rose-400"
                  }`}
                >
                  {dropErrorCount} drops
                </span>
                <span className="text-[10px] text-slate-400">({successCount} ok)</span>
              </div>
              <div className="text-[10px] text-slate-400 mt-0.5">
                Payload: {payloadBytes ? `${(payloadBytes / 1024).toFixed(1)} KB` : "~35 KB"}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Video Viewport Canvas */}
      <div className="relative flex-1 bg-slate-950 flex items-center justify-center p-3 overflow-hidden">
        <canvas
          ref={canvasRef}
          width={1280}
          height={720}
          className="max-h-full max-w-full rounded-lg shadow-2xl bg-black border border-slate-800/80 object-contain aspect-video"
        />

        {/* Gemini Servers Busy Temporary Status Banner */}
        {serverBusyNotice && (
          <div
            id="gemini-servers-busy-banner"
            className="absolute top-6 left-1/2 -translate-x-1/2 z-20 px-4 py-2 rounded-full bg-amber-950/95 border border-amber-500/70 text-amber-200 text-xs font-semibold flex items-center gap-2.5 backdrop-blur-md shadow-2xl animate-pulse"
          >
            <Clock className="w-4 h-4 text-amber-400 animate-spin" />
            <span>{serverBusyNotice}</span>
            <span className="text-[10px] bg-amber-500/20 text-amber-300 px-2 py-0.5 rounded-full font-mono font-bold">
              Cooldown 3s
            </span>
          </div>
        )}

        {/* Inference in-progress pulse banner */}
        {isInferring && (
          <div className="absolute top-6 right-6 px-3.5 py-1.5 rounded-full bg-slate-900/90 border border-sky-400/50 text-sky-300 text-xs font-semibold flex items-center gap-2 backdrop-blur shadow-lg animate-pulse">
            <Sparkles className="w-3.5 h-3.5 text-sky-400 animate-spin" />
            <span>Gemini Grounding...</span>
          </div>
        )}

        {/* Disconnected / Error Overlay */}
        {(!isStreaming || errorMessage) && (
          <div className="absolute inset-0 m-3 bg-slate-950/85 backdrop-blur-md rounded-lg flex flex-col items-center justify-center text-center p-6 space-y-4">
            <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-full text-rose-400">
              <AlertTriangle className="w-8 h-8" />
            </div>
            <div>
              <h4 className="text-base font-bold text-slate-100">
                Camera Feed Offline or Access Restricted
              </h4>
              <p className="text-xs text-slate-400 mt-1 max-w-md">
                {errorMessage ||
                  "Ensure your webcam is connected, not locked by other applications (Zoom/Teams), and permissions are granted."}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button
                id="retry-camera-btn"
                onClick={() => startCamera(selectedDeviceId)}
                className="px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-xs font-semibold flex items-center gap-2 transition"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Retry Camera
              </button>
              <button
                id="open-troubleshoot-btn"
                onClick={() => setShowTroubleshoot(true)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-semibold transition"
              >
                Troubleshooting Guide
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Controls Bar */}
      <div className="px-5 py-4 bg-slate-900 border-t border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          {/* Manual Detect Objects Button */}
          <button
            id="detect-objects-btn"
            disabled={!isStreaming || isInferring}
            onClick={runDetection}
            className={`px-4 py-2.5 rounded-lg text-xs font-bold flex items-center gap-2 transition ${
              isInferring
                ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                : "bg-sky-600 hover:bg-sky-500 text-white shadow-md shadow-sky-900/30"
            }`}
          >
            <Sparkles className="w-4 h-4 text-sky-200" />
            Detect Frame (YOLO11)
          </button>

          {/* Continuous Stream Analysis Toggle */}
          <label className="flex items-center gap-2.5 px-3 py-2 bg-slate-800/80 hover:bg-slate-800 rounded-lg border border-slate-700 cursor-pointer transition select-none">
            <input
              id="continuous-stream-toggle"
              type="checkbox"
              checked={isContinuous}
              onChange={(e) => setIsContinuous(e.target.checked)}
              className="w-4 h-4 accent-emerald-500 rounded cursor-pointer"
            />
            <span className="text-xs font-semibold text-slate-200">
              Continuous YOLO11 Stream (30+ FPS)
            </span>
          </label>
        </div>

        <div className="flex items-center gap-3">
          {/* Clear active boxes */}
          <button
            id="clear-boxes-btn"
            onClick={() => setDetections([])}
            disabled={detections.length === 0}
            className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-xs text-slate-300 font-medium flex items-center gap-1.5 transition border border-slate-700"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear Boxes
          </button>

          {/* Snapshot & Export */}
          <button
            id="snapshot-export-btn"
            disabled={!isStreaming}
            onClick={handleExportSnapshot}
            className="px-4 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-bold text-white flex items-center gap-2 transition shadow-md shadow-emerald-950/40"
          >
            <Download className="w-4 h-4" />
            Snapshot & Export (PNG)
          </button>
        </div>
      </div>

      {/* Detected Object Tag Pills Bar */}
      {detections.length > 0 && (
        <div className="px-5 py-2.5 bg-slate-950 border-t border-slate-900 flex items-center gap-2 overflow-x-auto">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider shrink-0">
            Detected:
          </span>
          <div className="flex items-center gap-2 flex-wrap">
            {detections.map((d, i) => (
              <span
                key={`${d.label}-${i}`}
                style={{
                  borderColor: `${d.color}60`,
                  backgroundColor: `${d.color}15`,
                  color: d.color,
                }}
                className="px-2.5 py-0.5 rounded-full text-xs font-medium border flex items-center gap-1.5"
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: d.color }}
                />
                {d.label}
                {d.confidence && (
                  <span className="opacity-75 text-[10px]">
                    {Math.round(d.confidence * 100)}%
                  </span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Troubleshooting Modal */}
      <TroubleshootModal
        isOpen={showTroubleshoot}
        onClose={() => setShowTroubleshoot(false)}
        errorMessage={errorMessage || undefined}
      />
    </div>
  );
};
