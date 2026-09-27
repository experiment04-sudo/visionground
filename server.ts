import express from "express";
import path from "path";
import fs from "fs";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "25mb" }));

// Lazy initialization of Gemini SDK
let aiClient: GoogleGenAI | null = null;
function getGenAI(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY environment variable is missing.");
    }
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

// -----------------------------------------------------------------------------
// API Routes
// -----------------------------------------------------------------------------

// Health check
app.get("/api/health", (_req, res) => {
  res.json({
    status: "ok",
    hasApiKey: Boolean(process.env.GEMINI_API_KEY),
    timestamp: new Date().toISOString(),
  });
});

// Optimized fast spatial detection prompt to minimize generation latency
const SPATIAL_PROMPT = "Return JSON array of detected objects with box_2d [ymin, xmin, ymax, xmax] and label.";

interface DetectRetryResult {
  result: any; // eslint-disable-line @typescript-eslint/no-explicit-any
  modelUsed: string;
  apiLatencyMs: number;
}

// Resilient Gemini detect engine with exponential backoff & model fallback
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function detectWithRetry(imagePart: any, maxRetries = 2): Promise<DetectRetryResult> {
  const genAI = getGenAI();
  let delay = 300;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const isFallback = attempt === maxRetries;
    const modelToUse = isFallback ? "gemini-3.5-flash" : "gemini-3.6-flash";
    const promptText = isFallback
      ? "Detect all distinct objects. Return JSON array of objects with box_2d [ymin, xmin, ymax, xmax] and label."
      : "Detect all distinct objects. Return a raw JSON array of objects with box_2d [ymin, xmin, ymax, xmax] and label.";

    try {
      const apiStart = performance.now();
      const res = await genAI.models.generateContent({
        model: modelToUse,
        contents: [imagePart, promptText],
        config: {
          temperature: 0.2,
          responseMimeType: "application/json",
        },
      });
      const apiLatencyMs = Math.round(performance.now() - apiStart);

      return {
        result: res,
        modelUsed: modelToUse,
        apiLatencyMs,
      };
    } catch (err: any) { // eslint-disable-line @typescript-eslint/no-explicit-any
      const status =
        err?.error?.code ??
        err?.status ??
        (String(err).includes("503") ? 503 : undefined) ??
        (String(err).includes("429") ? 429 : undefined);
      const isRetryable =
        status === 503 ||
        status === 429 ||
        String(err).includes("UNAVAILABLE") ||
        String(err).includes("overloaded");

      // Fail fast on non-retryable errors (404, 401, 400)
      if (!isRetryable) {
        throw err;
      }

      if (attempt === maxRetries) {
        throw err;
      }

      console.warn(
        `[detectWithRetry] Attempt ${attempt + 1}/${maxRetries} failed (status: ${status}). Retrying with backoff ${Math.round(delay)}ms...`
      );
      await new Promise((r) => setTimeout(r, delay + Math.random() * 200));
      delay *= 2;
    }
  }

  throw new Error("detectWithRetry: Exhausted all retries without a response");
}

// Spatial object detection proxy using Gemini
app.post("/api/detect", async (req, res) => {
  const reqStart = performance.now();
  try {
    let imagePart = req.body.imagePart;
    if (!imagePart && req.body.imageBase64) {
      const cleanBase64 = req.body.imageBase64.replace(/^data:image\/\w+;base64,/, "");
      imagePart = {
        inlineData: {
          mimeType: req.body.mimeType || "image/jpeg",
          data: cleanBase64,
        },
      };
    }

    // Strict payload validation
    if (!imagePart?.inlineData?.data || !imagePart?.inlineData?.mimeType) {
      console.warn("[/api/detect] Rejected request: Missing or malformed image payload.");
      return res.status(400).json({
        error: "invalid_payload",
        message: "Malformed image payload structure.",
      });
    }

    const payloadBytes = Buffer.byteLength(imagePart.inlineData.data, "base64");
    console.log(
      `[/api/detect] Incoming frame: ${(payloadBytes / 1024).toFixed(1)} KB (${payloadBytes} bytes)`
    );

    const { result, modelUsed, apiLatencyMs } = await detectWithRetry(imagePart);
    const latencyMs = Math.round(performance.now() - reqStart);

    console.log(
      `[/api/detect SUCCESS] Model: ${modelUsed} | API Latency: ${apiLatencyMs}ms | Total Latency: ${latencyMs}ms | Payload: ${(payloadBytes / 1024).toFixed(1)} KB`
    );

    const rawText = result?.text || "[]";

    // Defensive parsing
    let cleaned = rawText.trim();
    const markdownMatch = cleaned.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    if (markdownMatch) {
      cleaned = markdownMatch[1].trim();
    }

    let parsed = [];
    try {
      parsed = JSON.parse(cleaned);
    } catch {
      const start = cleaned.indexOf("[");
      const end = cleaned.lastIndexOf("]");
      if (start !== -1 && end !== -1 && end > start) {
        parsed = JSON.parse(cleaned.substring(start, end + 1));
      }
    }

    if (!Array.isArray(parsed) && typeof parsed === "object" && parsed !== null) {
      const candidateKeys = ["objects", "detections", "items", "boxes"];
      for (const k of candidateKeys) {
        if (Array.isArray((parsed as Record<string, unknown>)[k])) {
          parsed = (parsed as Record<string, unknown>)[k] as unknown[];
          break;
        }
      }
    }

    const validDetections: Array<{
      label: string;
      box_2d: [number, number, number, number];
      confidence?: number;
    }> = [];

    if (Array.isArray(parsed)) {
      for (const item of parsed) {
        if (!item || typeof item !== "object") continue;
        const label = String(item.label || item.name || "object").trim();
        const box = item.box_2d || item.box || item.bbox;
        if (Array.isArray(box) && box.length === 4) {
          let [ymin, xmin, ymax, xmax] = box.map((n: unknown) => Number(n) || 0);

          if (ymin > ymax) [ymin, ymax] = [ymax, ymin];
          if (xmin > xmax) [xmin, xmax] = [xmax, xmin];

          // Clamp 0..1000
          ymin = Math.max(0, Math.min(1000, ymin));
          xmin = Math.max(0, Math.min(1000, xmin));
          ymax = Math.max(0, Math.min(1000, ymax));
          xmax = Math.max(0, Math.min(1000, xmax));

          if (ymax - ymin >= 5 && xmax - xmin >= 5) {
            validDetections.push({
              label,
              box_2d: [ymin, xmin, ymax, xmax],
              confidence: typeof item.confidence === "number" ? item.confidence : 0.95,
            });
          }
        }
      }
    }

    res.json({
      detections: validDetections,
      result,
      latencyMs,
      apiLatencyMs,
      modelUsed,
      payloadBytes,
    });
  } catch (err: any) { // eslint-disable-line @typescript-eslint/no-explicit-any
    const rawStatus = err?.status || err?.error?.code || (String(err).includes("503") ? 503 : 500);
    const status = typeof rawStatus === "number" && rawStatus >= 400 && rawStatus < 600 ? rawStatus : 500;
    console.error(
      `[/api/detect ERROR] Status: ${status} | Code: ${err?.code || err?.error?.code || "N/A"} | Name: ${err?.name || "Error"} | Message: ${err?.message || "Detection failed"}`
    );
    if (err?.stack) {
      console.error(`[/api/detect Stack]:`, err.stack);
    }
    return res.status(status).json({
      error: "detection_error",
      message: err?.message || "Detection failed.",
      status,
    });
  }
});

// Serve Python project files for live inspection and zip download
app.get("/api/python-files", (_req, res) => {
  const dirPath = path.join(process.cwd(), "webcam_vision_app");
  try {
    if (!fs.existsSync(dirPath)) {
      return res.status(404).json({ error: "webcam_vision_app folder not found" });
    }

    const fileNames = fs.readdirSync(dirPath);
    const files: Record<string, string> = {};

    for (const file of fileNames) {
      const fullPath = path.join(dirPath, file);
      const stat = fs.statSync(fullPath);
      if (stat.isFile()) {
        files[file] = fs.readFileSync(fullPath, "utf-8");
      }
    }

    res.json({ files });
  } catch (err: unknown) {
    res.status(500).json({ error: String(err) });
  }
});

// -----------------------------------------------------------------------------
// Vite Middleware / Static Serving
// -----------------------------------------------------------------------------
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  // Global error handling middleware (Must have 4 parameters)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unused-vars
  app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
    console.error("Express Unhandled Error:", err);
    const status = err.status || err.statusCode || 500;
    res.status(status).json({
      error: "server_error",
      message: err.message || "An internal server error occurred",
    });
  });

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});

