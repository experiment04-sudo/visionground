/**
 * VisionGround Backend Stress Test & Diagnostic Harness
 * ======================================================
 * Run with: npx tsx server_stress_test.ts
 *
 * Tests:
 * 1. Synthetic Frame Generation (valid 512x384 JPEG ~30KB)
 * 2. Concurrency Stress Test: 10 concurrent POST requests
 * 3. Rate-Limit / Spiking Test: 20 rapid sequential requests (100ms gap)
 * 4. Latency, Status Breakdown, Model Usage, and Memory Delta Benchmarking
 */

interface RequestResult {
  id: number;
  testType: "concurrency" | "spiking";
  status: number;
  latencyMs: number;
  apiLatencyMs?: number;
  modelUsed?: string;
  error?: string;
  detectionsCount?: number;
  payloadBytes?: number;
}

/**
 * Generates a valid 512x384 JPEG base64 payload (~30KB)
 * using standard JFIF SOF0 markers and safe comment padding.
 */
export function generateSyntheticFrame(): { base64: string; sizeBytes: number } {
  // Standard JFIF Grayscale 512x384 baseline JPEG binary
  const baseJpeg = Buffer.from([
    0xff, 0xd8, // SOI (Start of Image)
    0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, // APP0 (JFIF)
    0xff, 0xdb, 0x00, 0x43, 0x00, // DQT (Luminance Quantization Table)
    0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0a, 0x0c, 0x14,
    0x0d, 0x0c, 0x0b, 0x0b, 0x0c, 0x19, 0x12, 0x13, 0x0f, 0x14, 0x1d, 0x1a, 0x1f, 0x1e, 0x1d, 0x1a,
    0x1c, 0x1c, 0x20, 0x24, 0x2e, 0x27, 0x20, 0x22, 0x2c, 0x23, 0x1c, 0x1c, 0x28, 0x37, 0x29, 0x2c,
    0x30, 0x31, 0x34, 0x34, 0x34, 0x1f, 0x27, 0x39, 0x3d, 0x38, 0x32, 0x3c, 0x2e, 0x33, 0x34, 0x32,
    0xff, 0xc0, 0x00, 0x0b, 0x08, 0x01, 0x80, 0x02, 0x00, 0x01, 0x01, 0x11, 0x00, // SOF0: height=384 (0x0180), width=512 (0x0200)
    0xff, 0xc4, 0x00, 0x1f, 0x00, // DHT DC Table
    0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
    0xff, 0xc4, 0x00, 0xb5, 0x10, // DHT AC Table
    0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7d,
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
    0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3f, 0x00, // SOS: Start of Scan
  ]);

  // Add Comment (COM) segment to bring overall binary to ~30KB (28KB payload padding)
  const paddingLen = 28500;
  const comHeader = Buffer.from([0xff, 0xfe, (paddingLen + 2) >> 8, (paddingLen + 2) & 0xff]);
  const paddingBody = Buffer.alloc(paddingLen, 0x5a);

  // Scan termination & EOI (End of Image)
  const scanTrailer = Buffer.from([0x00, 0xff, 0xd9]);

  const completeJpeg = Buffer.concat([baseJpeg, comHeader, paddingBody, scanTrailer]);
  const base64 = completeJpeg.toString("base64");
  return { base64, sizeBytes: completeJpeg.length };
}

async function sendDetectRequest(
  url: string,
  base64Data: string,
  id: number,
  testType: "concurrency" | "spiking"
): Promise<RequestResult> {
  const start = performance.now();
  try {
    const payload = {
      imagePart: {
        inlineData: {
          mimeType: "image/jpeg",
          data: base64Data,
        },
      },
    };

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(15000),
    });

    const latencyMs = Math.round(performance.now() - start);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let json: any = {};
    try {
      json = await res.json();
    } catch {
      json = { error: "non_json_response" };
    }

    return {
      id,
      testType,
      status: res.status,
      latencyMs,
      apiLatencyMs: json.apiLatencyMs,
      modelUsed: json.modelUsed,
      detectionsCount: Array.isArray(json.detections) ? json.detections.length : 0,
      payloadBytes: json.payloadBytes,
      error: res.ok ? undefined : json.message || json.error || `HTTP ${res.status}`,
    };
  } catch (err: unknown) {
    const latencyMs = Math.round(performance.now() - start);
    return {
      id,
      testType,
      status: 0,
      latencyMs,
      error: (err as Error)?.message || "Network connection failure",
    };
  }
}

function computeStats(results: RequestResult[]) {
  const total = results.length;
  const successful = results.filter((r) => r.status === 200).length;
  const successRate = total > 0 ? ((successful / total) * 100).toFixed(1) : "0.0";

  const latencies = results.map((r) => r.latencyMs).filter((l) => typeof l === "number" && !isNaN(l));
  const minLatency = latencies.length > 0 ? Math.min(...latencies) : 0;
  const maxLatency = latencies.length > 0 ? Math.max(...latencies) : 0;
  const avgLatency =
    latencies.length > 0 ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length) : 0;

  const statusBreakdown: Record<string, number> = {
    "200 (OK)": 0,
    "400 (Bad Request)": 0,
    "429 (Rate Limit)": 0,
    "503 (Unavailable)": 0,
    "500 (Server Error)": 0,
    "Other / Network Fail": 0,
  };

  results.forEach((r) => {
    if (r.status === 200) statusBreakdown["200 (OK)"]++;
    else if (r.status === 400) statusBreakdown["400 (Bad Request)"]++;
    else if (r.status === 429) statusBreakdown["429 (Rate Limit)"]++;
    else if (r.status === 503) statusBreakdown["503 (Unavailable)"]++;
    else if (r.status === 500) statusBreakdown["500 (Server Error)"]++;
    else statusBreakdown["Other / Network Fail"]++;
  });

  const modelsUsed: Record<string, number> = {};
  results.forEach((r) => {
    if (r.modelUsed) {
      modelsUsed[r.modelUsed] = (modelsUsed[r.modelUsed] || 0) + 1;
    }
  });

  return {
    total,
    successful,
    successRate: `${successRate}%`,
    minLatency: `${minLatency} ms`,
    maxLatency: `${maxLatency} ms`,
    avgLatency: `${avgLatency} ms`,
    statusBreakdown,
    modelsUsed,
  };
}

async function runHarness() {
  const targetUrl = process.env.TEST_URL || "http://localhost:3000/api/detect";
  console.log("===============================================================================");
  console.log("             VisionGround Backend Stress & Diagnostic Harness                  ");
  console.log("===============================================================================");
  console.log(`Target Endpoint: ${targetUrl}`);
  console.log(`Timestamp:       ${new Date().toISOString()}`);

  const memBefore = process.memoryUsage();
  console.log(
    `Initial Memory:  RSS ${(memBefore.rss / 1024 / 1024).toFixed(2)} MB | Heap ${(memBefore.heapUsed / 1024 / 1024).toFixed(2)} MB`
  );

  // 1. Generate Synthetic Frame
  console.log("\n[Phase 1] Generating Synthetic 512x384 JPEG Frame...");
  const frame = generateSyntheticFrame();
  console.log(
    `✓ Synthetic frame ready: ${(frame.sizeBytes / 1024).toFixed(1)} KB binary (~${(frame.base64.length / 1024).toFixed(1)} KB Base64)`
  );

  // Verify health check before starting
  try {
    const healthRes = await fetch("http://localhost:3000/api/health");
    const healthJson = await healthRes.json();
    console.log(
      `✓ Health Check: status=${healthJson.status}, hasApiKey=${healthJson.hasApiKey}`
    );
  } catch (err: unknown) {
    console.warn(`! Health Check Warning: Could not ping /api/health (${(err as Error).message})`);
  }

  // 2. Concurrency Stress Test (10 concurrent requests)
  console.log("\n[Phase 2] Executing Concurrency Test: 10 Simultaneous Requests...");
  const concurrencyStart = performance.now();
  const concurrencyPromises: Promise<RequestResult>[] = [];
  for (let i = 1; i <= 10; i++) {
    concurrencyPromises.push(sendDetectRequest(targetUrl, frame.base64, i, "concurrency"));
  }
  const concurrencyResults = await Promise.all(concurrencyPromises);
  const concurrencyDuration = Math.round(performance.now() - concurrencyStart);
  console.log(`✓ Concurrency batch finished in ${concurrencyDuration} ms`);

  // 3. Spiking / Rate-Limit Test (20 sequential requests with 100ms gap)
  console.log("\n[Phase 3] Executing Rate-Limit & Spiking Test: 20 Sequential Requests (100ms gap)...");
  const spikingStart = performance.now();
  const spikingResults: RequestResult[] = [];
  for (let i = 1; i <= 20; i++) {
    const res = await sendDetectRequest(targetUrl, frame.base64, i, "spiking");
    spikingResults.push(res);
    console.log(
      `  [Spike ${String(i).padStart(2, " ")}/20] Status: ${res.status} | Latency: ${res.latencyMs}ms | Model: ${res.modelUsed || "N/A"}${res.error ? ` (${res.error.slice(0, 40)}...)` : ""}`
    );
    if (i < 20) {
      await new Promise((r) => setTimeout(r, 100));
    }
  }
  const spikingDuration = Math.round(performance.now() - spikingStart);
  console.log(`✓ Spiking test finished in ${spikingDuration} ms`);

  // 4. Memory Delta Analysis (Leak Check)
  const memAfter = process.memoryUsage();
  const heapDeltaMB = ((memAfter.heapUsed - memBefore.heapUsed) / 1024 / 1024).toFixed(2);
  const rssDeltaMB = ((memAfter.rss - memBefore.rss) / 1024 / 1024).toFixed(2);

  // 5. Compute and Display Benchmarks
  const allResults = [...concurrencyResults, ...spikingResults];
  const statsConcurrency = computeStats(concurrencyResults);
  const statsSpiking = computeStats(spikingResults);
  const statsOverall = computeStats(allResults);

  console.log("\n===============================================================================");
  console.log("                           BENCHMARK SUMMARY REPORT                            ");
  console.log("===============================================================================");

  console.log("\n--- OVERALL METRICS ---");
  console.table({
    "Total Requests Sent": statsOverall.total,
    "Successful (200 OK)": statsOverall.successful,
    "Overall Success Rate": statsOverall.successRate,
    "Average Latency": statsOverall.avgLatency,
    "Min Latency": statsOverall.minLatency,
    "Max Latency": statsOverall.maxLatency,
  });

  console.log("\n--- TEST BREAKDOWN ---");
  console.table({
    "Concurrency Test (10 requests)": {
      "Success Rate": statsConcurrency.successRate,
      "Avg Latency": statsConcurrency.avgLatency,
      "Min Latency": statsConcurrency.minLatency,
      "Max Latency": statsConcurrency.maxLatency,
    },
    "Spiking Test (20 requests)": {
      "Success Rate": statsSpiking.successRate,
      "Avg Latency": statsSpiking.avgLatency,
      "Min Latency": statsSpiking.minLatency,
      "Max Latency": statsSpiking.maxLatency,
    },
  });

  console.log("\n--- HTTP STATUS CODE BREAKDOWN ---");
  console.table(statsOverall.statusBreakdown);

  console.log("\n--- GEMINI MODEL USAGE (Primary vs Fallback) ---");
  if (Object.keys(statsOverall.modelsUsed).length === 0) {
    console.log("  (No successful API completions recorded)");
  } else {
    console.table(statsOverall.modelsUsed);
  }

  console.log("\n--- MEMORY CONSUMPTION & LEAK DIAGNOSTIC ---");
  console.table({
    "Initial RSS": `${(memBefore.rss / 1024 / 1024).toFixed(2)} MB`,
    "Final RSS": `${(memAfter.rss / 1024 / 1024).toFixed(2)} MB`,
    "RSS Delta": `${Number(rssDeltaMB) >= 0 ? "+" : ""}${rssDeltaMB} MB`,
    "Initial Heap Used": `${(memBefore.heapUsed / 1024 / 1024).toFixed(2)} MB`,
    "Final Heap Used": `${(memAfter.heapUsed / 1024 / 1024).toFixed(2)} MB`,
    "Heap Delta": `${Number(heapDeltaMB) >= 0 ? "+" : ""}${heapDeltaMB} MB`,
  });

  console.log("\nDiagnostic status check complete.");
  console.log("===============================================================================\n");
}

runHarness().catch((err) => {
  console.error("Stress test failed fatally:", err);
  process.exit(1);
});
