import React, { useEffect, useState } from "react";
import {
  FileCode,
  Download,
  Copy,
  Check,
  Terminal,
  Cpu,
  Layers,
  ShieldCheck,
  ExternalLink,
} from "lucide-react";
import JSZip from "jszip";

export const PythonCodeHub: React.FC = () => {
  const [files, setFiles] = useState<Record<string, string>>({});
  const [activeFile, setActiveFile] = useState<string>("main.py");
  const [copied, setCopied] = useState<boolean>(false);
  const [isZipping, setIsZipping] = useState<boolean>(false);

  useEffect(() => {
    fetch("/api/python-files")
      .then((res) => res.json())
      .then((data) => {
        if (data.files) {
          setFiles(data.files);
        }
      })
      .catch((err) => console.error("Failed to load python files:", err));
  }, []);

  const handleCopy = () => {
    const content = files[activeFile] || "";
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadZip = async () => {
    try {
      setIsZipping(true);
      const zip = new JSZip();
      const folder = zip.folder("webcam_vision_app");

      if (folder) {
        for (const [filename, content] of Object.entries(files)) {
          folder.file(filename, content);
        }
      }

      const blob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "webcam_vision_app.zip";
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to generate zip archive:", err);
    } finally {
      setIsZipping(false);
    }
  };

  const fileDescriptions: Record<string, string> = {
    "main.py":
      "PyQt6 Application Window, 30 FPS video rendering pipeline, letterbox projection, and UI controls.",
    "camera_manager.py":
      "Threaded OpenCV capture thread with hardware permission check and mid-stream disconnect handling.",
    "vision_engine.py":
      "Asynchronous Gemini 2.0 Flash spatial grounding client, QRunnable worker, and defensive coordinate parser.",
    "config.py":
      "Centralized settings: GEMINI_API_KEY, 30 FPS timing, 0-1000 coordinate scale, and high-contrast palette.",
    "requirements.txt":
      "Production Python dependencies: PyQt6, opencv-python, numpy, google-genai, python-dotenv.",
    "README.md":
      "Complete documentation, architecture breakdown, edge cases, and quick-start guide.",
    ".env.example":
      "Template for GEMINI_API_KEY and model configuration.",
  };

  const currentContent = files[activeFile] || "# Loading file content...";
  const lines = currentContent.split("\n");

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-100 rounded-xl overflow-hidden border border-slate-800">
      {/* Top Header & Zip Download */}
      <div className="px-6 py-4 bg-slate-900 border-b border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <Cpu className="w-5 h-5 text-sky-400" />
            <h3 className="text-base font-bold text-slate-100">
              Modular Desktop Python Architecture (/webcam_vision_app)
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Decoupled PyQt6 GUI + Threaded OpenCV Camera + Gemini 2.0 Flash Vision Engine
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            id="download-zip-btn"
            onClick={handleDownloadZip}
            disabled={isZipping || Object.keys(files).length === 0}
            className="px-4 py-2 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white rounded-lg text-xs font-bold flex items-center gap-2 transition shadow-md shadow-sky-950/40"
          >
            <Download className="w-4 h-4" />
            {isZipping ? "Bundling ZIP..." : "Download Full Package (.zip)"}
          </button>
        </div>
      </div>

      {/* Architecture Highlights Bar */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-2.5 p-4 bg-slate-950/70 border-b border-slate-800/80 text-xs">
        <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-800 flex items-start gap-2.5">
          <div className="p-1.5 bg-blue-500/10 text-blue-400 rounded-md shrink-0">
            <Layers className="w-4 h-4" />
          </div>
          <div>
            <div className="font-semibold text-slate-200">Decoupled Design</div>
            <div className="text-[11px] text-slate-400 mt-0.5">
              UI, camera capture, and vision AI reside in completely independent files.
            </div>
          </div>
        </div>

        <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-800 flex items-start gap-2.5">
          <div className="p-1.5 bg-emerald-500/10 text-emerald-400 rounded-md shrink-0">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <div className="font-semibold text-slate-200">Non-Blocking GUI</div>
            <div className="text-[11px] text-slate-400 mt-0.5">
              Vision inference runs on a QRunnable worker pool while video streams at 30 FPS.
            </div>
          </div>
        </div>

        <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-800 flex items-start gap-2.5">
          <div className="p-1.5 bg-amber-500/10 text-amber-400 rounded-md shrink-0">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <div>
            <div className="font-semibold text-slate-200">Aspect Ratio Math</div>
            <div className="text-[11px] text-slate-400 mt-0.5">
              Letterbox scaling projects 0-1000 coordinates onto rendered pixels accurately.
            </div>
          </div>
        </div>

        <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-800 flex items-start gap-2.5">
          <div className="p-1.5 bg-purple-500/10 text-purple-400 rounded-md shrink-0">
            <Terminal className="w-4 h-4" />
          </div>
          <div>
            <div className="font-semibold text-slate-200">Defensive Parsing</div>
            <div className="text-[11px] text-slate-400 mt-0.5">
              Strips markdown wrappers and sanitizes malformed LLM outputs gracefully.
            </div>
          </div>
        </div>
      </div>

      {/* Main Code View Area */}
      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* File Navigator Sidebar */}
        <div className="w-full md:w-64 bg-slate-900/90 border-r border-slate-800 p-3 space-y-1 overflow-y-auto">
          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider px-2 py-1.5">
            Project Files
          </div>
          {Object.keys(files).length === 0 ? (
            <div className="text-xs text-slate-500 px-2 py-2">Loading files...</div>
          ) : (
            Object.keys(files)
              .sort((a, b) => (a === "main.py" ? -1 : b === "main.py" ? 1 : a.localeCompare(b)))
              .map((filename) => {
                const isActive = activeFile === filename;
                return (
                  <button
                    key={filename}
                    onClick={() => setActiveFile(filename)}
                    className={`w-full text-left px-3 py-2 rounded-lg text-xs font-mono flex items-center gap-2 transition ${
                      isActive
                        ? "bg-sky-600/20 text-sky-300 border border-sky-500/40 font-bold"
                        : "text-slate-300 hover:bg-slate-800/80 hover:text-slate-100"
                    }`}
                  >
                    <FileCode
                      className={`w-4 h-4 shrink-0 ${
                        isActive ? "text-sky-400" : "text-slate-500"
                      }`}
                    />
                    <span className="truncate">{filename}</span>
                  </button>
                );
              })
          )}

          {/* Quick Start Run Commands */}
          <div className="pt-4 mt-4 border-t border-slate-800">
            <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider px-2 py-1.5">
              Local Execution
            </div>
            <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800 font-mono text-[11px] text-slate-300 space-y-1">
              <div className="text-slate-500"># Setup & Run</div>
              <div className="text-sky-400">pip install -r requirements.txt</div>
              <div className="text-emerald-400">python main.py</div>
            </div>
          </div>
        </div>

        {/* Code Content Viewer */}
        <div className="flex-1 flex flex-col bg-slate-950 overflow-hidden">
          {/* File Header */}
          <div className="px-5 py-3 bg-slate-900/60 border-b border-slate-800 flex items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-bold text-sky-400">
                  /webcam_vision_app/{activeFile}
                </span>
                <span className="text-[10px] text-slate-400 px-2 py-0.5 bg-slate-800 rounded">
                  {lines.length} lines
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                {fileDescriptions[activeFile] || "Python source file."}
              </p>
            </div>

            <button
              id="copy-code-btn"
              onClick={handleCopy}
              className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-medium flex items-center gap-1.5 transition"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400 font-bold">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy Code</span>
                </>
              )}
            </button>
          </div>

          {/* Code Viewer Body */}
          <div className="flex-1 overflow-auto p-4 font-mono text-xs leading-relaxed bg-slate-950">
            <table className="w-full border-collapse">
              <tbody>
                {lines.map((line, idx) => (
                  <tr key={idx} className="hover:bg-slate-900/60">
                    <td className="w-12 pr-4 text-right select-none text-slate-600 text-[11px] align-top">
                      {idx + 1}
                    </td>
                    <td className="text-slate-300 whitespace-pre font-mono">
                      {line}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
