import React, { useState } from "react";
import { LiveVisionStudio } from "./components/LiveVisionStudio";
import { PythonCodeHub } from "./components/PythonCodeHub";
import { Video, Code2, Sparkles, Shield, Compass } from "lucide-react";

export default function App() {
  const [activeTab, setActiveTab] = useState<"live" | "code">("live");

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-sky-500/30 selection:text-sky-200">
      {/* Top Navbar */}
      <header className="h-16 px-6 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between shrink-0 backdrop-blur-md sticky top-0 z-30">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-sky-600 to-cyan-400 flex items-center justify-center text-slate-950 shadow-md shadow-sky-500/20 font-black">
            <Sparkles className="w-5 h-5 text-slate-950" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-slate-100 tracking-tight">
                VisionGround
              </h1>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-500/10 text-sky-400 border border-sky-500/30">
                Gemini 2.0 Flash Grounding
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              Modular Python (PyQt6 + OpenCV) & Live Browser Companion
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center bg-slate-950/80 p-1 rounded-xl border border-slate-800">
          <button
            id="tab-live-studio-btn"
            onClick={() => setActiveTab("live")}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition ${
              activeTab === "live"
                ? "bg-sky-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
            }`}
          >
            <Video className="w-4 h-4" />
            Live Vision Studio
          </button>

          <button
            id="tab-python-code-btn"
            onClick={() => setActiveTab("code")}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition ${
              activeTab === "code"
                ? "bg-sky-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
            }`}
          >
            <Code2 className="w-4 h-4" />
            Python Desktop Architecture (/webcam_vision_app)
          </button>
        </div>

        <div className="hidden lg:flex items-center gap-3 text-xs text-slate-400">
          <span className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800/60 rounded-lg border border-slate-700/60">
            <Shield className="w-3.5 h-3.5 text-emerald-400" />
            30 FPS • Non-blocking QThread
          </span>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 p-4 md:p-6 max-w-[1600px] w-full mx-auto flex flex-col overflow-hidden">
        {activeTab === "live" ? <LiveVisionStudio /> : <PythonCodeHub />}
      </main>
    </div>
  );
}
