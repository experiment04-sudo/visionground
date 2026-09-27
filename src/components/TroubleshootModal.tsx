import React from "react";
import { AlertCircle, X, ShieldAlert, CheckCircle2 } from "lucide-react";
import { PERMISSION_TROUBLESHOOTING } from "../constants";

interface TroubleshootModalProps {
  isOpen: boolean;
  onClose: () => void;
  errorMessage?: string;
}

export const TroubleshootModal: React.FC<TroubleshootModalProps> = ({
  isOpen,
  onClose,
  errorMessage,
}) => {
  if (!isOpen) return null;

  return (
    <div
      id="troubleshoot-modal-overlay"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200"
    >
      <div
        id="troubleshoot-modal-card"
        className="bg-slate-900 border border-slate-700 rounded-xl max-w-xl w-full p-6 shadow-2xl space-y-4"
      >
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-400">
              <ShieldAlert className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-slate-100">
                Camera Access Troubleshooting
              </h3>
              <p className="text-xs text-slate-400">
                Hardware check & permission resolution guide
              </p>
            </div>
          </div>
          <button
            id="close-troubleshoot-btn"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {errorMessage && (
          <div className="p-3 bg-rose-950/40 border border-rose-900/60 rounded-lg flex items-start gap-2.5 text-xs text-rose-300">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
            <span>{errorMessage}</span>
          </div>
        )}

        <div className="space-y-3 text-sm text-slate-300 bg-slate-950/60 p-4 rounded-lg border border-slate-800 max-h-72 overflow-y-auto font-mono text-xs leading-relaxed whitespace-pre-line">
          {PERMISSION_TROUBLESHOOTING}
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-slate-800">
          <span className="text-xs text-slate-400 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            Live video runs at 30 FPS with aspect ratio preservation
          </span>
          <button
            id="dismiss-troubleshoot-btn"
            onClick={onClose}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold rounded-lg transition"
          >
            Acknowledge & Close
          </button>
        </div>
      </div>
    </div>
  );
};
