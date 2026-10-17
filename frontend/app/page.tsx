"use client";

import { useCallback, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const CLASS_STYLES: Record<number, { color: string; bg: string; label: string }> = {
  0: { color: "text-red-400",   bg: "bg-red-900/40 border-red-700",    label: "Person" },
  1: { color: "text-blue-400",  bg: "bg-blue-900/40 border-blue-700",  label: "Car" },
  2: { color: "text-amber-400", bg: "bg-amber-900/40 border-amber-700", label: "Motorcycle" },
  3: { color: "text-green-400", bg: "bg-green-900/40 border-green-700", label: "Other Vehicle" },
};

type Detection = {
  class_name: string;
  class_id: number;
  confidence: number;
  bbox: [number, number, number, number];
};

type ApiResult = {
  detections: Detection[];
  counts: Record<string, number>;
  total: number;
  inference_ms: number;
  annotated_image: string;
  original_image: string;
};

export default function Page() {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ApiResult | null>(null);
  const [view, setView] = useState<"annotated" | "original">("annotated");
  const fileRef = useRef<HTMLInputElement>(null);

  const runDetection = useCallback(async (file: File) => {
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/detect`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data: ApiResult = await res.json();
      setResult(data);
      setView("annotated");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) runDetection(file);
    },
    [runDetection]
  );

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) runDetection(file);
  };

  return (
    <main className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center gap-4">
          <div className="w-8 h-8 rounded-full bg-cyan-500/20 border border-cyan-500/50 flex items-center justify-center">
            <span className="text-cyan-400 text-xs">⊕</span>
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-widest text-slate-200 uppercase">
              HIT-UAV Thermal Detection
            </h1>
            <p className="text-xs text-slate-500">YOLOv8n · HIT-UAV · test mAP50: 0.867</p>
          </div>
          <a
            href={`${API}/docs`}
            target="_blank"
            rel="noreferrer"
            className="ml-auto text-xs text-slate-500 hover:text-slate-300 font-mono transition-colors"
          >
            API docs ↗
          </a>
        </div>
      </header>

      <div className="flex-1 max-w-7xl mx-auto w-full px-6 py-8 flex flex-col gap-8">
        {/* Upload Zone */}
        <div
          className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-200
            ${dragging ? "border-cyan-400 bg-cyan-950/30" : "border-slate-700 hover:border-slate-500 bg-slate-900/40"}
            ${loading ? "pointer-events-none opacity-60" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => fileRef.current?.click()}
        >
          <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={onFileChange} />
          {loading ? (
            <div className="flex flex-col items-center gap-3">
              <div className="w-10 h-10 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
              <p className="text-slate-400 text-sm">Running detection...</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <div className="w-14 h-14 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-2xl">
                🛸
              </div>
              <p className="text-slate-300 font-medium">Upload thermal image</p>
              <p className="text-slate-500 text-sm">Drag & drop or click · JPG, PNG</p>
            </div>
          )}
        </div>

        {error && (
          <div className="bg-red-950/50 border border-red-800 rounded-lg px-4 py-3 text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Image Panel */}
            <div className="lg:col-span-2 flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setView("annotated")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors
                    ${view === "annotated" ? "bg-cyan-600 text-white" : "bg-slate-800 text-slate-400 hover:text-slate-200"}`}
                >
                  Detection Result
                </button>
                <button
                  onClick={() => setView("original")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors
                    ${view === "original" ? "bg-cyan-600 text-white" : "bg-slate-800 text-slate-400 hover:text-slate-200"}`}
                >
                  Original
                </button>
                <span className="ml-auto text-xs text-slate-500 font-mono">
                  {result.inference_ms} ms · {result.total} target{result.total !== 1 ? "s" : ""}
                </span>
              </div>
              <div className="rounded-xl overflow-hidden border border-slate-800 bg-slate-900">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`data:image/jpeg;base64,${view === "annotated" ? result.annotated_image : result.original_image}`}
                  alt="Detection result"
                  className="w-full object-contain"
                />
              </div>
            </div>

            {/* Detection Panel */}
            <div className="flex flex-col gap-4">
              {/* Summary */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
                <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Summary</h2>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(result.counts).map(([name, count]) => {
                    const cls = Object.values(CLASS_STYLES).find(s => s.label === name);
                    return (
                      <div key={name} className={`rounded-lg border px-3 py-2 ${cls?.bg ?? "bg-slate-800 border-slate-700"}`}>
                        <div className={`text-lg font-bold ${cls?.color ?? "text-slate-300"}`}>{count}</div>
                        <div className="text-xs text-slate-400">{name}</div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Detection List */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-2 flex-1">
                <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Detections</h2>
                <div className="flex flex-col gap-2 overflow-y-auto max-h-96">
                  {result.detections.map((d, i) => {
                    const style = CLASS_STYLES[d.class_id] ?? CLASS_STYLES[3];
                    const pct = Math.round(d.confidence * 100);
                    return (
                      <div key={i} className={`rounded-lg border px-3 py-2 ${style.bg}`}>
                        <div className="flex items-center justify-between mb-1">
                          <span className={`text-xs font-semibold ${style.color}`}>{d.class_name}</span>
                          <span className="text-xs font-mono text-slate-300">{pct}%</span>
                        </div>
                        <div className="w-full h-1 bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{ width: `${pct}%`, backgroundColor: "currentColor" }}
                          />
                        </div>
                        <div className="text-[10px] text-slate-500 mt-1 font-mono">
                          [{d.bbox.map(v => Math.round(v)).join(", ")}]
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
