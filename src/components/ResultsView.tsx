// src/components/ResultsView.tsx
import React from "react";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { writeTextFile } from "@tauri-apps/plugin-fs";

/**
 * ResultsView.tsx
 * Full replacement — patched & hardened.
 *
 * Assumptions:
 * - overlayImageSrc is a data URL (data:image/png;base64,...). If not provided,
 *   the component will try to show imageSrc which should also be a data URL.
 * - Invokes to Tauri are preserved: add_to_training_data, etc.
 */

export interface SiteMeta {
  sample_id: string;
  lat: number;
  lon: number;
  zoom: number;
  radius: number;
  provider: string;
}

export interface Cluster {
  cluster_id: number;
  num_detections: number;
  mean_confidence: number;
  centroid_px: [number, number];
}

export interface AiResult {
  // core
  sample_id: string;
  lat: number;
  lon: number;
  has_solar: boolean;
  confidence: number;

  // counts (support both conventions)
  panel_count_est?: number;
  panel_count_Est?: number;

  // estimates
  pv_area_sqm_est: number;
  capacity_kw_est: number;

  // QC
  qc_status: string;
  qc_notes: string[];

  // detection details
  bbox_or_mask: any[]; // array of boxes or masks

  // overlay and metadata
  audit_overlay_path?: string;
  image_metadata?: {
    source?: string;
    audit_overlay_path?: string;
    capture_date?: string;
  };

  // clusters if produced
  clusters?: Cluster[];
}

interface ResultsViewProps {
  meta: SiteMeta;
  result: AiResult;
  imageSrc: string | null; // stitched / cropped tile (data url)
  overlayImageSrc: string | null; // annotated overlay (data url)
  onBack: () => void;
}

function safeNumber(n: any, fallback = 0) {
  const parsed = Number(n);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export default function ResultsView({
  meta,
  result,
  imageSrc,
  overlayImageSrc,
  onBack,
}: ResultsViewProps) {
  // ---- Safe panel count (supports your two possible keys) ----
  const panelCount = safeNumber(
    result.panel_count_est ?? result.panel_count_Est ?? 0,
    0
  );

  // ---- Helper: export JSON to user-chosen path ----
  const handleExportJSON = async () => {
    try {
      const fullData = { meta, ...result };
      const defaultFileName = `detection_${meta.sample_id}_export.json`;
      const savePath = await save({
        defaultPath: defaultFileName,
        filters: [{ name: "JSON", extensions: ["json"] }],
      });
      if (!savePath) return;
      await writeTextFile(savePath, JSON.stringify(fullData, null, 2));
      alert(`✅ Exported to:\n${savePath}`);
    } catch (err) {
      console.error("Export error:", err);
      alert(`❌ Export failed: ${String(err)}`);
    }
  };

  // ---- Add detection to training dataset (backend) ----
  const handleLabelForTraining = async () => {
    try {
      await invoke("add_to_training_data", {
        detection: result,
      });
      alert("✅ Added to training data!");
    } catch (err) {
      console.error("add_to_training_data failed:", err);
      alert(`❌ Failed to add to training: ${String(err)}`);
    }
  };

  // ---- Compute UI status colours/icons ----
  const statusColor =
    result.has_solar && result.confidence > 0.7
      ? "text-green-400"
      : result.has_solar && result.confidence > 0.4
      ? "text-yellow-400"
      : "text-red-400";
  const statusIcon = result.has_solar
    ? result.confidence > 0.7
      ? "✓"
      : "⚠"
    : "✗";

  // ---- Derived strings ----
  const captureDate =
    result.image_metadata?.capture_date ?? result.image_metadata?.audit_overlay_path ?? "N/A";

  // ---- Debug text about overlay path availability ----
  const overlayPathText = result.audit_overlay_path ?? result.image_metadata?.audit_overlay_path ?? null;

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 text-white overflow-hidden">
      {/* HEADER */}
      <div className="flex items-center justify-between px-6 py-4 bg-slate-900 border-b border-slate-800">
        <div className="flex items-center gap-4">
          <button
            onClick={onBack}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm font-medium transition-colors"
          >
            ← Back
          </button>
          <div>
            <h2 className="text-lg font-bold">Detection Results</h2>
            <div className="text-xs text-slate-400 mt-1 font-mono">{meta.sample_id}</div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleExportJSON}
            className="px-3 py-2 bg-amber-500 hover:bg-amber-400 text-black rounded-md text-sm font-semibold"
          >
            💾 Export JSON
          </button>

          <button
            onClick={handleLabelForTraining}
            className="px-3 py-2 bg-amber-500 hover:bg-amber-400 text-black rounded-md text-sm font-semibold"
          >
            🏷️ Train
          </button>
        </div>
      </div>

      {/* CONTENT */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Image panel */}
          <div className="bg-slate-900 rounded-lg border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <h3 className="font-semibold text-lg">Detection Image</h3>
              <div className="text-xs text-slate-400">{overlayPathText ? "Overlay available" : "No overlay"}</div>
            </div>

            <div className="p-4">
              {/* Priority: overlay -> image -> fallback */}
              {overlayImageSrc ? (
                <img
                  src={overlayImageSrc}
                  alt="AI Detection Overlay"
                  className="w-full rounded-lg object-contain"
                  style={{ maxHeight: 720 }}
                />
              ) : imageSrc ? (
                <img
                  src={imageSrc}
                  alt="Satellite tile"
                  className="w-full rounded-lg object-contain"
                  style={{ maxHeight: 720 }}
                />
              ) : (
                <div className="w-full h-64 bg-slate-800 rounded-lg flex items-center justify-center">
                  <p className="text-slate-500">No image available</p>
                </div>
              )}

              {/* Overlay debug / guidance */}
              <div className="mt-3 text-xs">
                {!overlayImageSrc && overlayPathText && (
                  <div className="text-yellow-300">
                    ⚠ Overlay path returned (<span className="font-mono">{overlayPathText}</span>) but the overlay image did not load.
                    <div className="mt-1 text-slate-400">Check backend `load_overlay_image` and file permissions.</div>
                  </div>
                )}

                {!overlayImageSrc && !overlayPathText && (
                  <div className="text-slate-400">No overlay produced by model.</div>
                )}
              </div>
            </div>
          </div>

          {/* Results & metadata panel */}
          <div className="space-y-6">
            {/* Detection status */}
            <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
              <div className="flex items-start gap-6">
                <div className={`text-6xl font-bold ${statusColor}`}>{statusIcon}</div>
                <div className="flex-1">
                  <div className="text-2xl font-semibold">
                    {result.has_solar ? "Solar Panels Detected" : "No Solar Panels"}
                  </div>
                  <div className="text-sm text-slate-400 mt-1">Confidence: {(safeNumber(result.confidence, 0) * 100).toFixed(1)}%</div>
                  <div className="text-xs text-slate-400 mt-3">QC Status: {result.qc_status ?? "N/A"}</div>
                </div>
              </div>
            </div>

            {/* Metrics */}
            <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
              <h3 className="font-semibold text-lg mb-3">Installation Metrics</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-slate-400 text-sm">Panel Count</p>
                  <p className="text-2xl font-bold text-amber-400">{panelCount}</p>
                </div>

                <div>
                  <p className="text-slate-400 text-sm">Area (m²)</p>
                  <p className="text-2xl font-bold text-amber-400">{(safeNumber(result.pv_area_sqm_est, 0)).toFixed(1)}</p>
                </div>

                <div className="col-span-2">
                  <p className="text-slate-400 text-sm">Est. Capacity</p>
                  <p className="text-3xl font-bold text-amber-400">{(safeNumber(result.capacity_kw_est, 0)).toFixed(2)} kW</p>
                </div>
              </div>
            </div>

            {/* Location & image metadata */}
            <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
              <h3 className="font-semibold text-lg mb-3">Location Details</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">Latitude:</span>
                  <span className="font-mono">{meta.lat.toFixed(7)}</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-slate-400">Longitude:</span>
                  <span className="font-mono">{meta.lon.toFixed(7)}</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-slate-400">Zoom Level:</span>
                  <span>{meta.zoom}</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-slate-400">Provider:</span>
                  <span className="font-mono">{meta.provider}</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-slate-400">Capture Date:</span>
                  <span className="font-mono">{captureDate ?? "N/A"}</span>
                </div>
              </div>
            </div>

            {/* QC notes */}
            {result.qc_notes?.length > 0 && (
              <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
                <h3 className="font-semibold text-lg mb-3">AI Quality Analysis</h3>
                <ul className="space-y-2">
                  {result.qc_notes.map((note, i) => (
                    <li key={i} className="text-sm text-slate-300 flex items-start gap-2">
                      <span className="text-amber-400">•</span>
                      <span>{note}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Clusters */}
            {result.clusters && result.clusters.length > 0 && (
              <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
                <h3 className="font-semibold text-lg mb-3">Detected Panel Clusters</h3>
                <p className="text-slate-400 text-sm mb-3">Clusters group nearby detections to identify arrays.</p>
                <ul className="space-y-2">
                  {result.clusters.map((c) => (
                    <li key={c.cluster_id} className="flex items-center justify-between text-sm text-slate-300">
                      <div>
                        <div className="font-semibold">Cluster {c.cluster_id}</div>
                        <div className="text-slate-400 text-xs">{c.num_detections} detections • mean conf {(c.mean_confidence*100).toFixed(1)}%</div>
                      </div>
                      <div className="text-slate-400 text-xs font-mono">[{Math.round(c.centroid_px[0])}, {Math.round(c.centroid_px[1])}]</div>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Raw JSON debug (collapsible-ish) */}
            <div className="bg-slate-900 rounded-lg border border-slate-800 p-4">
              <details>
                <summary className="text-sm text-slate-400 cursor-pointer">Raw result JSON (expand)</summary>
                <pre className="text-xs text-slate-300 mt-3 whitespace-pre-wrap max-h-60 overflow-auto p-2 bg-slate-800 rounded">
{JSON.stringify(result, null, 2)}
                </pre>
              </details>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
