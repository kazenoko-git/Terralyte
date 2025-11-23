import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { writeTextFile } from "@tauri-apps/plugin-fs";

export interface SiteMeta {
  sample_id: string;
  lat: number;
  lon: number;
  zoom: number;
  radius: number;
  provider: string;
}

export interface AiResult {
  sample_id: string;
  lat: number;
  lon: number;
  has_solar: boolean;
  confidence: number;
  panel_count_est?: number;
  panel_count_Est?: number;     // Judges' format
  pv_area_sqm_est: number;
  capacity_kw_est: number;
  qc_status: string;
  qc_notes: string[];
  bbox_or_mask: any[];
  audit_overlay_path?: string;
  image_metadata?: {
    source: string;
    audit_overlay_path?: string;
    capture_date: string;
  };
  clusters?: Array<{
    cluster_id: number;
    num_detections: number;
    mean_confidence: number;
    centroid_px: [number, number];
  }>;
}

interface ResultsViewProps {
  meta: SiteMeta;
  result: AiResult;
  imageSrc: string | null;
  overlayImageSrc: string | null;
  onBack: () => void;
}

export default function ResultsView({
  meta,
  result,
  imageSrc,
  overlayImageSrc,
  onBack,
}: ResultsViewProps) {
  // ---- FIX PANEL COUNT SAFE ACCESS ----
  const panelCount =
    result.panel_count_est ??
    result.panel_count_Est ??
    0;

  // ---- EXPORT JSON ----
  const handleExportJSON = async () => {
    try {
      const fullData = { ...meta, ...result };
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
      alert(`❌ Export failed: ${err}`);
    }
  };

  // ---- ADD TO TRAINING ----
  const handleLabelForTraining = async () => {
    try {
      await invoke("add_to_training_data", {
        detection: result,
      });
      alert("✅ Added to training data!");
    } catch (err) {
      alert(`❌ Failed: ${err}`);
    }
  };

  // ---- UI STATUS COLORS ----
  const statusColor = result.has_solar
    ? result.confidence > 0.7
      ? "text-green-400"
      : "text-yellow-400"
    : "text-red-400";

  const statusIcon = result.has_solar
    ? result.confidence > 0.7
      ? "✓"
      : "⚠"
    : "✗";

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
          <h2 className="text-xl font-bold">Detection Results</h2>
        </div>

        <button
          onClick={handleExportJSON}
          className="px-4 py-2 bg-amber-500 hover:bg-amber-400 text-black rounded-lg text-sm font-semibold transition-all"
        >
          💾 Export JSON
        </button>
        <button
          onClick={handleLabelForTraining}
          className="px-4 py-2 bg-amber-500 hover:bg-amber-400 text-black rounded-lg text-sm font-semibold transition-all"
        >
          🏷️ Train
        </button>
      </div>

      {/* CONTENT */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* IMAGE PANEL */}
          <div className="bg-slate-900 rounded-lg border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800">
              <h3 className="font-semibold text-lg">Detection Result</h3>
            </div>
            <div className="p-4">

              {/* Priority: overlay → original image → fallback */}
              {overlayImageSrc ? (
                <img
                  src={overlayImageSrc}
                  alt="AI Detection Overlay"
                  className="w-full rounded-lg"
                />
              ) : imageSrc ? (
                <img
                  src={imageSrc}
                  alt="Satellite tile"
                  className="w-full rounded-lg"
                />
              ) : (
                <div className="w-full h-64 bg-slate-800 rounded-lg flex items-center justify-center">
                  <p className="text-slate-500">No image available</p>
                </div>
              )}

            </div>
          </div>

          {/* RESULTS PANEL */}
          <div className="space-y-6">
            {/* STATUS */}
            <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
              <h3 className="font-semibold text-lg mb-4">Detection Status</h3>
              <div className={`text-5xl font-bold ${statusColor} mb-2`}>
                {statusIcon}
              </div>
              <p className="text-2xl font-semibold mb-1">
                {result.has_solar ? "Solar Panels Detected" : "No Solar Panels"}
              </p>
              <p className="text-slate-400 text-sm">
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </p>
            </div>

            {/* METRICS */}
            <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
              <h3 className="font-semibold text-lg mb-4">Installation Metrics</h3>
              <div className="grid grid-cols-2 gap-4">

                <div>
                  <p className="text-slate-400 text-sm">Panel Count</p>
                  <p className="text-2xl font-bold text-amber-400">
                    {panelCount}
                  </p>
                </div>

                <div>
                  <p className="text-slate-400 text-sm">Area (m²)</p>
                  <p className="text-2xl font-bold text-amber-400">
                    {result.pv_area_sqm_est.toFixed(1)}
                  </p>
                </div>

                <div className="col-span-2">
                  <p className="text-slate-400 text-sm">Est. Capacity</p>
                  <p className="text-3xl font-bold text-amber-400">
                    {result.capacity_kw_est.toFixed(2)} kW
                  </p>
                </div>
              </div>
            </div>

            {/* LOCATION INFO */}
            <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
              <h3 className="font-semibold text-lg mb-4">Location Details</h3>
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
                  <span className="uppercase">{meta.provider}</span>
                </div>

                {/* Optional metadata */}
                {result.image_metadata && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Source:</span>
                      <span>{result.image_metadata.source}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Capture Date:</span>
                      <span>{result.image_metadata.capture_date}</span>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* QC NOTES */}
            {result.qc_notes && result.qc_notes.length > 0 && (
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

            {/* CLUSTERS */}
            {result.clusters && result.clusters.length > 0 && (
              <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
                <h3 className="font-semibold text-lg mb-3">Detected Panel Clusters</h3>
                <p className="text-slate-400 text-sm mb-2">Clusters group nearby detections (single-link, ~10m radius).</p>
                <ul className="space-y-2">
                  {result.clusters.map((c) => (
                    <li key={c.cluster_id} className="text-sm text-slate-300 flex items-start justify-between">
                      <div>
                        <div className="font-semibold">Cluster {c.cluster_id}</div>
                        <div className="text-slate-400 text-xs">{c.num_detections} detections • mean conf { (c.mean_confidence*100).toFixed(1) }%</div>
                      </div>
                      <div className="text-slate-400 text-xs font-mono">
                        [{c.centroid_px[0].toFixed(0)}, {c.centroid_px[1].toFixed(0)}]
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}

          </div>
        </div>
      </div>
    </div>
  );
}
