// src/App.tsx — Fully patched & safe version
import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";

import Sidebar from "./components/Sidebar";
import MapPicker from "./components/MapPicker";
import ResultsView, { AiResult, SiteMeta } from "./components/ResultsView";
import SplashScreen from "./components/SplashScreen";

type FetchParams = {
  zoom: number;
  radius: number;
  provider: string;
};

type View = "select" | "results";

export default function App() {
  const [showSplash, setShowSplash] = useState(true);

  const [lat, setLat] = useState(12.8604075);
  const [lon, setLon] = useState(77.6625644);

  const [view, setView] = useState<View>("select");
  const [loading, setLoading] = useState(false);

  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [overlayImageSrc, setOverlayImageSrc] = useState<string | null>(null);

  const [siteMeta, setSiteMeta] = useState<SiteMeta | null>(null);
  const [aiResult, setAiResult] = useState<AiResult | null>(null);

  // Splash screen first
  if (showSplash) {
    return <SplashScreen onComplete={() => setShowSplash(false)} />;
  }

  // ================================
  //         MAIN ANALYSIS
  // ================================
  const handleAnalyze = async ({ zoom, radius, provider }: FetchParams) => {
    setLoading(true);
    console.log("Start AI analysis:", { lat, lon, zoom, radius, provider });

    try {
      // -------------------------------------
      // 1) FETCH TILE & CENTER CROP
      // -------------------------------------
      const stitchedTile = await invoke<string>("fetch_and_crop_tile", {
        lat,
        lon,
        zoom,
        radius,
        provider,
        crop_center: true,
        crop_size: 512,
      });

      if (!stitchedTile) throw new Error("Tile fetch returned empty.");

      console.log("Fetched + cropped tile OK. Length:", stitchedTile.length);
      setImageSrc(stitchedTile);

      // -------------------------------------
      // 2) BUILD METADATA OBJECT
      // -------------------------------------
      const meta: SiteMeta = {
        sample_id: `${Date.now()}`,
        lat,
        lon,
        zoom,
        radius,
        provider: provider.toLowerCase(),
      };

      setSiteMeta(meta);

      // -------------------------------------
      // 3) RUN AI ANALYSIS
      // -------------------------------------
      const aiRaw = await invoke<string>("run_ai_analysis", {
        imageB64: stitchedTile,
        sample_id: meta.sample_id,
        model_size: "nano", // performance boost
      });

      console.log("AI Output (raw):", aiRaw?.slice(0, 500));

      // Some logs may be printed before JSON
      const lastJsonLine = aiRaw
        .trim()
        .split("\n")
        .filter((line) => line.trim().startsWith("{"))
        .pop();

      if (!lastJsonLine) throw new Error("No JSON detected from AI pipeline.");

      const parsed: AiResult = JSON.parse(lastJsonLine);

      const fullResult: AiResult = {
        ...parsed,
        sample_id: meta.sample_id,
        lat: meta.lat,
        lon: meta.lon,
        image_metadata: parsed.image_metadata ?? parsed.image_metadata,
      };

      setAiResult(fullResult);

      // -------------------------------------
      // 4) LOAD OVERLAY IMAGE IF AVAILABLE
      // -------------------------------------
      if (parsed.audit_overlay_path) {
        try {
          const overlayB64 = await invoke<string>("load_overlay_image", {
            imagePath: parsed.audit_overlay_path,
          });

          if (overlayB64) {
            setOverlayImageSrc(overlayB64);
          } else {
            console.warn("Overlay path exists, but loading failed.");
          }
        } catch (e) {
          console.warn("Overlay load failed:", e);
        }

        // persist overlay into /Terralyte/detections
        await invoke("save_audit_overlay", {
          imagePath: parsed.audit_overlay_path,
          sampleId: meta.sample_id,
        });
      }

      // -------------------------------------
      // 5) AUTO-SAVE JSON INTO /Terralyte/detections
      // -------------------------------------
      await invoke("save_detection_json", {
        data: {
          ...fullResult,
          zoom,
          radius,
          provider,
        },
        filename: `detection_${meta.sample_id}.json`,
        outDir: "/Terralyte/detections",
      });

      // -------------------------------------
      // 6) SWAP TO RESULTS VIEW
      // -------------------------------------
      setView("results");
    } catch (err) {
      console.error("AI pipeline failed:", err);
      alert(`AI processing failed: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  // ================================
  //        RESULTS VIEW
  // ================================
  if (view === "results" && siteMeta && aiResult) {
    return (
      <ResultsView
        meta={siteMeta}
        result={aiResult}
        imageSrc={imageSrc}
        overlayImageSrc={overlayImageSrc}
        onBack={() => setView("select")}
      />
    );
  }

  // ================================
  //        SELECTION UI
  // ================================
  return (
    <div className="flex h-screen w-screen bg-slate-950 text-white overflow-hidden min-w-0 min-h-0">
      <Sidebar
        lat={lat}
        lon={lon}
        setLat={setLat}
        setLon={setLon}
        onFetch={handleAnalyze}
        loading={loading}
      />

      <div className="flex-1 flex min-w-0 min-h-0">
        <MapPicker
          lat={lat}
          lon={lon}
          onChange={(newLat, newLon) => {
            setLat(newLat);
            setLon(newLon);
          }}
        />
      </div>
    </div>
  );
}
