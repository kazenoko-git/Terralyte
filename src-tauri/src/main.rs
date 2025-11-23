#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{env, fs, path::PathBuf, process::Command};
use tauri::{command, Manager};
use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri_plugin_dialog;
use tauri_plugin_fs;
use tauri_plugin_shell;



// ============================================================
// DATA STRUCTURES
// ============================================================

#[derive(Debug, Serialize, Deserialize)]
struct SolarDetection {
    // ADDED DEFAULTS TO ALL CSV/APP FIELDS
    #[serde(default)]
    sample_id: String,

    #[serde(default)]
    lat: f64,

    #[serde(default)]
    lon: f64,

    has_solar: bool,
    confidence: f64,
    panel_count_est: usize,
    pv_area_sqm_est: f64,
    capacity_kw_est: f64,
    qc_status: String,

    #[serde(default)]
    qc_notes: Vec<String>,

    #[serde(default)]
    bbox_or_mask: Vec<Vec<Vec<f64>>>,

    // THESE THREE MUST *ALSO* BE OPTIONAL
    #[serde(default)]
    zoom: u32,

    #[serde(default)]
    radius: u32,

    #[serde(default)]
    provider: String,

    #[serde(default)]
    audit_overlay_path: Option<String>,

    #[serde(default)]
    image_metadata: Option<ImageMetadata>,
}


#[derive(Debug, Serialize, Deserialize)]
struct ImageMetadata {
    source: String,
    capture_date: String,
}

#[derive(Debug, Deserialize)]
struct CsvRow {
    sample_id: String,
    lat: f64,
    lon: f64,
}

// ============================================================
// PATH HELPERS (FLAT STRUCTURE)
// ============================================================
//
// src-tauri is inside project root
// python scripts are in project root
// audit_overlays, detections, tile_cache are in project root
//

fn get_paths() -> Result<(PathBuf, PathBuf), String> {
    let tauri_dir = env::current_dir().map_err(|e| e.to_string())?;
    let project_root = tauri_dir
        .parent()
        .ok_or("Failed to find project root")?
        .to_path_buf();

    Ok((tauri_dir, project_root))
}

// System python
fn py() -> PathBuf {
    PathBuf::from("/usr/local/bin/python3")
}

// ============================================================
// TAURI COMMANDS
// ============================================================

#[command]
fn fetch_and_crop_tile(lat: f64, lon: f64, zoom: u32, radius: u32, provider: String)
    -> Result<String, String>
{
    let (_tauri_dir, project_root) = get_paths()?;
    let python_path = py();

    let script_path = project_root.join("imagenRunner.py");
    if !script_path.exists() {
        return Err(format!("imagenRunner.py missing at {}", script_path.display()));
    }

    let output = Command::new(&python_path)
        .current_dir(&project_root)
        .arg(&script_path)
        .arg(lat.to_string())
        .arg(lon.to_string())
        .arg(zoom.to_string())
        .arg(radius.to_string())
        .arg(provider)
        .arg("--crop")
        .output()
        .map_err(|e| format!("Failed to spawn python: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    if !output.status.success() {
        return Err(format!("Python error: {}", stdout));
    }

    let rel = stdout.lines().last().unwrap_or("").trim();
    if rel.is_empty() {
        return Err("Python returned no path".into());
    }

    let img_path = project_root.join(rel);
    if !img_path.exists() {
        return Err(format!("Image not found: {}", img_path.display()));
    }

    let bytes = fs::read(img_path).map_err(|e| e.to_string())?;
    Ok(format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(bytes)))
}

#[command]
fn fetch_stitched_tile(lat: f64, lon: f64, zoom: u32, radius: u32, provider: String)
    -> Result<String, String>
{
    let (_tauri_dir, project_root) = get_paths()?;
    let python_path = py();

    let script_path = project_root.join("imagenRunner.py");
    if !script_path.exists() {
        return Err(format!("imagenRunner.py missing: {}", script_path.display()));
    }

    let output = Command::new(&python_path)
        .current_dir(&project_root)
        .arg(&script_path)
        .arg(lat.to_string())
        .arg(lon.to_string())
        .arg(zoom.to_string())
        .arg(radius.to_string())
        .arg(provider)
        .arg("--crop")
        .output()
        .map_err(|e| format!("Failed to spawn python: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    if !output.status.success() {
        return Err(format!("Python error: {}", stdout));
    }

    let rel = stdout.lines().last().unwrap_or("").trim();
    if rel.is_empty() {
        return Err("Python returned no path".into());
    }

    let img_path = project_root.join(rel);

    let bytes = fs::read(img_path).map_err(|e| e.to_string())?;
    Ok(format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(bytes)))
}

#[command]
fn run_ai_analysis(image_b64: String) -> Result<String, String> {
    let (_tauri_dir, project_root) = get_paths()?;
    let python_path = py();

    let tmp = project_root.join("tmp_input.png");
    let script = project_root.join("run_model.py");
    let model = project_root.join("verifier2.pt");

    fs::write(
        &tmp,
        general_purpose::STANDARD.decode(
            image_b64.replace("data:image/png;base64,", "")
        ).map_err(|e| e.to_string())?
    ).map_err(|e| format!("Failed to write PNG: {e}"))?;

    let output = Command::new(&python_path)
        .arg(&script)
        .arg(&tmp)
        .arg(&model)
        .output()
        .map_err(|e| format!("Failed to run AI script: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        return Err(format!("AI script error: {}", stderr));
    }

    // --- REAL JSON EXTRACTION FIX ---
    let start = stdout.find('{').ok_or("No JSON detected in AI output")?;
    let end = stdout.rfind('}').ok_or("Invalid JSON output")?;
    let json_str = &stdout[start..=end];

    Ok(json_str.to_string())
}


#[command]
fn process_csv_batch(
    csv_path: String,
    zoom: u32,
    radius: u32,
    provider: String,
) -> Result<Vec<SolarDetection>, String> {
    use csv::ReaderBuilder;

    // Figure out paths
    let (project_root, gui_dir) = get_paths()?;

    // Open CSV
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(&csv_path)
        .map_err(|e| format!("Failed to read CSV: {}", e))?;

    let mut results = Vec::new();

    for row in reader.deserialize() {
        let row: CsvRow = row.map_err(|e| format!("Bad CSV row: {}", e))?;

        // Fetch stitched tile
        let tile_b64 = fetch_stitched_tile(
            row.lat,
            row.lon,
            zoom,
            radius,
            provider.clone(),
        )?;

        // Run AI on that
        let ai_json = run_ai_analysis(tile_b64)?;

        // Extract JSON line from stdout
        let json_line = ai_json
            .lines()
            .find(|l| l.trim().starts_with('{'))
            .ok_or("AI output missing JSON")?;

        let mut det: SolarDetection =
            serde_json::from_str(json_line)
                .map_err(|e| format!("AI JSON parse error: {}", e))?;

        // Fill metadata
        det.sample_id = row.sample_id;
        det.lat = row.lat;
        det.lon = row.lon;
        det.zoom = zoom;
        det.radius = radius;
        det.provider = provider.clone();

        results.push(det);
    }

    Ok(results)
}

#[command]
fn save_batch_results(
    detections: Vec<SolarDetection>,
    batch_name: String,
) -> Result<String, String> {
    let (_tauri_dir, project_root) = get_paths()?;

    
    let output_dir = project_root.join("batch_results");
    std::fs::create_dir_all(&output_dir).map_err(|e| e.to_string())?;
    
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let filename = format!("{}_{}.json", batch_name, timestamp);
    let output_path = output_dir.join(&filename);
    
    let json_string = serde_json::to_string_pretty(&detections)
        .map_err(|e| format!("Failed to serialize: {}", e))?;
    
    std::fs::write(&output_path, json_string)
        .map_err(|e| format!("Failed to write file: {}", e))?;
    
    Ok(output_path.to_string_lossy().to_string())
}


#[command]
fn load_overlay_image(image_path: String) -> Result<String, String> {
    let path = PathBuf::from(&image_path);

    if !path.exists() {
        return Err(format!("Overlay missing: {}", image_path));
    }

    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    Ok(format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(bytes)))
}

#[command]
fn save_detection_json(data: serde_json::Value, filename: String) -> Result<String, String> {
    let (_tauri_dir, project_root) = get_paths()?;

    let dir = project_root.join("detections");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let path = dir.join(&filename);

    // Normalize data: if it's an object → wrap it in an array
    let normalized = match data {
        serde_json::Value::Array(_) => data,
        serde_json::Value::Object(_) => serde_json::Value::Array(vec![data]),
        _ => return Err("Invalid data type: expected object or array".into()),
    };

    let json_string = serde_json::to_string_pretty(&normalized)
        .map_err(|e| format!("Failed to serialize: {}", e))?;

    fs::write(&path, json_string).map_err(|e| e.to_string())?;

    Ok(path.display().to_string())
}


#[command]
fn save_audit_overlay(image_path: String, sample_id: String) -> Result<String, String> {
    let (_tauri_dir, project_root) = get_paths()?;

    let dir = project_root.join("audit_overlays");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let filename = format!("audit_{}_{}.png", sample_id, chrono::Local::now().format("%Y%m%d_%H%M%S"));
    let out = dir.join(filename);

    fs::copy(&image_path, &out).map_err(|e| e.to_string())?;
    Ok(out.display().to_string())
}

#[command]
fn clear_tile_cache() -> Result<String, String> {
    let (_tauri_dir, project_root) = get_paths()?;
    let cache_dir = project_root.join("tile_cache");

    if cache_dir.exists() {
        fs::remove_dir_all(&cache_dir).map_err(|e| e.to_string())?;
    }

    Ok("Cache cleared".into())
}

#[command]
fn get_cache_size() -> Result<u64, String> {
    let (_tauri_dir, project_root) = get_paths()?;
    let cache_dir = project_root.join("tile_cache");

    if !cache_dir.exists() {
        return Ok(0);
    }

    let mut total = 0u64;

    fn walk(path: &PathBuf, total: &mut u64) -> std::io::Result<()> {
        for entry in fs::read_dir(path)? {
            let e = entry?;
            let p = e.path();
            if p.is_dir() {
                walk(&p, total)?;
            } else {
                *total += e.metadata()?.len();
            }
        }
        Ok(())
    }

    walk(&cache_dir, &mut total).map_err(|e| e.to_string())?;
    Ok(total)
}

#[command]
fn add_to_training_data(detection: SolarDetection) -> Result<String, String> {
    let (_tauri_dir, project_root) = get_paths()?;
    let training_file = project_root.join("qc_training_data.json");

    let mut data = if training_file.exists() {
        serde_json::from_str::<Vec<Value>>(
            &fs::read_to_string(&training_file).unwrap_or("[]".into())
        ).unwrap_or_default()
    } else {
        Vec::new()
    };

    data.push(json!({
        "timestamp": chrono::Local::now().to_rfc3339(),
        "confidence": detection.confidence,
        "panel_count": detection.panel_count_est,
        "area_sqm": detection.pv_area_sqm_est,
        "capacity_kw": detection.capacity_kw_est,
        "has_solar": detection.has_solar,
        "qc_notes": detection.qc_notes
    }));

    fs::write(&training_file, serde_json::to_string_pretty(&data).unwrap())
        .map_err(|e| e.to_string())?;

    Ok("Added".into())
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    tauri::Builder::default()
    .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            if let Some(win) = app.get_webview_window("main") {
                win.set_title("Terralyte")?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            fetch_and_crop_tile,
            fetch_stitched_tile,
            run_ai_analysis,
            load_overlay_image,
            save_detection_json,
            save_audit_overlay,
            clear_tile_cache,
            get_cache_size,
            add_to_training_data,
            process_csv_batch,
            save_batch_results
        ])
        .run(tauri::generate_context!())
        .expect("error running Tauri app");
}
