"""
LIDC-IDRI Full Pipeline: Conversion → nnU-Net Training → Prediction → 3D Browser Visualization
==============================================================================================
Fixes applied:
  - np.int deprecation patched at startup (pylidc bug with NumPy >= 1.24)
  - DICOM path read directly from folder structure — no pylidc path mismatch
  - XML parsed directly (no pylidc.scan.to_volume() call that triggered np.int)
  - pylidcrc auto-written so pylidc works if needed later

Requirements:
    pip install pylidc SimpleITK pydicom numpy scipy scikit-image plotly dash nibabel tqdm

Usage:
    python lidc_pipeline.py --mode convert      # Step 1: Convert DICOM+XML → NIfTI masks
    python lidc_pipeline.py --mode train        # Step 2: Run nnU-Net training
    python lidc_pipeline.py --mode predict      # Step 3: Run prediction on test cases
    python lidc_pipeline.py --mode visualize    # Step 4: Launch browser 3D viewer
    python lidc_pipeline.py --mode all          # Run every step
"""

# ── Patch numpy BEFORE anything else imports it ──────────────────────────────
import numpy as np
if not hasattr(np, "int"):     np.int     = int
if not hasattr(np, "float"):   np.float   = float
if not hasattr(np, "bool"):    np.bool    = bool
if not hasattr(np, "complex"): np.complex = complex
if not hasattr(np, "object"):  np.object  = object
if not hasattr(np, "str"):     np.str     = str
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, re, glob, argparse, json, shutil, subprocess, warnings
warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════
#  >>>  EDIT THIS LINE to your actual LIDC-IDRI root folder  <<<
# ═════════════════════════════════════════════════════════════
LIDC_ROOT = r"C:\Users\hslab\OneDrive\Desktop\MANUAL\final_lung\manifest-1600709154662\LIDC-IDRI"
# ═════════════════════════════════════════════════════════════

# Output workspace (created automatically beside this script)
BASE_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnunet_workspace")
RAW_DIR         = os.path.join(BASE_DIR, "nnUNet_raw")
PREPROCESSED    = os.path.join(BASE_DIR, "nnUNet_preprocessed")
RESULTS_DIR     = os.path.join(BASE_DIR, "nnUNet_results")
DATASET_NAME    = "Dataset001_Lung"
DATASET_DIR     = os.path.join(RAW_DIR, DATASET_NAME)
IMAGES_TR       = os.path.join(DATASET_DIR, "imagesTr")
LABELS_TR       = os.path.join(DATASET_DIR, "labelsTr")
IMAGES_TS       = os.path.join(DATASET_DIR, "imagesTs")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")

MAX_CASES  = None  # None = all patients; e.g. 30 for a quick test
MIN_SLICES = 10    # skip scout/localizer series with fewer slices
FOLD       = 0     # nnU-Net cross-validation fold to train/predict with


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _write_pylidcrc():
    cfg = os.path.expanduser("~/.pylidcrc")
    with open(cfg, "w") as f:
        f.write(f"[dicom]\npath = {LIDC_ROOT}\n")
    print(f"[INFO] ~/.pylidcrc  →  {LIDC_ROOT}")


def _find_best_series(patient_dir):
    """Walk patient folder and return (series_dir, sorted_dcm_paths) for the
    series with the most DICOM files (≥ MIN_SLICES). Skips scout folders."""
    best = (None, [])
    for root, dirs, files in os.walk(patient_dir):
        dcms = sorted([os.path.join(root, f)
                       for f in files if f.lower().endswith(".dcm")])
        if len(dcms) >= MIN_SLICES and len(dcms) > len(best[1]):
            best = (root, dcms)
    return best


def _parse_xml_annotations(xml_path):
    """Parse LIDC XML → list of nodule dicts with per-slice ROI polygons.
    Returns: [{'roi': [{'z': float, 'xy': [(x,y),...]},...]}]
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root_el = tree.getroot()

    def tag(el):
        return re.sub(r"\{.*?\}", "", el.tag)

    nodules = []
    for session in root_el:
        if tag(session) != "readingSession":
            continue
        for nod_el in session:
            if tag(nod_el) not in ("unblindedReadNodule", "blindedReadNodule"):
                continue
            roi_list = []
            for roi_el in nod_el:
                if tag(roi_el) != "roi":
                    continue
                z_pos, xy = None, []
                for child in roi_el:
                    ct = tag(child)
                    if ct == "imageZposition":
                        try:   z_pos = float(child.text)
                        except: pass
                    elif ct == "edgeMap":
                        xs = [float(e.text) for e in child if tag(e) == "xCoord"]
                        ys = [float(e.text) for e in child if tag(e) == "yCoord"]
                        if xs and ys:
                            xy.append((xs[0], ys[0]))
                if z_pos is not None and xy:
                    roi_list.append({"z": z_pos, "xy": xy})
            if roi_list:
                nodules.append({"roi": roi_list})
    return nodules


def _build_mask(vol_shape, z_positions, nodules):
    """Rasterise nodule ROI polygons into a (Z,Y,X) binary mask."""
    from skimage.draw import polygon as sk_polygon

    mask  = np.zeros(vol_shape, dtype=np.uint8)
    z_arr = np.array(z_positions)

    for nod in nodules:
        for roi in nod["roi"]:
            diffs   = np.abs(z_arr - roi["z"])
            nearest = int(np.argmin(diffs))
            if diffs[nearest] > 5.0:
                continue
            xs = np.array([p[0] for p in roi["xy"]])
            ys = np.array([p[1] for p in roi["xy"]])
            rr, cc = sk_polygon(ys, xs, shape=(vol_shape[1], vol_shape[2]))
            mask[nearest, rr, cc] = 1
    return mask


# ─────────────────────────────────────────────────────────────
# STEP 1 – Convert LIDC DICOM + XML → NIfTI
# ─────────────────────────────────────────────────────────────

def convert_lidc_to_nnunet():
    try:
        import pydicom
        import SimpleITK as sitk
        from tqdm import tqdm
    except ImportError as e:
        print(f"[ERROR] {e}\nInstall: pip install pydicom SimpleITK tqdm scikit-image")
        sys.exit(1)

    _write_pylidcrc()

    for d in (IMAGES_TR, LABELS_TR, IMAGES_TS):
        os.makedirs(d, exist_ok=True)

    if not os.path.isdir(LIDC_ROOT):
        print(f"[ERROR] LIDC_ROOT not found:\n  {LIDC_ROOT}")
        print("Edit LIDC_ROOT at the top of this script.")
        sys.exit(1)

    patient_dirs = sorted([
        os.path.join(LIDC_ROOT, d) for d in os.listdir(LIDC_ROOT)
        if d.startswith("LIDC-IDRI-") and
           os.path.isdir(os.path.join(LIDC_ROOT, d))
    ])
    if MAX_CASES:
        patient_dirs = patient_dirs[:MAX_CASES]

    print(f"[INFO] {len(patient_dirs)} patient folders found.")

    case_ids, skipped, case_idx = [], 0, 0

    for patient_dir in tqdm(patient_dirs, desc="Converting"):
        patient_id = os.path.basename(patient_dir)

        series_dir, dcm_files = _find_best_series(patient_dir)
        if not dcm_files:
            skipped += 1
            continue

        # Find XML (search recursively inside patient folder)
        xml_files = glob.glob(os.path.join(patient_dir, "**", "*.xml"), recursive=True)
        xml_path  = xml_files[0] if xml_files else None

        try:
            # Load DICOM slices
            slices = []
            for f in dcm_files:
                try:
                    slices.append(pydicom.dcmread(f))
                except Exception:
                    continue

            if len(slices) < MIN_SLICES:
                skipped += 1
                continue

            # Sort by z-position
            try:
                slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            except Exception:
                try:
                    slices.sort(key=lambda s: int(s.InstanceNumber))
                except Exception:
                    pass

            vol_list, z_pos_list = [], []
            pixel_spacing = 1.0
            slice_thick   = 1.0

            for s in slices:
                try:
                    arr       = s.pixel_array.astype(np.float32)
                    slope     = float(getattr(s, "RescaleSlope",     1))
                    intercept = float(getattr(s, "RescaleIntercept", 0))
                    vol_list.append(arr * slope + intercept)

                    try:    z_pos_list.append(float(s.ImagePositionPatient[2]))
                    except: z_pos_list.append(float(len(z_pos_list)))

                    try:    pixel_spacing = float(s.PixelSpacing[0])
                    except: pass
                    try:    slice_thick   = float(s.SliceThickness)
                    except: pass
                except Exception:
                    continue

            if len(vol_list) < MIN_SLICES:
                skipped += 1
                continue

            volume = np.stack(vol_list, axis=0)   # (Z, Y, X)

            # Build mask from XML
            mask = np.zeros(volume.shape, dtype=np.uint8)
            if xml_path:
                try:
                    nodules = _parse_xml_annotations(xml_path)
                    if nodules:
                        mask = _build_mask(volume.shape, z_pos_list, nodules)
                except Exception as e:
                    print(f"[WARN] XML failed for {patient_id}: {e}")

            # Write NIfTI
            case_id  = f"case_{case_idx:04d}"
            img_sitk = sitk.GetImageFromArray(volume.astype(np.float32))
            img_sitk.SetSpacing([pixel_spacing, pixel_spacing, slice_thick])
            lbl_sitk = sitk.GetImageFromArray(mask)
            lbl_sitk.SetSpacing([pixel_spacing, pixel_spacing, slice_thick])

            if case_idx < int(len(patient_dirs) * 0.8):
                sitk.WriteImage(img_sitk, os.path.join(IMAGES_TR, f"{case_id}_0000.nii.gz"))
                sitk.WriteImage(lbl_sitk, os.path.join(LABELS_TR, f"{case_id}.nii.gz"))
            else:
                sitk.WriteImage(img_sitk, os.path.join(IMAGES_TS, f"{case_id}_0000.nii.gz"))

            case_ids.append({"id": case_id, "patient": patient_id,
                             "nodule_voxels": int(mask.sum()),
                             "slices": len(vol_list)})
            case_idx += 1

        except Exception as e:
            print(f"[WARN] {patient_id}: {e}")
            skipped += 1

    print(f"\n[INFO] Converted {len(case_ids)}, skipped {skipped}.")
    with_nod = sum(1 for c in case_ids if c["nodule_voxels"] > 0)
    print(f"[INFO] Cases WITH nodule masks: {with_nod}/{len(case_ids)}")

    # dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels":        {"background": 0, "nodule": 1},
        "numTraining":   len(os.listdir(IMAGES_TR)),
        "file_ending":   ".nii.gz",
        "name":          DATASET_NAME,
        "description":   "LIDC-IDRI lung nodule segmentation",
        "tensorImageSize": "3D",
        "reference":     "https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI",
        "licence":       "CC BY 3.0",
        "relase":        "1.0",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    with open(os.path.join(DATASET_DIR, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    with open(os.path.join(BASE_DIR, "case_index.json"), "w") as f:
        json.dump(case_ids, f, indent=2)

    print(f"[OK] dataset.json → {DATASET_DIR}")
    print("[NEXT] python lidc_pipeline.py --mode train")


# ─────────────────────────────────────────────────────────────
# STEP 2 – Train nnU-Net
# ─────────────────────────────────────────────────────────────

def train_nnunet():
    env = os.environ.copy()
    env["nnUNet_raw"]          = RAW_DIR
    env["nnUNet_preprocessed"] = PREPROCESSED
    env["nnUNet_results"]      = RESULTS_DIR

    dataset_id = DATASET_NAME.split("_")[0].replace("Dataset", "")  # "001"

    def run(cmd):
        print(f"\n[CMD] {cmd}")
        r = subprocess.run(cmd, shell=True, env=env)
        if r.returncode != 0:
            sys.exit(r.returncode)

    print("[2a] Planning and preprocessing …")
    run(f"nnUNetv2_plan_and_preprocess -d {dataset_id} -c 3d_fullres --verify_dataset_integrity")

    print("[2b] Training fold 0 …")
    run(f"nnUNetv2_train {dataset_id} 3d_fullres {FOLD} --npz")

    print("[OK] Training done.")
    print("[NEXT] python lidc_pipeline.py --mode predict")


# ─────────────────────────────────────────────────────────────
# STEP 3 – Predict + False Positive Reduction
# ─────────────────────────────────────────────────────────────

def predict_and_filter():
    import SimpleITK as sitk
    from scipy import ndimage
    from tqdm import tqdm

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    raw_pred = os.path.join(PREDICTIONS_DIR, "raw")
    os.makedirs(raw_pred, exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_raw"]          = RAW_DIR
    env["nnUNet_preprocessed"] = PREPROCESSED
    env["nnUNet_results"]      = RESULTS_DIR
    dataset_id = DATASET_NAME.split("_")[0].replace("Dataset", "")

    cmd = (
    f"nnUNetv2_predict "
    f"-i {IMAGES_TS} "
    f"-o {raw_pred} "
    f"-d {dataset_id} "
    f"-c 3d_fullres "
    f"-f {FOLD} "
    f"-tr nnUNetTrainer "
    f"-chk checkpoint_best.pth")
    print(f"[CMD] {cmd}")
    subprocess.run(cmd, shell=True, env=env)

    filtered_dir = os.path.join(PREDICTIONS_DIR, "filtered")
    os.makedirs(filtered_dir, exist_ok=True)

    pred_files = [f for f in os.listdir(raw_pred) if f.endswith(".nii.gz")]
    all_stats  = {}

    for fname in tqdm(pred_files, desc="FP filter"):
        sitk_mask = sitk.ReadImage(os.path.join(raw_pred, fname))
        spacing   = sitk_mask.GetSpacing()
        mask      = sitk.GetArrayFromImage(sitk_mask).astype(np.uint8)
        vox_vol   = spacing[0] * spacing[1] * spacing[2]

        labeled, n = ndimage.label(mask)
        filtered   = np.zeros_like(mask)
        nodules    = []

        for lid in range(1, n + 1):
            comp    = labeled == lid
            vol_mm3 = comp.sum() * vox_vol
            diam    = 2 * ((3 * vol_mm3) / (4 * np.pi)) ** (1 / 3)
            if diam < 3.0 or diam > 35.0:
                continue

            coords = np.argwhere(comp)
            ranges = np.sort((coords.max(0) - coords.min(0) + 1).astype(float))
            if ranges[2] / max(ranges[0], 1) > 3.5:
                continue

            filtered[comp] = 1
            nodules.append({
                "id":           lid,
                "volume_mm3":   round(vol_mm3, 2),
                "diameter_mm":  round(diam, 2),
                "centroid_zyx": [round(c, 1) for c in coords.mean(0).tolist()],
            })

        out = sitk.GetImageFromArray(filtered)
        out.CopyInformation(sitk_mask)
        sitk.WriteImage(out, os.path.join(filtered_dir, fname))
        all_stats[fname] = nodules

    stats_path = os.path.join(PREDICTIONS_DIR, "nodule_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"[OK] Filtered → {filtered_dir}")
    print(f"[OK] Stats    → {stats_path}")
    print("[NEXT] python lidc_pipeline.py --mode visualize")


# ─────────────────────────────────────────────────────────────
# STEP 4 – Interactive 3D browser viewer
# ─────────────────────────────────────────────────────────────

def launch_viewer():
    try:
        import dash
        from dash import dcc, html, Input, Output
        import plotly.graph_objects as go
        import nibabel as nib
        from skimage.measure import marching_cubes
    except ImportError as e:
        print(f"[ERROR] {e}\nInstall: pip install dash plotly nibabel scikit-image")
        sys.exit(1)

    filtered_dir = os.path.join(PREDICTIONS_DIR, "filtered")
    stats_path   = os.path.join(PREDICTIONS_DIR, "nodule_stats.json")

    pred_files = (sorted([f for f in os.listdir(filtered_dir) if f.endswith(".nii.gz")])
                  if os.path.isdir(filtered_dir) else [])

    if not pred_files:
        print("[INFO] No predictions found – running in DEMO mode.")
        _create_demo_case(filtered_dir)
        pred_files = ["demo_case.nii.gz"]

    nodule_stats = {}
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            nodule_stats = json.load(f)

    COLORS = ["#FF4B4B","#FF8C42","#FFD166","#06D6A0",
              "#118AB2","#A239CA","#F72585","#4CC9F0"]

    def build_figure(case_name, show_gt=False):
        mask_path = os.path.join(filtered_dir, case_name)
        if not os.path.exists(mask_path):
            return go.Figure()

        from scipy.ndimage import label as ndlabel
        mask_data = nib.load(mask_path).get_fdata().astype(np.uint8)
        labeled, n = ndlabel(mask_data)
        stats = nodule_stats.get(case_name, [])
        fig   = go.Figure()

        for lid in range(1, n + 1):
            comp = (labeled == lid).astype(np.float32)
            if comp.sum() < 5:
                continue
            try:
                verts, faces, _, _ = marching_cubes(comp, level=0.5)
            except Exception:
                continue

            color = COLORS[(lid - 1) % len(COLORS)]
            stat  = next((s for s in stats if s["id"] == lid), None)
            diam  = stat["diameter_mm"] if stat else "?"
            vol   = stat["volume_mm3"]  if stat else "?"

            fig.add_trace(go.Mesh3d(
                x=verts[:,2], y=verts[:,1], z=verts[:,0],
                i=faces[:,0], j=faces[:,1], k=faces[:,2],
                color=color, opacity=0.85,
                name=f"Nodule {lid}  Ø{diam}mm",
                hovertemplate=(
                    f"<b>Predicted Nodule {lid}</b><br>"
                    f"Diameter: {diam} mm<br>"
                    f"Volume: {vol} mm³<extra></extra>"
                ),
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.5),
            ))

        if show_gt:
            gt_path = os.path.join(LABELS_TR, case_name)
            if os.path.exists(gt_path):
                gt = nib.load(gt_path).get_fdata().astype(np.uint8)
                if gt.sum() > 0:
                    try:
                        verts, faces, _, _ = marching_cubes(gt.astype(float), level=0.5)
                        fig.add_trace(go.Mesh3d(
                            x=verts[:,2], y=verts[:,1], z=verts[:,0],
                            i=faces[:,0], j=faces[:,1], k=faces[:,2],
                            color="#00BFFF", opacity=0.25,
                            name="Ground Truth",
                            hovertemplate="<b>Ground Truth</b><extra></extra>",
                        ))
                    except Exception:
                        pass

        fig.update_layout(
            paper_bgcolor="#0d1117",
            scene=dict(
                bgcolor="#0d1117",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(font=dict(color="#e0e0e0", size=11),
                        bgcolor="rgba(0,0,0,0.5)",
                        bordercolor="#333", borderwidth=1),
            height=680,
        )
        return fig

    # ── Dash app ──────────────────────────────────────────────────────
    app = dash.Dash(__name__, title="Lung Nodule 3D Viewer")

    gt_available = os.path.isdir(LABELS_TR)

    app.layout = html.Div(
        style={"backgroundColor":"#0d1117","minHeight":"100vh",
               "fontFamily":"monospace","color":"#e0e0e0"},
        children=[
            html.Div(style={
                "background":"linear-gradient(135deg,#1a1f2e,#0d1117)",
                "borderBottom":"1px solid #30363d","padding":"18px 28px",
                "display":"flex","alignItems":"center","gap":"14px",
            }, children=[
                html.Div("🫁", style={"fontSize":"30px"}),
                html.Div([
                    html.H1("Lung Nodule 3D Viewer",
                            style={"margin":"0","fontSize":"20px",
                                   "color":"#58a6ff","letterSpacing":"1px"}),
                    html.P("nnU-Net 3d_fullres · FP-filtered · LIDC-IDRI",
                           style={"margin":"2px 0 0","fontSize":"11px","color":"#8b949e"}),
                ]),
            ]),

            html.Div(style={
                "padding":"12px 28px","background":"#161b22",
                "borderBottom":"1px solid #30363d","display":"flex",
                "alignItems":"center","gap":"20px","flexWrap":"wrap",
            }, children=[
                html.Label("Case:", style={"color":"#8b949e","fontSize":"13px"}),
                dcc.Dropdown(
                    id="case-dd",
                    options=[{"label":f,"value":f} for f in pred_files],
                    value=pred_files[0] if pred_files else None,
                    clearable=False,
                    style={"width":"340px","backgroundColor":"#21262d",
                           "color":"#e0e0e0","border":"1px solid #30363d",
                           "borderRadius":"6px"},
                ),
                (dcc.Checklist(
                    id="gt-toggle",
                    options=[{"label":"  Show Ground Truth (blue)","value":"gt"}],
                    value=[],
                    style={"color":"#8b949e","fontSize":"13px"},
                ) if gt_available else html.Span(id="gt-toggle")),
                html.Div(id="status-text",
                         style={"fontSize":"13px","color":"#3fb950","marginLeft":"auto"}),
            ]),

            html.Div(style={"display":"flex","height":"calc(100vh - 145px)"},
                     children=[
                html.Div(style={"flex":"1","padding":"8px"},
                         children=[dcc.Graph(id="viewer-3d",
                                             style={"height":"100%"})]),

                html.Div(style={
                    "width":"290px","borderLeft":"1px solid #30363d",
                    "background":"#161b22","padding":"18px","overflowY":"auto",
                }, children=[
                    html.H3("Detected Nodules",
                            style={"color":"#58a6ff","fontSize":"14px","marginTop":"0"}),
                    html.Div(id="nodule-table"),
                    html.Hr(style={"borderColor":"#30363d","margin":"18px 0"}),
                    html.H3("Risk Guide",style={"color":"#8b949e","fontSize":"12px"}),
                    *[html.Div(t, style={"fontSize":"11px","color":"#8b949e","marginBottom":"4px"})
                      for t in ["🟢  < 6mm   →  Low / follow-up",
                                "🟡  6–10mm  →  Moderate risk",
                                "🔴  > 10mm  →  High / biopsy consult"]],
                    html.Hr(style={"borderColor":"#30363d","margin":"18px 0"}),
                    html.H3("FP Filters Applied",style={"color":"#8b949e","fontSize":"12px"}),
                    *[html.Div(t, style={"fontSize":"11px","color":"#8b949e","marginBottom":"3px"})
                      for t in ["✓ Size: 3–35 mm only",
                                "✓ Aspect ratio < 3.5 (no vessels)",
                                "✓ Connected component labeling"]],
                ]),
            ]),
        ]
    )

    @app.callback(
        Output("viewer-3d","figure"),
        Output("nodule-table","children"),
        Output("status-text","children"),
        Input("case-dd","value"),
        Input("gt-toggle","value"),
    )
    def update(case_name, gt_vals):
        if not case_name:
            return go.Figure(), [], ""
        fig   = build_figure(case_name, "gt" in (gt_vals or []))
        stats = nodule_stats.get(case_name, [])
        rows  = []
        for s in stats:
            d     = s["diameter_mm"]
            color = "#FF4B4B" if d>10 else "#FFD166" if d>6 else "#3fb950"
            risk  = "HIGH" if d>10 else "MEDIUM" if d>6 else "LOW"
            rows.append(html.Div(style={
                "background":"#21262d",
                "border":f"1px solid {color}33",
                "borderLeft":f"3px solid {color}",
                "borderRadius":"4px","padding":"10px","marginBottom":"8px",
            }, children=[
                html.Div(f"Nodule {s['id']}",
                         style={"fontWeight":"bold","fontSize":"13px","color":color}),
                html.Div(f"Ø {d} mm",
                         style={"fontSize":"12px","color":"#e0e0e0"}),
                html.Div(f"Vol: {s['volume_mm3']} mm³",
                         style={"fontSize":"11px","color":"#8b949e"}),
                html.Div(f"Risk: {risk}",
                         style={"fontSize":"11px","color":color}),
                html.Div(
                    f"Z={s['centroid_zyx'][0]:.0f}  "
                    f"Y={s['centroid_zyx'][1]:.0f}  "
                    f"X={s['centroid_zyx'][2]:.0f}",
                    style={"fontSize":"10px","color":"#484f58","marginTop":"4px"}),
            ]))
        return fig, rows, f"✓ {len(stats)} nodule(s) after FP filtering"

    print("\n" + "="*60)
    print("  🫁  Lung Nodule 3D Viewer  →  http://127.0.0.1:8050")
    print("="*60 + "\n")
    app.run(debug=False, port=8050)


# ─────────────────────────────────────────────────────────────
# Demo phantom
# ─────────────────────────────────────────────────────────────

def _create_demo_case(out_dir):
    try:
        import nibabel as nib
    except ImportError:
        return
    os.makedirs(out_dir, exist_ok=True)

    mask = np.zeros((128,128,128), dtype=np.uint8)
    def sphere(cx,cy,cz,r):
        Z,Y,X = np.ogrid[:128,:128,:128]
        return (Z-cz)**2+(Y-cy)**2+(X-cx)**2 <= r**2

    mask[sphere(64,64,64,8)] = 1   # 16 mm HIGH
    mask[sphere(40,50,70,4)] = 1   #  8 mm MEDIUM
    mask[sphere(80,90,45,2)] = 1   #  4 mm LOW

    nib.save(nib.Nifti1Image(mask, np.eye(4)),
             os.path.join(out_dir, "demo_case.nii.gz"))

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    with open(os.path.join(PREDICTIONS_DIR, "nodule_stats.json"), "w") as f:
        json.dump({"demo_case.nii.gz": [
            {"id":1,"volume_mm3":2145.0,"diameter_mm":16.0,"centroid_zyx":[64.0,64.0,64.0]},
            {"id":2,"volume_mm3": 268.0,"diameter_mm": 8.0,"centroid_zyx":[40.0,50.0,70.0]},
            {"id":3,"volume_mm3":  33.5,"diameter_mm": 4.0,"centroid_zyx":[80.0,90.0,45.0]},
        ]}, f, indent=2)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["convert","train","predict","visualize","all"],
                        default="visualize")
    args = parser.parse_args()
    os.makedirs(BASE_DIR, exist_ok=True)

    if args.mode in ("convert","all"):
        print("\n"+"="*60+"\n  STEP 1/4 – Convert LIDC → nnU-Net\n"+"="*60)
        convert_lidc_to_nnunet()
    if args.mode in ("train","all"):
        print("\n"+"="*60+"\n  STEP 2/4 – Train nnU-Net 3d_fullres\n"+"="*60)
        train_nnunet()
    if args.mode in ("predict","all"):
        print("\n"+"="*60+"\n  STEP 3/4 – Predict + FP Reduction\n"+"="*60)
        predict_and_filter()
    if args.mode in ("visualize","all"):
        print("\n"+"="*60+"\n  STEP 4/4 – 3D Browser Viewer\n"+"="*60)
        launch_viewer()

if __name__ == "__main__":
    main()