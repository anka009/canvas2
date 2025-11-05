# zellkern_robust.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json
from pathlib import Path
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates

# -------------------- Hilfsfunktionen --------------------
def ensure_odd(k):
    k = int(k)
    return k if k % 2 == 1 else k + 1

def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def dedup_points(points, min_dist=6):
    out = []
    for p in points:
        if not any(is_near(p, q, r=min_dist) for q in out):
            out.append(p)
    return out

def get_centers(mask, min_area=50):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M.get("m00", 0) != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
    return centers

def safe_array_to_list(arr):
    if arr is None:
        return None
    return list(map(int, list(arr)))

def load_image(file):
    img = Image.open(file).convert("RGB")
    return np.array(img)

def apply_hue_wrap(hsv_img, hmin, hmax, smin, smax, vmin, vmax):
    if hmin <= hmax:
        return cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img, np.array([0, smin, vmin]), np.array([hmax, smax, vmax]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([180, smax, vmax]))
        return cv2.bitwise_or(mask_lo, mask_hi)

def clahe_rgb(img):
    # CLAHE on L channel via LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# -------------------- Robust HSV-Bereich (Median + IQR + Buffer) --------------------
def compute_hsv_range(points, hsv_img, sample_radius=5, buffer=(0,0,0)):
    """
    points: list of (x,y) on displayed image coords
    hsv_img: HSV image (display)
    returns: (hmin,hmax,smin,smax,vmin,vmax) or None
    """
    if not points:
        return None

    samples = []
    h_vals = []
    s_vals = []
    v_vals = []
    for (x,y) in points:
        x0 = max(0, x - sample_radius)
        x1 = min(hsv_img.shape[1], x + sample_radius + 1)
        y0 = max(0, y - sample_radius)
        y1 = min(hsv_img.shape[0], y + sample_radius + 1)
        region = hsv_img[y0:y1, x0:x1]
        if region.size:
            region = region.reshape(-1,3)
            h_vals.extend(region[:,0].tolist())
            s_vals.extend(region[:,1].tolist())
            v_vals.extend(region[:,2].tolist())

    if not h_vals:
        return None

    h = np.array(h_vals, dtype=int)
    s = np.array(s_vals, dtype=int)
    v = np.array(v_vals, dtype=int)

    # handle hue wrap-around by mapping to circular domain: try two strategies and pick the narrower range
    def circ_range(arr):
        # arr in 0..179
        med = np.median(arr)
        shifted = (arr - med + 180) % 180
        lo = np.percentile(shifted, 15)
        hi = np.percentile(shifted, 85)
        # convert back
        lo = int((lo + med - 180) % 180)
        hi = int((hi + med - 180) % 180)
        return lo, hi

    # median + IQR approach for s and v
    s_med = int(np.median(s))
    v_med = int(np.median(v))
    s_iqr = int(np.subtract(*np.percentile(s, [75,25])))
    v_iqr = int(np.subtract(*np.percentile(v, [75,25])))

    # dynamic tolerances but bounded
    s_tol = max(20, min(80, s_iqr*2))
    v_tol = max(20, min(80, v_iqr*2))

    # hue: try normal median approach then circular
    h_med = int(np.median(h))
    h_iqr = int(np.subtract(*np.percentile(h, [75,25])))
    h_tol = max(8, min(40, h_iqr*2 + 8))

    # circular correction if distribution near 0/179
    h_min_try = (h_med - h_tol) % 180
    h_max_try = (h_med + h_tol) % 180
    # also compute circ range
    h_min_circ, h_max_circ = circ_range(h)

    # choose the option yielding smaller interval length
    def interval_len(a,b):
        if a <= b:
            return b - a
        else:
            return (180 - a) + b

    if interval_len(h_min_try, h_max_try) <= interval_len(h_min_circ, h_max_circ):
        h_min, h_max = h_min_try, h_max_try
    else:
        h_min, h_max = h_min_circ, h_max_circ

    # apply buffers from sidebar
    bh, bs, bv = buffer
    h_min = int((h_min - bh) % 180)
    h_max = int((h_max + bh) % 180)
    s_min = max(0, int(s_med - s_tol - bs))
    s_max = min(255, int(s_med + s_tol + bs))
    v_min = max(0, int(v_med - v_tol - bv))
    v_max = min(255, int(v_med + v_tol + bv))

    return (h_min, h_max, s_min, s_max, v_min, v_max)

# -------------------- Save/Load calibration --------------------
CALIB_FILE = "kalib_robust.json"
def save_calib(aec, hema, bg):
    data = {"aec": safe_array_to_list(aec), "hema": safe_array_to_list(hema), "bg": safe_array_to_list(bg)}
    try:
        with open(CALIB_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        st.success("Kalibrierung gespeichert.")
    except Exception as e:
        st.error(f"Speichern fehlgeschlagen: {e}")

def load_calib():
    p = Path(CALIB_FILE)
    if not p.exists():
        st.warning("Keine gespeicherte Kalibrierung gefunden.")
        return None, None, None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        a = np.array(data.get("aec")) if data.get("aec") else None
        h = np.array(data.get("hema")) if data.get("hema") else None
        b = np.array(data.get("bg")) if data.get("bg") else None
        st.success("Kalibrierung geladen.")
        return a,h,b
    except Exception as e:
        st.error(f"Laden fehlgeschlagen: {e}")
        return None, None, None

# -------------------- Session State Defaults --------------------
if "app" not in st.session_state:
    st.session_state["app"] = {
        "aec_points": [],
        "hema_points": [],
        "bg_points": [],
        "manual_aec": [],
        "manual_hema": [],
        "aec_hsv": None,
        "hema_hsv": None,
        "bg_hsv": None,
        "last_click_id": None,
        "last_file": None,
        "display_width": 1200,
        "last_auto_run": 0
    }

APP = st.session_state["app"]

# -------------------- UI: Header & Upload --------------------
st.set_page_config(page_title="Zellkern-Zähler (robust)", layout="wide")
st.title("🧬 Zellkern-Zähler — robust & mit Masken-Vorschau")

uploaded = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded:
    st.info("Bitte ein Bild hochladen, um zu starten.")
    st.stop()

# reset on new file
if uploaded.name != APP["last_file"]:
    APP.update({
        "aec_points": [],
        "hema_points": [],
        "bg_points": [],
        "manual_aec": [],
        "manual_hema": [],
        "aec_hsv": None,
        "hema_hsv": None,
        "bg_hsv": None,
        "last_click_id": None,
        "last_auto_run": 0
    })
    APP["last_file"] = uploaded.name

# -------------------- Left: image and interactions --------------------
col1, col2 = st.columns([2,1])
with col1:
    DISPLAY_WIDTH = st.slider("Bildbreite (px)", 400, 1600, APP["display_width"], step=100)
    APP["display_width"] = DISPLAY_WIDTH

    image_orig = load_image(uploaded)
    H_orig, W_orig = image_orig.shape[:2]
    scale = DISPLAY_WIDTH / W_orig if W_orig > 0 else 1.0
    image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig * scale)), interpolation=cv2.INTER_AREA)
    image_disp = cv2.cvtColor(image_disp, cv2.COLOR_BGR2RGB) if image_disp.shape[2] == 3 else image_disp
    # optional contrast equalization
    if st.sidebar.checkbox("CLAHE (Kontrastverstärkung)", value=True):
        proc_disp = clahe_rgb(image_disp)
    else:
        proc_disp = image_disp.copy()

    hsv_disp = cv2.cvtColor(proc_disp, cv2.COLOR_RGB2HSV)
    marked = proc_disp.copy()

    # draw existing points
    def draw_points(img, points, color, r=6):
        for (x,y) in points:
            cv2.circle(img, (int(x), int(y)), r, color, -1)

    draw_points(marked, APP["aec_points"], (255,100,100), r=6)      # AEC red
    draw_points(marked, APP["hema_points"], (100,100,255), r=6)     # Hema blue
    draw_points(marked, APP["bg_points"], (255,255,0), r=6)         # bg yellow
    draw_points(marked, APP["manual_aec"], (255,165,0), r=6)
    draw_points(marked, APP["manual_hema"], (128,0,128), r=6)

    click_key = f"click_img_{APP['last_auto_run']}"
    coords = streamlit_image_coordinates(Image.fromarray(marked), key=click_key, width=DISPLAY_WIDTH)

    # robust click handling:
    if coords:
        cx, cy = int(coords["x"]), int(coords["y"])
        click_id = (cx, cy)
        last = APP.get("last_click_id")
        # only treat as new if different from last processed click
        if last != click_id:
            mode = st.sidebar.radio("Modus", ["Keine", "AEC (Kalib.)", "Hämatoxylin (Kalib.)", "Hintergrund (Kalib.)", "AEC manuell", "Hämatoxylin manuell", "Punkt löschen"], index=0)
            if mode == "Punkt löschen":
                # remove near points from all lists
                for k in ["aec_points","hema_points","bg_points","manual_aec","manual_hema"]:
                    APP[k] = [p for p in APP[k] if not is_near(p, (cx,cy), r=12)]
                st.info(f"Punkte in der Nähe von ({cx},{cy}) gelöscht.")
            elif mode == "AEC (Kalib.)":
                APP["aec_points"].append((cx,cy))
                st.success(f"AEC Kalibrierungs-Punkt: ({cx},{cy})")
            elif mode == "Hämatoxylin (Kalib.)":
                APP["hema_points"].append((cx,cy))
                st.success(f"Hämatoxylin Kalibrierungs-Punkt: ({cx},{cy})")
            elif mode == "Hintergrund (Kalib.)":
                APP["bg_points"].append((cx,cy))
                st.success(f"Hintergrund-Punkt: ({cx},{cy})")
            elif mode == "AEC manuell":
                APP["manual_aec"].append((cx,cy))
            elif mode == "Hämatoxylin manuell":
                APP["manual_hema"].append((cx,cy))
            APP["last_click_id"] = click_id
    else:
        # clear last id so a future click with same coords will be accepted
        APP["last_click_id"] = None

    # dedup
    for k in ["aec_points","hema_points","bg_points","manual_aec","manual_hema"]:
        APP[k] = dedup_points(APP[k], min_dist=6)

    st.image(marked, use_column_width=True)

# -------------------- Right: controls --------------------
with col2:
    st.sidebar.markdown("### Kalibrierung & Parameter")
    sample_radius = st.sidebar.slider("Kalibrierungs-Probenradius (px)", 1, 20, 5)
    buffer_h = st.sidebar.slider("Hue-Buffer", 0, 30, 3)
    buffer_s = st.sidebar.slider("Sättigungs-Buffer", 0, 100, 10)
    buffer_v = st.sidebar.slider("Value-Buffer", 0, 100, 10)
    blur_k = st.sidebar.slider("Blur-Kernel (ungerade)", 1, 21, 5)
    blur_k = ensure_odd(blur_k)
    min_area = st.sidebar.number_input("Mindestfläche (px)", 10, 5000, 100)
    alpha = st.sidebar.slider("Alpha (Kontrast)", 0.5, 3.0, 1.0, step=0.1)
    show_masks = st.sidebar.checkbox("Masken-Vorschau anzeigen", value=True)

    st.sidebar.markdown("### Aktionen")
    if st.sidebar.button("AEC kalibrieren"):
        if APP["aec_points"]:
            APP["aec_hsv"] = compute_hsv_range(APP["aec_points"], hsv_disp, sample_radius=sample_radius, buffer=(buffer_h,buffer_s,buffer_v))
            cnt = len(APP["aec_points"])
            APP["aec_points"] = []
            st.success(f"AEC Kalibrierung aus {cnt} Punkten gespeichert.")
        else:
            st.warning("Keine AEC-Punkte vorhanden.")

    if st.sidebar.button("Hämatoxylin kalibrieren"):
        if APP["hema_points"]:
            APP["hema_hsv"] = compute_hsv_range(APP["hema_points"], hsv_disp, sample_radius=sample_radius, buffer=(buffer_h,buffer_s,buffer_v))
            cnt = len(APP["hema_points"])
            APP["hema_points"] = []
            st.success(f"Hämatoxylin Kalibrierung aus {cnt} Punkten gespeichert.")
        else:
            st.warning("Keine Hämatoxylin-Punkte vorhanden.")

    if st.sidebar.button("Hintergrund kalibrieren"):
        if APP["bg_points"]:
            APP["bg_hsv"] = compute_hsv_range(APP["bg_points"], hsv_disp, sample_radius=sample_radius, buffer=(buffer_h,buffer_s,buffer_v))
            cnt = len(APP["bg_points"])
            APP["bg_points"] = []
            st.success(f"Hintergrund Kalibrierung aus {cnt} Punkten gespeichert.")
        else:
            st.warning("Keine Hintergrund-Punkte vorhanden.")

    st.sidebar.markdown("### Schnellaktionen")
    if st.sidebar.button("Alle Punkte löschen"):
        for k in ["aec_points","hema_points","bg_points","manual_aec","manual_hema"]:
            APP[k] = []
        st.info("Alle Punkte entfernt.")

    if st.sidebar.button("Kalibrierung zurücksetzen"):
        APP["aec_hsv"] = None
        APP["hema_hsv"] = None
        APP["bg_hsv"] = None
        st.info("Kalibrierungen zurückgesetzt.")

    if st.sidebar.button("Kalibrierung speichern"):
        save_calib(APP["aec_hsv"], APP["hema_hsv"], APP["bg_hsv"])

    if st.sidebar.button("Kalibrierung laden"):
        a,h,b = load_calib()
        APP["aec_hsv"], APP["hema_hsv"], APP["bg_hsv"] = a,h,b

    if st.sidebar.button("Auto-Erkennung ausführen"):
        APP["last_auto_run"] = (APP["last_auto_run"] or 0) + 1
        # no st.rerun(); rely on key change for image widget

# -------------------- Auto-Erkennung --------------------
# Run detection only when requested
if APP.get("last_auto_run", 0) and APP["last_auto_run"] > 0:
    proc = cv2.convertScaleAbs(proc_disp, alpha=alpha, beta=0)
    if blur_k > 1:
        proc = cv2.GaussianBlur(proc, (blur_k, blur_k), 0)
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

    # prepare masks
    mask_aec = np.zeros(hsv_proc.shape[:2], dtype=np.uint8)
    mask_hema = np.zeros(hsv_proc.shape[:2], dtype=np.uint8)

    if APP["aec_hsv"] is not None:
        hmin,hmax,smin,smax,vmin,vmax = map(int, APP["aec_hsv"])
        mask_aec = apply_hue_wrap(hsv_proc, hmin,hmax,smin,smax,vmin,vmax)
    if APP["hema_hsv"] is not None:
        hmin,hmax,smin,smax,vmin,vmax = map(int, APP["hema_hsv"])
        mask_hema = apply_hue_wrap(hsv_proc, hmin,hmax,smin,smax,vmin,vmax)

    # exclude background if available
    if APP["bg_hsv"] is not None:
        hmin,hmax,smin,smax,vmin,vmax = map(int, APP["bg_hsv"])
        mask_bg = apply_hue_wrap(hsv_proc, hmin,hmax,smin,smax,vmin,vmax)
        mask_aec = cv2.bitwise_and(mask_aec, cv2.bitwise_not(mask_bg))
        mask_hema = cv2.bitwise_and(mask_hema, cv2.bitwise_not(mask_bg))

    # morphology to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_aec = cv2.morphologyEx(mask_aec, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_aec = cv2.morphologyEx(mask_aec, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_hema = cv2.morphologyEx(mask_hema, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_hema = cv2.morphologyEx(mask_hema, cv2.MORPH_CLOSE, kernel, iterations=1)

    # get centers
    aec_centers = get_centers(mask_aec, min_area)
    hema_centers = get_centers(mask_hema, min_area)

    # remove centers near manual bg points to reduce artifacts
    if APP["bg_points"]:
        aec_centers = [p for p in aec_centers if not any(is_near(p, q, r=12) for q in APP["bg_points"])]
        hema_centers = [p for p in hema_centers if not any(is_near(p, q, r=12) for q in APP["bg_points"])]

    # merge with manual points (manual have priority)
    merged_aec = APP["manual_aec"].copy()
    for p in aec_centers:
        if not any(is_near(p, q, r=10) for q in merged_aec):
            merged_aec.append(p)
    merged_hema = APP["manual_hema"].copy()
    for p in hema_centers:
        if not any(is_near(p, q, r=10) for q in merged_hema):
            merged_hema.append(p)

    # dedup and store as display coords
    APP["aec_points"] = dedup_points(merged_aec, min_dist=6)
    APP["hema_points"] = dedup_points(merged_hema, min_dist=6)

    # show mask previews if requested
    if show_masks:
        st.markdown("### Masken-Vorschau")
        colA, colB = st.columns(2)
        with colA:
            st.image(mask_aec, caption=f"AEC Mask (detected: {len(APP['aec_points'])})", use_column_width=True)
        with colB:
            st.image(mask_hema, caption=f"Hämatoxylin Mask (detected: {len(APP['hema_points'])})", use_column_width=True)

# -------------------- Summary & CSV Export --------------------
all_aec = APP["aec_points"] or []
all_hema = APP["hema_points"] or []

st.markdown(f"### Ergebnis: AEC={len(all_aec)}, Hämatoxylin={len(all_hema)} (manuell AEC={len(APP['manual_aec'])}, manuell Hema={len(APP['manual_hema'])})")

if all_aec or all_hema:
    df_rows = []
    for x,y in all_aec:
        df_rows.append({"X_display": int(x), "Y_display": int(y), "Type":"AEC"})
    for x,y in all_hema:
        df_rows.append({"X_display": int(x), "Y_display": int(y), "Type":"Hämatoxylin"})
    df = pd.DataFrame(df_rows)
    # convert to original coords
    df["X_original"] = (df["X_display"] / scale).round().astype("Int64")
    df["Y_original"] = (df["Y_display"] / scale).round().astype("Int64")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("CSV exportieren", data=csv, file_name="zellkerne_robust.csv", mime="text/csv")

# -------------------- Debug (optional) --------------------
with st.expander("🔧 Debug Info"):
    st.write({
        "aec_hsv": APP["aec_hsv"],
        "hema_hsv": APP["hema_hsv"],
        "bg_hsv": APP["bg_hsv"],
        "aec_points_count": len(APP["aec_points"]),
        "hema_points_count": len(APP["hema_points"]),
        "manual_aec": APP["manual_aec"],
        "manual_hema": APP["manual_hema"],
        "bg_points": APP["bg_points"],
        "last_auto_run": APP["last_auto_run"],
        "last_file": APP["last_file"],
    })
