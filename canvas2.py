# canvas_bg_debug.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide")
st.title("🛠️ Robust: Hintergrundsubtraktion Debug / Demo")

# --- Upload ---
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.stop()

# --- Load image (RGB) and prepare display/scales ---
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H, W = image_orig.shape[:2]
disp_w = st.sidebar.slider("Anzeige-Breite", 300, 1400, 800)
scale = disp_w / W
image_disp = cv2.resize(image_orig, (disp_w, int(H * scale)), interpolation=cv2.INTER_AREA)

# Convert to HSV once (original size)
hsv = cv2.cvtColor(image_orig, cv2.COLOR_RGB2HSV)

# --- session state for background points ---
if "bg_points" not in st.session_state:
    st.session_state.bg_points = []

# UI controls
st.sidebar.markdown("### Methode")
method = st.sidebar.selectbox("Subtraktions-Modus", ["S/V Subtraktion (default)", "Skalierung (multiplikativ)"])
apply_btn = st.sidebar.button("Anwenden: Hintergrund entfernen")
clear_btn = st.sidebar.button("Alle BG-Punkte löschen")
show_debug = st.sidebar.checkbox("Debug Informationen anzeigen", True)
use_clahe_if_dark = st.sidebar.checkbox("CLAHE auf V anwenden, falls Ergebnis zu dunkel", True)
clahe_clip = st.sidebar.slider("CLAHE clipLimit", 1.0, 5.0, 2.0, 0.5)
clahe_tile = st.sidebar.slider("CLAHE tileGridSize", 4, 16, 8, 2)

# Draw existing bg points on display
vis = image_disp.copy()
for (x, y) in st.session_state.bg_points:
    cv2.circle(vis, (int(x * scale), int(y * scale)), 6, (255, 255, 0), 2)

# Click to add background point (coordinates converted to original)
coords = streamlit_image_coordinates(Image.fromarray(vis), width=disp_w, key="bg_debug_click")
if coords:
    x_disp = int(coords["x"]); y_disp = int(coords["y"])
    x_orig = int(round(x_disp / scale)); y_orig = int(round(y_disp / scale))
    # safety clamp
    x_orig = max(0, min(W - 1, x_orig))
    y_orig = max(0, min(H - 1, y_orig))
    st.session_state.bg_points.append((x_orig, y_orig))

# Buttons for clear
if clear_btn:
    st.session_state.bg_points = []

# Show selection preview
st.markdown("#### Hintergrundpunkte (Original-Koordinaten)")
st.write(st.session_state.bg_points)

# Show main images
col1, col2 = st.columns(2)
with col1:
    st.image(image_disp, caption="Original (skaliert, nur zur Ansicht)", use_column_width=True)
with col2:
    st.image(vis, caption="Mit gesetzten BG-Punkten (gelb)", use_column_width=True)

# If apply requested and there are bg points -> compute and visualize
if apply_btn:
    if not st.session_state.bg_points:
        st.warning("Keine Hintergrundpunkte gesetzt — bitte Punkte im Bild anklicken.")
    else:
        # collect bg pixels
        bg_vals = np.array([hsv[y, x] for (x, y) in st.session_state.bg_points])  # shape (N,3)
        # diagnostics
        h_mean, s_mean, v_mean = bg_vals.mean(axis=0)
        h_std, s_std, v_std = bg_vals.std(axis=0)
        if show_debug:
            st.markdown("**Debug: Hintergrundpunkte Stats**")
            st.write({
                "Anzahl Punkte": int(bg_vals.shape[0]),
                "H_mean, S_mean, V_mean": [float(h_mean), float(s_mean), float(v_mean)],
                "H_std, S_std, V_std": [float(h_std), float(s_std), float(v_std)],
                "dtype hsv": str(hsv.dtype),
                "hsv shape": hsv.shape
            })

        # Create working copy as float to avoid overflow/underflow
        hsv_float = hsv.astype(np.float32)

        if method == "S/V Subtraktion (default)":
            # Subtract S and V mean only (H left unchanged), more conservative than subtracting H
            hsv_float[..., 1] = hsv_float[..., 1] - float(s_mean)
            hsv_float[..., 2] = hsv_float[..., 2] - float(v_mean)

            # Clip to valid ranges, then cast back
            hsv_sub = np.empty_like(hsv, dtype=np.uint8)
            # H channel: keep original (converted safely)
            hsv_sub[..., 0] = np.clip(hsv_float[..., 0], 0, 180).astype(np.uint8)
            hsv_sub[..., 1] = np.clip(hsv_float[..., 1], 0, 255).astype(np.uint8)
            hsv_sub[..., 2] = np.clip(hsv_float[..., 2], 0, 255).astype(np.uint8)
        else:
            # multiplicative scaling: compute factor from bg mean's V (avoid division by zero)
            factor = 1.0
            if v_mean > 1e-3:
                # aim to scale image V so that bg mean becomes a mid-grey value (e.g. 128)
                target = 128.0
                factor = target / float(v_mean)
            if show_debug:
                st.write(f"Skalierungsfaktor für V: {factor:.3f}")
            hsv_float[..., 2] = hsv_float[..., 2] * factor
            # clip and cast
            hsv_sub = np.empty_like(hsv, dtype=np.uint8)
            hsv_sub[..., 0] = np.clip(hsv_float[..., 0], 0, 180).astype(np.uint8)
            hsv_sub[..., 1] = np.clip(hsv_float[..., 1], 0, 255).astype(np.uint8)
            hsv_sub[..., 2] = np.clip(hsv_float[..., 2], 0, 255).astype(np.uint8)

        # Convert to RGB for visualization
        rgb_sub = cv2.cvtColor(hsv_sub, cv2.COLOR_HSV2RGB)

        # If result is too dark, optionally apply CLAHE on V channel to bring back contrast
        avg_brightness = rgb_sub.mean()
        if show_debug:
            st.write(f"avg brightness after sub: {avg_brightness:.1f} (0..255)")
        if use_clahe_if_dark and avg_brightness < 30:
            if show_debug: st.write("Ergebnis zu dunkel — CLAHE auf V wird angewendet.")
            # apply CLAHE on V in HSV space
            hsv_clahe = hsv_sub.copy()
            clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_tile),int(clahe_tile)))
            v_chan = clahe.apply(hsv_clahe[..., 2])
            hsv_clahe[..., 2] = v_chan
            rgb_sub = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2RGB)
            if show_debug:
                st.write("CLAHE angewendet.")

        # Build difference image for quick visual check
        diff = cv2.absdiff(image_orig, rgb_sub)
        # resize for display
        before = cv2.resize(image_orig, (disp_w, int(H * scale)))
        after = cv2.resize(rgb_sub, (disp_w, int(H * scale)))
        diff_disp = cv2.resize(diff, (disp_w, int(H * scale)))
        combined = np.hstack([before, after, diff_disp])
        st.image(combined, caption="Links: Original | Mitte: Hintergrund entfernt | Rechts: Differenz", use_column_width=True)

        # Show histograms of V channel before/after
        colA, colB = st.columns(2)
        v_before = cv2.cvtColor(image_orig, cv2.COLOR_RGB2HSV)[...,2].ravel()
        v_after = cv2.cvtColor(rgb_sub, cv2.COLOR_RGB2HSV)[...,2].ravel()
        colA.write("Histogram V (original)")
        colA.write(np.histogram(v_before, bins=32)[0].tolist())
        colB.write("Histogram V (after)")
        colB.write(np.histogram(v_after, bins=32)[0].tolist())

        st.success("Hintergrundsubtraktion ausgeführt. Schau dir die Differenz an und ändere Methode/Parameter, falls nötig.")
