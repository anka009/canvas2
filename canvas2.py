# canvas-nervt_final_v2.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

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

def compute_hsv_range(points, hsv_img, buffer_h=8, buffer_s=30, buffer_v=25):
    if not points:
        return None
    vals = np.array([hsv_img[y, x] for (x, y) in points])
    h = vals[:,0].astype(int)
    s = vals[:,1].astype(int)
    v = vals[:,2].astype(int)

    h_min = int(max(0, np.min(h) - buffer_h))
    h_max = int(min(180, np.max(h) + buffer_h))
    s_min = int(max(0, np.min(s) - buffer_s))
    s_max = int(min(255, np.max(s) + buffer_s))
    v_min = int(max(0, np.min(v) - buffer_v))
    v_max = int(min(255, np.max(v) + buffer_v))

    return (h_min, h_max, s_min, s_max, v_min, v_max)

def apply_hue_wrap(hsv_img, hmin, hmax, smin, smax, vmin, vmax):
    if hmin <= hmax:
        mask = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img, np.array([0, smin, vmin]), np.array([hmax, smax, vmax]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([180, smax, vmax]))
        mask = cv2.bitwise_or(mask_lo, mask_hi)
    return mask

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler mit 2 manuellen Modi", layout="wide")
st.title("🧬 Zellkern-Zähler – Automatisch & Zwei manuelle Modi")

# -------------------- Session state defaults --------------------
defaults = {
    "aec_points": [], "hema_points": [], "bg_points": [],
    "manual_aec": [], "manual_hema": [],
    "aec_hsv": None, "hema_hsv": None, "bg_hsv": None,
    "last_file": None, "disp_width": 1400,
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

# Reset session state when new file
if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_points","hema_points","bg_points","manual_aec","manual_hema","aec_hsv","hema_hsv","bg_hsv"]:
        st.session_state[k] = [] if "points" in k else None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1400

# -------------------- Bildgröße / scale --------------------
colW1, colW2 = st.columns([2,1])
with colW1:
    DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100, key="disp_width_slider")
    st.session_state.disp_width = DISPLAY_WIDTH

# -------------------- Bildvorbereitung --------------------
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig * scale)), interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

# -------------------- Parameter-Slider --------------------
st.markdown("### ⚙️ Filter- und Erkennungsparameter")
col1, col2, col3 = st.columns(3)
with col1:
    blur_kernel = st.slider("🔧 Blur (ungerade)", 1, 21, 5, step=2, key="blur_slider")
    min_area = st.number_input("📏 Mindestfläche (px)", 10, 2000, 100, key="min_area_input")
with col2:
    alpha = st.slider("🌗 Alpha (Kontrast)", 0.1, 3.0, 1.0, step=0.1, key="alpha_slider")
with col3:
    circle_radius = st.slider("⚪ Kreisradius (Anzeige)", 3, 20, 8, key="circle_slider")
    line_thickness = st.slider("📏 Linienstärke (Anzeige)", 1, 5, 2, key="thickness_slider")

# -------------------- Kalibrierungs-Modi --------------------
st.markdown("### 🎨 Kalibrierung & manuelle Modi")
colA, colB, colC, colD, colE = st.columns(5)
with colA:
    aec_mode = st.checkbox("🔴 AEC markieren (Kalibrierung)", key="aec_mode_cb")
with colB:
    hema_mode = st.checkbox("🔵 Hämatoxylin markieren (Kalibrierung)", key="hema_mode_cb")
with colC:
    bg_mode = st.checkbox("🖌 Hintergrund markieren", key="bg_mode_cb")
with colD:
    manual_aec_mode = st.checkbox("🟠 AEC manuell", key="manual_aec_cb")
with colE:
    manual_hema_mode = st.checkbox("🟣 Hämatoxylin manuell", key="manual_hema_cb")

delete_mode = st.checkbox("🗑️ Löschmodus (alle Kategorien)", key="delete_mode_cb")

# -------------------- Anzeige: markierte Punkte --------------------
marked_disp = image_disp.copy()
# automatische Punkte
for (x,y) in st.session_state.aec_points:
    cv2.circle(marked_disp, (x,y), circle_radius, (255,0,0), line_thickness)
for (x,y) in st.session_state.hema_points:
    cv2.circle(marked_disp, (x,y), circle_radius, (0,0,255), line_thickness)
# manuelle Punkte
for (x,y) in st.session_state.manual_aec:
    cv2.circle(marked_disp, (x,y), circle_radius, (255,165,0), line_thickness)
for (x,y) in st.session_state.manual_hema:
    cv2.circle(marked_disp, (x,y), circle_radius, (128,0,128), line_thickness)
# Hintergrund
for (x,y) in st.session_state.bg_points:
    cv2.circle(marked_disp, (x,y), circle_radius, (255,255,0), line_thickness)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key="clickable_image", width=DISPLAY_WIDTH)

# -------------------- Klick-Logik --------------------
if coords is not None:
    x, y = coords["x"], coords["y"]
    if delete_mode:
        for list_name in ["aec_points","hema_points","bg_points","manual_aec","manual_hema"]:
            st.session_state[list_name] = [p for p in st.session_state[list_name] if not is_near(p,(x,y), r=circle_radius)]
    elif aec_mode:
        st.session_state.aec_points.append((x,y))
    elif hema_mode:
        st.session_state.hema_points.append((x,y))
    elif bg_mode:
        st.session_state.bg_points.append((x,y))
    elif manual_aec_mode:
        st.session_state.manual_aec.append((x,y))
    elif manual_hema_mode:
        st.session_state.manual_hema.append((x,y))

# -------------------- Kalibrierung Button --------------------
col_cal1, col_cal2 = st.columns(2)
with col_cal1:
    if st.button("⚡ Kalibrierung aus markierten Punkten"):
        st.session_state.aec_hsv = compute_hsv_range(st.session_state.aec_points, hsv_disp)
        st.session_state.hema_hsv = compute_hsv_range(st.session_state.hema_points, hsv_disp)
        st.session_state.bg_hsv  = compute_hsv_range(st.session_state.bg_points, hsv_disp)
        st.success("Kalibrierung berechnet.")
with col_cal2:
    if st.button("🧹 Hintergrundpunkte löschen"):
        st.session_state.bg_points = []
        st.info("Hintergrundpunkte gelöscht.")

# -------------------- Auto-Erkennung --------------------
if st.button("🤖 Auto-Erkennung"):
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc, (blur_kernel, blur_kernel), 0)
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

    # Hintergrundmaske
    if st.session_state.bg_hsv:
        bhmin,bhmax,bsmin,bsmax,bvmin,bvmax = st.session_state.bg_hsv
        bg_mask = apply_hue_wrap(hsv_proc, bhmin,bhmax,bsmin, bsmax, bvmin, bvmax)
