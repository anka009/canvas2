# canvas_final.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd

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
    h_min = max(0, np.min(h) - buffer_h)
    h_max = min(180, np.max(h) + buffer_h)
    s_min = max(0, np.min(s) - buffer_s)
    s_max = min(255, np.max(s) + buffer_s)
    v_min = max(0, np.min(v) - buffer_v)
    v_max = min(255, np.max(v) + buffer_v)
    return (h_min, h_max, s_min, s_max, v_min, v_max)

def apply_hue_wrap(hsv_img, hmin, hmax, smin, smax, vmin, vmax):
    if hmin <= hmax:
        mask = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax]))
    else:  # wrap-around hue
        mask_lo = cv2.inRange(hsv_img, np.array([0, smin, vmin]), np.array([hmax, smax, vmax]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([180, smax, vmax]))
        mask = cv2.bitwise_or(mask_lo, mask_hi)
    return mask

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler", layout="wide")
st.title("🧬 Zellkern-Zähler – Automatisch & Zwei manuelle Modi")

# -------------------- Session State --------------------
for key in ["aec_points","hema_points","bg_points","manual_aec","manual_hema","aec_hsv","hema_hsv","bg_hsv","last_file"]:
    if key not in st.session_state:
        if "points" in key or "manual" in key:
            st.session_state[key] = []
        else:
            st.session_state[key] = None
if "disp_width" not in st.session_state:
    st.session_state.disp_width = 1400

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

# Reset bei neuem Bild
if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_points","hema_points","bg_points","manual_aec","manual_hema","aec_hsv","hema_hsv","bg_hsv"]:
        st.session_state[k] = [] if "points" in k else None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1400

# -------------------- Bild vorbereiten --------------------
colW1, colW2 = st.columns([2,1])
with colW1:
    DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100, key="disp_width_slider")
    st.session_state.disp_width = DISPLAY_WIDTH

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig*scale)), interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

# -------------------- Parameter --------------------
st.markdown("### ⚙️ Filterparameter")
col1, col2, col3 = st.columns(3)
with col1:
    blur_kernel = st.slider("🔧 Blur (ungerade)", 1, 21, 5, step=2)
    min_area = st.number_input("📏 Mindestfläche", 10, 2000, 100)
with col2:
    alpha = st.slider("🌗 Alpha (Kontrast)", 0.1, 3.0, 1.0, step=0.1)
with col3:
    circle_radius = st.slider("⚪ Kreisradius", 3, 20, 8)
    line_thickness = st.slider("📏 Linienstärke", 1, 5, 2)

# -------------------- Modi --------------------
st.markdown("### 🎨 Markierungsmodi")
colA, colB, colC, colD, colE, colF = st.columns(6)
with colA: aec_mode = st.checkbox("🔴 AEC markieren (Kalibrierung)")
with colB: hema_mode = st.checkbox("🔵 Hämatoxylin markieren (Kalibrierung)")
with colC: bg_mode = st.checkbox("🖌 Hintergrund markieren")
with colD: manual_aec_mode = st.checkbox("🟠 AEC manuell")
with colE: manual_hema_mode = st.checkbox("🟣 Hämatoxylin manuell")
with colF: delete_mode = st.checkbox("🗑️ Löschen (alle Kategorien)")

# -------------------- Bildanzeige --------------------
marked_disp = image_disp.copy()
# automatische Punkte
for (x,y) in st.session_state.aec_points: cv2.circle(marked_disp,(x,y),circle_radius,(255,0,0),line_thickness)
for (x,y) in st.session_state.hema_points: cv2.circle(marked_disp,(x,y),circle_radius,(0,0,255),line_thickness)
# manuelle Punkte
for (x,y) in st.session_state.manual_aec: cv2.circle(marked_disp,(x,y),circle_radius,(255,165,0),line_thickness)
for (x,y) in st.session_state.manual_hema: cv2.circle(marked_disp,(x,y),circle_radius,(128,0,128),line_thickness)
# Hintergrundpunkte
for (x,y) in st.session_state.bg_points: cv2.circle(marked_disp,(x,y),circle_radius,(255,255,0),line_thickness)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key="clickable_image", width=DISPLAY_WIDTH)

# -------------------- Klicklogik --------------------
if coords:
    x, y = coords["x"], coords["y"]
    if delete_mode:
        for key in ["aec_points","hema_points","bg_points","manual_aec","manual_hema"]:
            st.session_state[key] = [p for p in st.session_state[key] if not is_near(p,(x,y),circle_radius)]
    elif aec_mode: st.session_state.aec_points.append((x,y))
    elif hema_mode: st.session_state.hema_points.append((x,y))
    elif bg_mode: st.session_state.bg_points.append((x,y))
    elif manual_aec_mode: st.session_state.manual_aec.append((x,y))
    elif manual_hema_mode: st.session_state.manual_hema.append((x,y))

# -------------------- Kalibrierung --------------------
col_cal1, col_cal2 = st.columns(2)
with col_cal1:
    if st.button("⚡ Kalibrierung berechnen"):
        st.session_state.aec_hsv = compute_hsv_range(st.session_state.aec_points, hsv_disp)
        st.session_state.hema_hsv = compute_hsv_range(st.session_state.hema_points, hsv_disp)
        st.session_state.bg_hsv = compute_hsv_range(st.session_state.bg_points, hsv_disp)
        st.success("Kalibrierung gespeichert.")
with col_cal2:
    if st.button("🧹 Hintergrundpunkte löschen"):
        st.session_state.bg_points = []
        st.info("Hintergrundpunkte gelöscht.")

# -------------------- Auto-Erkennung --------------------
if st.button("🤖 Auto-Erkennung"):
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
    if blur_kernel > 1: proc = cv2.GaussianBlur(proc,(blur_kernel,blur_kernel),0)
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)
    
    # AEC automatisch
    if st.session_state.aec_hsv:
        hmin,hmax,smin,smax,vmin,vmax = st.session_state.aec_hsv
        mask = apply_hue_wrap(hsv_proc,hmin,hmax,smin,smax,vmin,vmax)
        st.session_state.aec_points = get_centers(mask,min_area)
    # Hämatoxylin automatisch
    if st.session_state.hema_hsv:
        hmin,hmax,smin,smax,vmin,vmax = st.session_state.hema_hsv
        mask = apply_hue_wrap(hsv_proc,hmin,hmax,smin,smax,vmin,vmax)
        st.session_state.hema_points = get_centers(mask,min_area)

# -------------------- Anzeige der Gesamtzahlen --------------------
all_aec = st.session_state.aec_points + st.session_state.manual_aec
all_hema = st.session_state.hema_points + st.session_state.manual_hema
st.markdown(f"### 🔢 Gesamt: AEC={len(all_aec)}, Hämatoxylin={len(all_hema)}")

# -------------------- CSV Export --------------------
df_list = []
for (x,y) in all_aec: df_list.append({"X_display":x,"Y_display":y,"Type":"AEC"})
for (x,y) in all_hema: df_list.append({"X_display":x,"Y_display":y,"Type":"Hämatoxylin"})
if df_list:
    df = pd.DataFrame(df_list)
    df["X_original"] = (df["X_display"]/scale).round().astype("Int64")
    df["Y_original"] = (df["Y_display"]/scale).round().astype("Int64")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")
