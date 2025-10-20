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

def dedup_points(points, min_dist=5):
    """Doppelte Punkte innerhalb min_dist entfernen"""
    out = []
    for p in points:
        if all(np.linalg.norm(np.array(p)-np.array(q)) >= min_dist for q in out):
            out.append(p)
    return out

def get_centers(mask, min_area=50):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M.get("m00",0) != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                centers.append((cx,cy))
    return centers

def compute_hsv_range_circle(points, hsv_img, radius=5, buffer_h=8, buffer_s=30, buffer_v=25):
    H, W = hsv_img.shape[:2]
    all_vals = []
    for (cx,cy) in points:
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx)**2 + (yy - cy)**2 <= radius**2
        vals = hsv_img[mask]
        if vals.size > 0:
            all_vals.append(vals)
    if not all_vals:
        return None
    all_vals = np.vstack(all_vals)
    h,s,v = all_vals[:,0], all_vals[:,1], all_vals[:,2]
    h_min = max(0, np.min(h)-buffer_h)
    h_max = min(180, np.max(h)+buffer_h)
    s_min = max(0, np.min(s)-buffer_s)
    s_max = min(255, np.max(s)+buffer_s)
    v_min = max(0, np.min(v)-buffer_v)
    v_max = min(255, np.max(v)+buffer_v)
    return (h_min,h_max,s_min,s_max,v_min,v_max)

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler", layout="wide")
st.title("🧬 Zellkern-Zähler – Final Version")

# -------------------- Session State --------------------
default_keys = ["aec_points","hema_points","bg_points","manual_aec","manual_hema",
                "aec_hsv","hema_hsv","bg_hsv","last_file","disp_width","last_auto_run"]
for key in default_keys:
    if key not in st.session_state:
        if "points" in key or "manual" in key:
            st.session_state[key] = []
        elif "last_auto_run" in key:
            st.session_state[key] = 0
        elif "disp_width" in key:
            st.session_state[key] = 1400
        else:
            st.session_state[key] = None

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_points","hema_points","bg_points","manual_aec","manual_hema","aec_hsv","hema_hsv","bg_hsv"]:
        st.session_state[k] = [] if "points" in k or "manual" in k else None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1400

# -------------------- Bild vorbereiten --------------------
colW1, colW2 = st.columns([2,1])
with colW1:
    DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
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

# -------------------- Aktionen Buttons --------------------
st.markdown("### ⚡ Aktionen")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🧹 Alle Punkte löschen"):
        for k in ["aec_points","hema_points","bg_points","manual_aec","manual_hema"]:
            st.session_state[k] = []
        st.success("Alle Punkte gelöscht.")
with col2:
    if st.button("🧾 Kalibrierung zurücksetzen"):
        st.session_state.aec_hsv = None
        st.session_state.hema_hsv = None
        st.session_state.bg_hsv = None
        st.info("Kalibrierung zurückgesetzt.")
with col3:
    if st.button("⚡ Kalibrierung berechnen"):
        st.session_state.aec_hsv = compute_hsv_range_circle(st.session_state.aec_points, hsv_disp, radius=circle_radius)
        st.session_state.hema_hsv = compute_hsv_range_circle(st.session_state.hema_points, hsv_disp, radius=circle_radius)
        st.session_state.bg_hsv = compute_hsv_range_circle(st.session_state.bg_points, hsv_disp, radius=circle_radius)
        st.success("Kalibrierung gespeichert.")
with col4:
    if st.button("🤖 Auto-Erkennung starten"):
        st.session_state.last_auto_run += 1

# -------------------- Bildanzeige + Klicklogik --------------------
marked_disp = image_disp.copy()
for points_list, color in [
    (st.session_state.aec_points,(255,0,0)),
    (st.session_state.hema_points,(0,0,255)),
    (st.session_state.manual_aec,(255,165,0)),
    (st.session_state.manual_hema,(128,0,128)),
    (st.session_state.bg_points,(255,255,0)),
]:
    for (x,y) in points_list:
        cv2.circle(marked_disp,(x,y),circle_radius,color,-1)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key=f"clickable_image_{st.session_state.last_auto_run}", width=DISPLAY_WIDTH)

if coords:
    x, y = int(coords["x"]), int(coords["y"])
    mode = st.radio("Markierungsmodus", ["AEC","Hämatoxylin","Hintergrund","Manuell AEC","Manuell Hämatoxylin","Löschen"])
    if mode=="Löschen":
        for k in ["aec_points","hema_points","manual_aec","manual_hema","bg_points"]:
            st.session_state[k] = [p for p in st.session_state[k] if not is_near(p,(x,y),circle_radius)]
    elif mode=="AEC": st.session_state.aec_points.append((x,y))
    elif mode=="Hämatoxylin": st.session_state.hema_points.append((x,y))
    elif mode=="Hintergrund": st.session_state.bg_points.append((x,y))
    elif mode=="Manuell AEC": st.session_state.manual_aec.append((x,y))
    elif mode=="Manuell Hämatoxylin": st.session_state.manual_hema.append((x,y))

# Dedup aller Punkte
for k in ["aec_points","hema_points","manual_aec","manual_hema","bg_points"]:
    st.session_state[k] = dedup_points(st.session_state[k], min_dist=max(4,circle_radius//2))

# -------------------- Auto-Erkennung --------------------
if st.session_state.last_auto_run > 0:
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
    if blur_kernel > 1: proc = cv2.GaussianBlur(proc,(blur_kernel,blur_kernel),0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    r = clahe.apply(proc[:,:,0])
    g = clahe.apply(proc[:,:,1])
    b = clahe.apply(proc[:,:,2])

    if st.session_state.bg_points:
        bg_vals = np.array([proc[y,x] for (x,y) in st.session_state.bg_points])
        if len(bg_vals) > 0:
            bg_mean = np.mean(bg_vals, axis=0)
            r = np.clip(r - bg_mean[0], 0, 255)
            g = np.clip(g - bg_mean[1], 0, 255)
            b = np.clip(b - bg_mean[2], 0, 255)

    # Masken erstellen
    if st.session_state.aec_hsv:
        vmin,vmax = st.session_state.aec_hsv[4], st.session_state.aec_hsv[5]
        mask_aec = cv2.inRange(r, vmin, vmax)
        st.session_state.aec_points = get_centers(mask_aec, min_area)

    if st.session_state.hema_hsv:
        vmin,vmax = st.session_state.hema_hsv[4], st.session_state.hema_hsv[5]
        mask_hema = cv2.inRange(b, vmin, vmax)
        st.session_state.hema_points = get_centers(mask_hema, min_area)

    st.session_state.last_auto_run = 0

# -------------------- Gesamtzählung --------------------
all_aec = (st.session_state.aec_points or []) + (st.session_state.manual_aec or [])
all_hema = (st.session_state.hema_points or []) + (st.session_state.manual_hema or [])
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
