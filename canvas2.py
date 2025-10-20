# canvas_robust.py
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
            if M.get("m00",0) != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                centers.append((cx,cy))
    return centers

def compute_hsv_stats(points, hsv_img, buffer=10):
    if not points:
        return None
    vals = np.array([hsv_img[y,x] for (x,y) in points])
    h_mean = int(np.mean(vals[:,0]))
    s_mean = int(np.mean(vals[:,1]))
    v_mean = int(np.mean(vals[:,2]))
    h_std = int(np.std(vals[:,0])) + buffer
    s_std = int(np.std(vals[:,1])) + buffer
    v_std = int(np.std(vals[:,2])) + buffer
    h_min, h_max = max(0,h_mean-h_std), min(180,h_mean+h_std)
    s_min, s_max = max(0,s_mean-s_std), min(255,s_mean+s_std)
    v_min, v_max = max(0,v_mean-v_std), min(255,v_mean+v_std)
    return (h_min,h_max,s_min,s_max,v_min,v_max)

def apply_hue_wrap(hsv_img, hmin,hmax,smin,smax,vmin,vmax):
    if hmin <= hmax:
        mask = cv2.inRange(hsv_img, np.array([hmin,smin,vmin]), np.array([hmax,smax,vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img, np.array([0,smin,vmin]), np.array([hmax,smax,vmax]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin,smin,vmin]), np.array([180,smax,vmax]))
        mask = cv2.bitwise_or(mask_lo,mask_hi)
    return mask

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler Robust", layout="wide")
st.title("🧬 Zellkern-Zähler – Robust & Stabil")

# -------------------- Session State --------------------
keys = ["aec_points","hema_points","manual_aec","manual_hema",
        "aec_hsv","hema_hsv","bg_points","last_file","run_auto"]
for key in keys:
    if key not in st.session_state:
        st.session_state[key] = [] if "points" in key or "manual" in key else None

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_points","hema_points","manual_aec","manual_hema","aec_hsv","hema_hsv","bg_points"]:
        st.session_state[k] = [] if "points" in k or "manual" in k else None
    st.session_state.last_file = uploaded_file.name

# -------------------- Bild vorbereiten --------------------
DISPLAY_WIDTH = st.slider("Bildbreite", 400, 2000, 800)
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig*scale)), interpolation=cv2.INTER_AREA)
hsv_orig = cv2.cvtColor(image_orig, cv2.COLOR_RGB2HSV)

# -------------------- Sidebar: Modus & Aktionen --------------------
with st.sidebar:
    st.markdown("### Modus")
    mode = st.radio("Punkt setzen:", ["AEC Kalibrierung","Hämatoxylin Kalibrierung",
                                     "Manuell AEC","Manuell Hämatoxylin","Hintergrund","Löschen"])
    st.markdown("### Aktionen")
    if st.button("Alle Punkte löschen"):
        for k in ["aec_points","hema_points","manual_aec","manual_hema","bg_points"]:
            st.session_state[k] = []
    if st.button("Kalibrierung berechnen"):
        st.session_state.aec_hsv = compute_hsv_stats(st.session_state.aec_points, hsv_orig)
        st.session_state.hema_hsv = compute_hsv_stats(st.session_state.hema_points, hsv_orig)
        st.success("Kalibrierung gespeichert.")
    if st.button("Auto-Erkennung"):
        st.session_state.run_auto = True

# -------------------- Klicklogik (Originalkoordinaten) --------------------
marked_disp = image_disp.copy()
def scale_points_to_disp(points):
    return [(int(x*scale), int(y*scale)) for (x,y) in points]

for pts, color in [
    (st.session_state.aec_points,(255,0,0)),
    (st.session_state.hema_points,(0,0,255)),
    (st.session_state.manual_aec,(255,165,0)),
    (st.session_state.manual_hema,(128,0,128)),
    (st.session_state.bg_points,(255,255,0))
]:
    for (x_orig, y_orig) in pts:
        x_disp, y_disp = int(x_orig*scale), int(y_orig*scale)
        cv2.circle(marked_disp,(x_disp, y_disp),5,color,2)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp), width=DISPLAY_WIDTH)
if coords:
    x_disp, y_disp = int(coords["x"]), int(coords["y"])
    x_orig, y_orig = int(x_disp/scale), int(y_disp/scale)
    if mode=="Löschen":
        for k in ["aec_points","hema_points","manual_aec","manual_hema","bg_points"]:
            st.session_state[k] = [p for p in st.session_state[k] if not is_near(p,(x_orig,y_orig),5)]
    elif mode=="AEC Kalibrierung":
        st.session_state.aec_points.append((x_orig,y_orig))
    elif mode=="Hämatoxylin Kalibrierung":
        st.session_state.hema_points.append((x_orig,y_orig))
    elif mode=="Manuell AEC":
        st.session_state.manual_aec.append((x_orig,y_orig))
    elif mode=="Manuell Hämatoxylin":
        st.session_state.manual_hema.append((x_orig,y_orig))
    elif mode=="Hintergrund":
        st.session_state.bg_points.append((x_orig,y_orig))

# -------------------- Auto-Erkennung --------------------
if st.session_state.get('run_auto'):
    st.session_state.run_auto = False
    proc = image_orig.copy()
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

    # Hintergrundmittel berechnen
    bg_mean = [0,0,0]
    if st.session_state.bg_points:
        bg_vals = np.array([hsv_proc[y,x] for (x,y) in st.session_state.bg_points])
        bg_mean = bg_vals.mean(axis=0)
        hsv_proc[:,:,0] = np.clip(hsv_proc[:,:,0]-bg_mean[0],0,180)
        hsv_proc[:,:,1] = np.clip(hsv_proc[:,:,1]-bg_mean[1],0,255)
        hsv_proc[:,:,2] = np.clip(hsv_proc[:,:,2]-bg_mean[2],0,255)

    # AEC
    if st.session_state.aec_hsv:
        hmin,hmax,smin,smax,vmin,vmax = st.session_state.aec_hsv
        mask = apply_hue_wrap(hsv_proc,hmin,hmax,smin,smax,vmin,vmax)
        st.session_state.aec_points = get_centers(mask,min_area=50)

    # Hämatoxylin
    if st.session_state.hema_hsv:
        hmin,hmax,smin,smax,vmin,vmax = st.session_state.hema_hsv
        mask = apply_hue_wrap(hsv_proc,hmin,hmax,smin,smax,vmin,vmax)
        st.session_state.hema_points = get_centers(mask,min_area=50)

# -------------------- Ergebnisse + CSV --------------------
all_aec = st.session_state.aec_points + st.session_state.manual_aec
all_hema = st.session_state.hema_points + st.session_state.manual_hema
st.markdown(f"### Gesamt: AEC={len(all_aec)}, Hämatoxylin={len(all_hema)}")

df_list = [{"X_display":int(x*scale),"Y_display":int(y*scale),"Type":"AEC"} for (x,y) in all_aec]
df_list += [{"X_display":int(x*scale),"Y_display":int(y*scale),"Type":"Hämatoxylin"} for (x,y) in all_hema]
if df_list:
    df = pd.DataFrame(df_list)
    df["X_original"] = [x for x,_ in all_aec+all_hema]
    df["Y_original"] = [y for _,y in all_aec+all_hema]
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("CSV exportieren", data=csv, file_name="zellkerne.csv")
