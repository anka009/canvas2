import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd

st.title("🔹 Auto-Erkennung mit Hintergrundsubtraktion")

# --- Upload ---
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.stop()

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
DISPLAY_WIDTH = 800
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig*scale)), interpolation=cv2.INTER_AREA)
hsv_orig = cv2.cvtColor(image_orig, cv2.COLOR_RGB2HSV)

# --- Session State ---
for key in ["aec_points","hema_points","manual_aec","manual_hema","bg_points","aec_hsv","hema_hsv","run_auto"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "points" in key or "manual" in key else None

# --- Hintergrundpunkte setzen ---
marked_disp = image_disp.copy()
for (x,y) in st.session_state.bg_points:
    cv2.circle(marked_disp, (int(x*scale), int(y*scale)), 5, (255,255,0), 2)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp), width=DISPLAY_WIDTH)
if coords:
    x_disp, y_disp = coords["x"], coords["y"]
    x_orig, y_orig = int(x_disp/scale), int(y_disp/scale)
    st.session_state.bg_points.append((x_orig,y_orig))

st.image(marked_disp, caption="Hintergrundpunkte setzen", use_column_width=True)

# --- Hintergrundsubtraktion ---
if st.session_state.bg_points:
    bg_vals = np.array([hsv_orig[y,x] for (x,y) in st.session_state.bg_points])
    bg_mean = bg_vals.mean(axis=0)
    hsv_proc = hsv_orig.copy()
    h,s,v = cv2.split(hsv_proc)
    s = np.clip(s - bg_mean[1],0,255).astype(np.uint8)
    v = np.clip(v - bg_mean[2],0,255).astype(np.uint8)
    hsv_proc = cv2.merge([h,s,v])
    rgb_proc = cv2.cvtColor(hsv_proc, cv2.COLOR_HSV2RGB)
    st.image(rgb_proc, caption="Hintergrund subtrahiert", use_column_width=True)

# --- Auto-Erkennung ---
def compute_hsv_stats(points, hsv_img, buffer=10):
    if not points:
        return None
    vals = np.array([hsv_img[y,x] for (x,y) in points])
    h_mean,s_mean,v_mean = vals.mean(axis=0)
    h_std,s_std,v_std = vals.std(axis=0)+buffer
    h_min,h_max = max(0,int(h_mean-h_std)), min(180,int(h_mean+h_std))
    s_min,s_max = max(0,int(s_mean-s_std)), min(255,int(s_mean+s_std))
    v_min,v_max = max(0,int(v_mean-v_std)), min(255,int(v_mean+v_std))
    return (h_min,h_max,s_min,s_max,v_min,v_max)

def apply_hue_wrap(hsv_img,hmin,hmax,smin,smax,vmin,vmax):
    if hmin <= hmax:
        mask = cv2.inRange(hsv_img, np.array([hmin,smin,vmin]), np.array([hmax,smax,vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img, np.array([0,smin,vmin]), np.array([hmax,smax,vmin]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin,smin,vmin]), np.array([180,smax,vmax]))
        mask = cv2.bitwise_or(mask_lo,mask_hi)
    return mask

def get_centers(mask,min_area=50):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers=[]
    for c in contours:
        if cv2.contourArea(c)>=min_area:
            M=cv2.moments(c)
            if M.get("m00",0)!=0:
                cx=int(M["m10"]/M["m00"])
                cy=int(M["m01"]/M["m00"])
                centers.append((cx,cy))
    return centers

st.markdown("### Auto-Erkennung")
col1,col2=st.columns(2)
with col1:
    if st.button("Kalibrierung AEC berechnen"):
        st.session_state.aec_hsv = compute_hsv_stats(st.session_state.aec_points,hsv_proc)
with col2:
    if st.button("Kalibrierung Hämatoxylin berechnen"):
        st.session_state.hema_hsv = compute_hsv_stats(st.session_state.hema_points,hsv_proc)

if st.button("Auto-Erkennung starten"):
    auto_points = {"AEC":[],"Hämatoxylin":[]}
    if st.session_state.aec_hsv:
        hmin,hmax,smin,smax,vmin,vmax=st.session_state.aec_hsv
        mask = apply_hue_wrap(hsv_proc,hmin,hmax,smin,smax,vmin,vmax)
        auto_points["AEC"] = get_centers(mask)
    if st.session_state.hema_hsv:
        hmin,hmax,smin,smax,vmin,vmax=st.session_state.hema_hsv
        mask = apply_hue_wrap(hsv_proc,hmin,hmax,smin,smax,vmin,vmax)
        auto_points["Hämatoxylin"] = get_centers(mask)
    
    # Anzeige
    disp_show = image_disp.copy()
    for pts,color in [(auto_points["AEC"],(255,0,0)),(auto_points["Hämatoxylin"],(0,0,255))]:
        for x,y in pts:
            cv2.circle(disp_show,(int(x*scale),int(y*scale)),5,color,2)
    st.image(disp_show, caption="Auto-Erkennung Ergebnisse", use_column_width=True)
