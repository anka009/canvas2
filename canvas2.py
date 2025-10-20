import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("🔹 Hintergrundsubtraktion Demo")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.stop()

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
DISPLAY_WIDTH = 800
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig*scale)), interpolation=cv2.INTER_AREA)
hsv_orig = cv2.cvtColor(image_orig, cv2.COLOR_RGB2HSV)

# Session State für Hintergrundpunkte
if "bg_points" not in st.session_state:
    st.session_state.bg_points = []

# Bildanzeige mit Punkten
marked_disp = image_disp.copy()
for (x,y) in st.session_state.bg_points:
    cv2.circle(marked_disp, (int(x*scale), int(y*scale)), 5, (255,255,0), 2)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp), width=DISPLAY_WIDTH)
if coords:
    x_disp, y_disp = coords["x"], coords["y"]
    x_orig, y_orig = int(x_disp/scale), int(y_disp/scale)
    st.session_state.bg_points.append((x_orig,y_orig))

st.image(marked_disp, caption="Hintergrundpunkte setzen", use_column_width=True)

# -------------------- Hintergrundsubtraktion --------------------
if st.session_state.bg_points:
    bg_vals = np.array([hsv_orig[y,x] for (x,y) in st.session_state.bg_points])
    bg_mean = bg_vals.mean(axis=0)
    hsv_bg_sub = hsv_orig.copy()
    hsv_bg_sub[:,:,0] = np.clip(hsv_bg_sub[:,:,0]-bg_mean[0], 0, 180)
    hsv_bg_sub[:,:,1] = np.clip(hsv_bg_sub[:,:,1]-bg_mean[1], 0, 255)
    hsv_bg_sub[:,:,2] = np.clip(hsv_bg_sub[:,:,2]-bg_mean[2], 0, 255)
    
    rgb_bg_sub = cv2.cvtColor(hsv_bg_sub, cv2.COLOR_HSV2RGB)
    st.image(rgb_bg_sub, caption="Hintergrund subtrahiert", use_column_width=True)
