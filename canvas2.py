import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide")
st.title("🎨 Hintergrund-Test (stabile Basis)")

# --- Upload ---
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.stop()

# --- Lade Originalbild ---
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H, W = image_orig.shape[:2]
disp_w = 800
scale = disp_w / W
image_disp = cv2.resize(image_orig, (disp_w, int(H*scale)))

hsv = cv2.cvtColor(image_orig, cv2.COLOR_RGB2HSV)

# --- Session State ---
if "bg_points" not in st.session_state:
    st.session_state.bg_points = []

# --- Punkte setzen ---
temp_disp = image_disp.copy()
for (x, y) in st.session_state.bg_points:
    cv2.circle(temp_disp, (int(x*scale), int(y*scale)), 6, (255, 255, 0), 2)

coords = streamlit_image_coordinates(Image.fromarray(temp_disp), width=disp_w, key="bg_click")
if coords:
    x_orig = int(coords["x"] / scale)
    y_orig = int(coords["y"] / scale)
    st.session_state.bg_points.append((x_orig, y_orig))

st.image(temp_disp, caption="Klicke auf Hintergrundbereiche")

# --- Buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Alle Punkte löschen"):
        st.session_state.bg_points = []
with col2:
    apply_sub = st.button("🧮 Hintergrund subtrahieren")

# --- Subtraktion ---
if apply_sub and st.session_state.bg_points:
    bg_vals = np.array([hsv[y, x] for (x, y) in st.session_state.bg_points])
    bg_mean = bg_vals.mean(axis=0)

    h, s, v = cv2.split(hsv)
    s = np.clip(s - bg_mean[1], 0, 255)
    v = np.clip(v - bg_mean[2], 0, 255)
    hsv_sub = cv2.merge([h, s, v]).astype(np.uint8)
    rgb_sub = cv2.cvtColor(hsv_sub, cv2.COLOR_HSV2RGB)

    before = cv2.resize(image_orig, (disp_w, int(H*scale)))
    after = cv2.resize(rgb_sub, (disp_w, int(H*scale)))
    st.image(np.hstack([before, after]), caption="Links Original / Rechts Hintergrund entfernt")
