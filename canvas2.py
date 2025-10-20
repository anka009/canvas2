import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("Graustufen Hintergrund-Subtraktion")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","png","tif","tiff"])
if not uploaded_file:
    st.stop()

# --- Bild laden & 8-bit ---
img = np.array(Image.open(uploaded_file))

if img.dtype == np.uint16:
    img_8bit = cv2.convertScaleAbs(img, alpha=255.0/65535.0)
elif img.dtype in [np.float32, np.float64]:
    img_8bit = np.clip(img*255,0,255).astype(np.uint8)
else:
    img_8bit = img.astype(np.uint8)

# --- Graustufen ---
if len(img_8bit.shape)==3 and img_8bit.shape[2]==3:
    gray = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2GRAY)
else:
    gray = img_8bit.copy()

H, W = gray.shape

# --- Session für Hintergrundpunkte ---
if "bg_points" not in st.session_state:
    st.session_state.bg_points = []

vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
for (x, y) in st.session_state.bg_points:
    cv2.circle(vis, (x, y), 5, (255,255,0), 2)

coords = streamlit_image_coordinates(Image.fromarray(vis), key="bg_click_gray", width=800)
if coords:
    x = int(np.clip(coords["x"], 0, W-1))
    y = int(np.clip(coords["y"], 0, H-1))
    st.session_state.bg_points.append((x, y))

st.image(vis, caption="Hintergrundpunkte (gelb)", use_column_width=True)

# --- Hintergrund abziehen ---
if st.button("Hintergrund subtrahieren") and st.session_state.bg_points:
    bg_vals = np.array([gray[y,x] for (x,y) in st.session_state.bg_points])
    bg_mean = int(np.mean(bg_vals))

    gray_sub = cv2.subtract(gray, bg_mean)  # automatisch clipped 0–255

    # Optional: CLAHE für besseren Kontrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_sub = clahe.apply(gray_sub)

    st.image(gray_sub, caption="Bild ohne Hintergrund (Graustufen)", use_column_width=True)
    st.session_state.bg_sub_image = gray_sub
