import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("Bild ohne Hintergrund für Kern-Erkennung")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","png","tif","tiff"])
if not uploaded_file:
    st.stop()

# --- Bild laden & 8-bit konvertieren ---
img = np.array(Image.open(uploaded_file))
if img.dtype == np.uint16:
    img_8bit = cv2.convertScaleAbs(img, alpha=255.0/65535.0)
elif img.dtype in [np.float32, np.float64]:
    img_8bit = np.clip(img*255,0,255).astype(np.uint8)
else:
    img_8bit = img.astype(np.uint8)

if len(img_8bit.shape)==2:
    img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
elif img_8bit.shape[2]==4:
    img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_RGBA2RGB)

img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]

# --- Hintergrundpunkte setzen ---
if "bg_points" not in st.session_state:
    st.session_state.bg_points = []

vis = img_rgb.copy()
for (x, y) in st.session_state.bg_points:
    cv2.circle(vis, (x, y), 5, (255,255,0), 2)

coords = streamlit_image_coordinates(Image.fromarray(vis), key="bg_click", width=800)
if coords:
    x = int(np.clip(coords["x"], 0, W-1))
    y = int(np.clip(coords["y"], 0, H-1))
    st.session_state.bg_points.append((x, y))

st.image(vis, caption="Hintergrundpunkte (gelb)", use_column_width=True)

# --- Hintergrund abziehen, nur Value-Kanal ---
if st.button("Hintergrund subtrahieren") and st.session_state.bg_points:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    bg_vals = np.array([hsv[y,x] for (x,y) in st.session_state.bg_points])
    bg_mean = bg_vals.mean(axis=0).astype(np.uint8)

    h, s, v = cv2.split(hsv)
    v = cv2.subtract(v, bg_mean[2])  # nur Helligkeit abziehen
    v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)  # optional strecken
    # CLAHE für besseren Kontrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)

    hsv_sub = cv2.merge([h,s,v])
    rgb_sub = cv2.cvtColor(hsv_sub, cv2.COLOR_HSV2RGB)

    st.image(rgb_sub, caption="Bild ohne Hintergrund", use_column_width=True)

    # Optional: zurück in session_state speichern für Auto-Erkennung
    st.session_state.bg_sub_image = rgb_sub
