import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("Hintergrundsubtraktion (8-bit sicher)")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","png","tif","tiff"])
if not uploaded_file:
    st.stop()

# --- Bild laden ---
img = np.array(Image.open(uploaded_file))

# --- 8-bit Konvertierung ---
if img.dtype == np.uint16:
    img_8bit = cv2.convertScaleAbs(img, alpha=255.0/65535.0)
elif img.dtype in [np.float32, np.float64]:
    img_8bit = np.clip(img*255,0,255).astype(np.uint8)
else:
    img_8bit = img.astype(np.uint8)

# --- sicherstellen, dass 3 Kanäle ---
if len(img_8bit.shape) == 2:
    img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
elif img_8bit.shape[2] == 4:
    img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_RGBA2RGB)

# --- RGB für Streamlit (BGR → RGB) ---
img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2RGB)

H, W = img_rgb.shape[:2]
scale = 800 / W
img_disp = cv2.resize(img_rgb, (int(W*scale), int(H*scale)))

# --- Session für Hintergrundpunkte ---
if "bg_points" not in st.session_state:
    st.session_state.bg_points = []

# --- klickbare Punkte ---
vis = img_disp.copy()
for (x, y) in st.session_state.bg_points:
    cv2.circle(vis, (int(x*scale), int(y*scale)), 5, (255,255,0), 2)

coords = streamlit_image_coordinates(Image.fromarray(vis), key="bg_click", width=int(W*scale))
if coords:
    x_orig = int(round(coords["x"]/scale))
    y_orig = int(round(coords["y"]/scale))
    x_orig = np.clip(x_orig,0,W-1)
    y_orig = np.clip(y_orig,0,H-1)
    st.session_state.bg_points.append((x_orig,y_orig))

st.write("Hintergrundpunkte (Original-Koordinaten):", st.session_state.bg_points)
st.image(vis, caption="BG-Punkte (gelb) auf Bild", use_column_width=True)

# --- Hintergrund abziehen ---
if st.button("Hintergrund subtrahieren") and st.session_state.bg_points:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    bg_vals = np.array([hsv[y,x] for (x,y) in st.session_state.bg_points])
    bg_mean = bg_vals.mean(axis=0).astype(np.uint8)

    hsv_sub = cv2.subtract(hsv, bg_mean)
    rgb_sub = cv2.cvtColor(hsv_sub, cv2.COLOR_HSV2RGB)

    st.image(
        np.hstack([img_rgb, rgb_sub]),
        caption="Links: Original | Rechts: Hintergrund subtrahiert",
        use_column_width=True
    )
