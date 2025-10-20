import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("Hintergrund-Differenz Visualisierung (Graustufen)")

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
if len(img_8bit.shape) == 3 and img_8bit.shape[2] == 3:
    gray = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2GRAY)
else:
    gray = img_8bit.copy()

H, W = gray.shape

# --- Session für Hintergrundpunkte ---
if "bg_points" not in st.session_state:
    st.session_state.bg_points = []

# --- Originalbild mit Punkten ---
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
for (x, y) in st.session_state.bg_points:
    cv2.circle(vis, (x, y), 5, (255, 255, 0), 2)

coords = streamlit_image_coordinates(Image.fromarray(vis), key="bg_click_points", width=800)
if coords:
    x = int(np.clip(coords["x"], 0, W-1))
    y = int(np.clip(coords["y"], 0, H-1))
    st.session_state.bg_points.append((x, y))

st.image(vis, caption="Originalbild mit Hintergrundpunkten (gelb)", use_column_width=True)

# --- Differenzbild berechnen ---
if st.button("Differenzbild erzeugen") and st.session_state.bg_points:
    bg_vals = np.array([gray[y,x] for (x,y) in st.session_state.bg_points])
    bg_mean = int(np.mean(bg_vals))

    # Subtraktion, alles <0 auf 0
    diff = gray.astype(np.int16) - bg_mean
    diff[diff < 0] = 0
    diff = diff.astype(np.uint8)

    # Optional: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    diff_clahe = clahe.apply(diff)

    # --- Nebeneinander darstellen ---
    combined = np.hstack([
        cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(diff_clahe, cv2.COLOR_GRAY2RGB)
    ])
    st.image(combined, caption="Links: Original | Rechts: Differenzbild", use_column_width=True)

    # Für weitere Schritte speichern
    st.session_state.bg_sub_image = diff_clahe
