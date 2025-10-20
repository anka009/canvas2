import cv2
import numpy as np
from PIL import Image
import streamlit as st

st.title("Hintergrund-Subtraktion mit 8-bit-Konvertierung")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "tif", "tiff"])

if uploaded_file:
    img = np.array(Image.open(uploaded_file))

    # 🔹 1️⃣ In 8-Bit konvertieren, egal ob 12-, 16- oder float-Bild
    if img.dtype == np.uint16:
        img_8bit = cv2.convertScaleAbs(img, alpha=255.0/65535.0)
    elif img.dtype in [np.float32, np.float64]:
        img_8bit = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img_8bit = img.astype(np.uint8)

    # 🔹 2️⃣ Sicherstellen, dass 3 Kanäle vorhanden sind
    if len(img_8bit.shape) == 2:
        img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2HSV)

    # 🔹 Session-State vorbereiten
    if "bg_points" not in st.session_state:
        st.session_state.bg_points = []

    st.write("Klicke ins Bild, um Hintergrundpunkte zu setzen (Kalibrierung).")

    # 🎨 Klickbare Bildfläche (du kannst hier st_image or canvas verwenden)
    import streamlit_image_coordinates

    result = streamlit_image_coordinates.streamlit_image_coordinates(img_8bit, key="bg_coords")

    # Wenn auf das Bild geklickt wurde
    if result is not None and "x" in result and "y" in result:
        st.session_state.bg_points.append((int(result["x"]), int(result["y"])))

    # 🔹 3️⃣ Hintergrund berechnen und subtrahieren
    if st.button("Hintergrund subtrahieren") and st.session_state.bg_points:
        bg_vals = np.array([hsv[y, x] for (x, y) in st.session_state.bg_points])
        bg_mean = bg_vals.mean(axis=0).astype(np.uint8)

        hsv_sub = cv2.subtract(hsv, bg_mean)
        rgb_sub = cv2.cvtColor(hsv_sub, cv2.COLOR_HSV2RGB)

        st.image(
            np.hstack([img_8bit, rgb_sub]),
            caption="Links: Original | Rechts: Hintergrund subtrahiert",
            channels="RGB",
            use_container_width=True
        )

        st.write(f"Hintergrundmittelwert (HSV): {bg_mean}")
    else:
        st.image(img_8bit, caption="Originalbild", use_container_width=True)
