import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception as e:
    st.error("❌ Benötigtes Paket 'streamlit-image-coordinates' fehlt.\n"
             "Installiere es mit: `pip install streamlit-image-coordinates`")
    st.stop()

# -----------------------------
# INITIALISIERUNG
# -----------------------------
st.set_page_config(page_title="Canvas 2 – Farb- & Punktanalyse", layout="wide")

if "calib_points" not in st.session_state:
    st.session_state["calib_points"] = []
if "manual_points" not in st.session_state:
    st.session_state["manual_points"] = []
if "auto_points" not in st.session_state:
    st.session_state["auto_points"] = []
if "mode" not in st.session_state:
    st.session_state["mode"] = "none"
if "auto_run" not in st.session_state:
    st.session_state["auto_run"] = False

# -----------------------------
# FUNKTIONEN
# -----------------------------

def filter_background(img, brightness_thresh=230):
    """Hintergrund durch Helligkeit filtern"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < brightness_thresh
    img_filtered = img.copy()
    img_filtered[~mask] = 255
    return img_filtered

def dedup_points(points, min_dist=5):
    """Doppelte Punkte entfernen"""
    result = []
    for p in points:
        if not any(((abs(p[0]-q[0]) < min_dist) and (abs(p[1]-q[1]) < min_dist)) for q in result):
            result.append(p)
    return result

def get_centers(mask, min_area=50):
    """Berechne Zentren der Konturen"""
    m = mask.copy()
    res = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        contours, _ = res
    else:
        _, contours, _ = res
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M.get("m00", 0) != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
    return centers

def run_auto_detection(img=None):
    """Simulierter Auto-Detektor"""
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    pts = get_centers(mask, min_area=60)
    return pts

# -----------------------------
# UI SETUP
# -----------------------------
st.title("🎨 Canvas 2 – Interaktive Punkteerkennung")

uploaded = st.file_uploader("Bild hochladen", type=["jpg", "png", "tif", "tiff"])
if not uploaded:
    st.stop()

# Bild einlesen
image = np.array(Image.open(uploaded).convert("RGB"))
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# -----------------------------
# Steuerung / Moduswahl
# -----------------------------
col1, col2, col3, col4 = st.columns([2,2,2,2])
with col1:
    mode = st.radio(
        "Modus wählen:",
        ["Keine", "Kalibrieren", "Punkte hinzufügen", "Punkte löschen", "Auto-Erkennung"],
        index=["none", "calibration", "add", "delete", "auto"].index(st.session_state["mode"])
    )
    st.session_state["mode"] = mode.lower()

with col2:
    if st.button("🧹 Alle Punkte löschen"):
        st.session_state["calib_points"].clear()
        st.session_state["manual_points"].clear()
        st.session_state["auto_points"].clear()
        st.success("Alle Punkte gelöscht.")

with col3:
    brightness_thresh = st.slider("Hintergrund-Helligkeit (Filter)", 100, 255, 230, 5)

with col4:
    if st.button("Auto-Erkennung starten", key="auto_btn"):
        st.session_state["auto_run"] = True

# -----------------------------
# Hintergrund-Filter anwenden
# -----------------------------
filtered_img = filter_background(image_bgr, brightness_thresh)

# -----------------------------
# Koordinaten aufnehmen
# -----------------------------
coord = streamlit_image_coordinates(filtered_img, key="click")

if coord is not None:
    if st.session_state["mode"] == "calibration":
        st.session_state["calib_points"].append((coord["x"], coord["y"]))
    elif st.session_state["mode"] == "add":
        st.session_state["manual_points"].append((coord["x"], coord["y"]))
    elif st.session_state["mode"] == "delete":
        # Nächsten Punkt finden und löschen
        all_pts = st.session_state["manual_points"] + st.session_state["calib_points"]
        if len(all_pts) > 0:
            nearest = min(all_pts, key=lambda p: (p[0]-coord["x"])**2 + (p[1]-coord["y"])**2)
            for arr in ["manual_points", "calib_points"]:
                if nearest in st.session_state[arr]:
                    st.session_state[arr].remove(nearest)
                    break

# -----------------------------
# Auto-Erkennung
# -----------------------------
if st.session_state["auto_run"]:
    st.session_state["auto_points"] = run_auto_detection(filtered_img)
    st.session_state["auto_run"] = False

# -----------------------------
# Punkte zusammenfassen
# -----------------------------
st.session_state["calib_points"] = dedup_points(st.session_state["calib_points"])
st.session_state["manual_points"] = dedup_points(st.session_state["manual_points"])
st.session_state["auto_points"] = dedup_points(st.session_state["auto_points"])

# -----------------------------
# Bild mit Punkten anzeigen
# -----------------------------
display_img = filtered_img.copy()

# Kalibrierpunkte – Blau
for (x, y) in st.session_state["calib_points"]:
    cv2.circle(display_img, (x, y), 6, (255, 0, 0), -1)
# Manuelle Punkte – Grün
for (x, y) in st.session_state["manual_points"]:
    cv2.circle(display_img, (x, y), 6, (0, 255, 0), -1)
# Auto-Erkennung – Rot
for (x, y) in st.session_state["auto_points"]:
    cv2.circle(display_img, (x, y), 6, (0, 0, 255), -1)

st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="Analyseansicht", use_column_width=True)

# -----------------------------
# Ergebnisse
# -----------------------------
n_calib = len(st.session_state["calib_points"])
n_manual = len(st.session_state["manual_points"])
n_auto = len(st.session_state["auto_points"])

st.write(f"📍 **Kalibrierpunkte:** {n_calib} | 🟢 **Manuelle Punkte:** {n_manual} | 🔴 **Auto-Erkannt:** {n_auto}")

# -----------------------------
# Export als CSV
# -----------------------------
data = []
for (x, y) in st.session_state["calib_points"]:
    data.append(("Kalibrierung", x, y))
for (x, y) in st.session_state["manual_points"]:
    data.append(("Manuell", x, y))
for (x, y) in st.session_state["auto_points"]:
    data.append(("Auto", x, y))

if len(data) > 0:
    df = pd.DataFrame(data, columns=["Typ", "X", "Y"])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📄 Ergebnisse als CSV herunterladen", csv, "punkte.csv", "text/csv")

# -----------------------------
# Debug / Kontrolle
# -----------------------------
with st.expander("🧠 Debug Info"):
    st.write(st.session_state)
