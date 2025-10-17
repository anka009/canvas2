# canvas-nervt_final.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def get_centers(mask, min_area=50):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M.get("m00", 0) != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
    return centers

def compute_hsv_range(points, hsv_img, buffer_h=8, buffer_s=30, buffer_v=25):
    """Berechnet min/max HSV aus Klickpunkten (display-Koordinaten)."""
    if not points:
        return None
    vals = np.array([hsv_img[y, x] for (x, y) in points])
    h = vals[:,0].astype(int)
    s = vals[:,1].astype(int)
    v = vals[:,2].astype(int)

    h_min = int(max(0, np.min(h) - buffer_h))
    h_max = int(min(180, np.max(h) + buffer_h))
    s_min = int(max(0, np.min(s) - buffer_s))
    s_max = int(min(255, np.max(s) + buffer_s))
    v_min = int(max(0, np.min(v) - buffer_v))
    v_max = int(min(255, np.max(v) + buffer_v))

    # Wenn Hue über die 0/180-Grenze springt, es wird später beim Anwenden berücksichtigt
    return (h_min, h_max, s_min, s_max, v_min, v_max)

def apply_hue_wrap(hsv_img, hmin, hmax, smin, smax, vmin, vmax):
    """Erzeugt Maske mit Hue-Wrap (wenn hmin>hmax interpretiere als wrap)."""
    if hmin <= hmax:
        mask = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax]))
    else:
        # Wrap: [0..hmax] OR [hmin..180]
        mask_lo = cv2.inRange(hsv_img, np.array([0, smin, vmin]), np.array([hmax, smax, vmax]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([180, smax, vmax]))
        mask = cv2.bitwise_or(mask_lo, mask_hi)
    return mask

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Interaktiver Zellkern-Zähler (Kalibrierung)", layout="wide")
st.title("🧬 Interaktiver Zellkern-Zähler — Klick-Kalibrierung & automatische Erkennung")

# -------------------- Session state defaults --------------------
defaults = {
    "aec_points": [], "hema_points": [], "bg_points": [], "manual_points": [],
    "aec_hsv": None, "hema_hsv": None, "bg_hsv": None,
    "last_file": None, "disp_width": 1400,
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])

if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

# Reset session state when new file
if uploaded_file.name != st.session_state.last_file:
    st.session_state.aec_points = []
    st.session_state.hema_points = []
    st.session_state.bg_points = []
    st.session_state.manual_points = []
    st.session_state.aec_hsv = None
    st.session_state.hema_hsv = None
    st.session_state.bg_hsv = None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1400

# -------------------- Bildgröße / scale --------------------
colW1, colW2 = st.columns([2,1])
with colW1:
    DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100, key="disp_width_slider")
    st.session_state.disp_width = DISPLAY_WIDTH
with colW2:
    st.write("Bildanzeige skalieren")

# -------------------- Bildvorbereitung --------------------
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig * scale)), interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

# -------------------- Parameter-Slider --------------------
st.markdown("### ⚙️ Filter- und Erkennungsparameter")
col1, col2, col3 = st.columns(3)
with col1:
    blur_kernel = st.slider("🔧 Blur (ungerade)", 1, 21, 5, step=2, key="blur_slider")
    min_area = st.number_input("📏 Mindestfläche (px)", 10, 2000, 100, key="min_area_input")
with col2:
    alpha = st.slider("🌗 Alpha (Kontrast)", 0.1, 3.0, 1.0, step=0.1, key="alpha_slider")
with col3:
    circle_radius = st.slider("⚪ Kreisradius (Anzeige)", 3, 20, 8, key="circle_slider")
    line_thickness = st.slider("📏 Linienstärke (Anzeige)", 1, 5, 2, key="thickness_slider")

st.markdown("### 🎨 Kalibrierungs-Modi (klicke einige Punkte)")
colA, colB, colC = st.columns(3)
with colA:
    aec_mode = st.checkbox("🔴 AEC markieren (rot/braun)", key="aec_mode_cb")
with colB:
    hema_mode = st.checkbox("🔵 Hämatoxylin markieren (blau/lila)", key="hema_mode_cb")
with colC:
    bg_mode = st.checkbox("🖌 Hintergrund markieren", key="bg_mode_cb")

delete_mode = st.checkbox("🗑️ Löschmodus (Punkte entfernen)", key="delete_mode_cb")

# -------------------- Anzeige: markierte Punkte --------------------
marked_disp = image_disp.copy()
# draw existing points
for (x,y) in st.session_state.aec_points:
    cv2.circle(marked_disp, (x,y), circle_radius, (255,0,0), line_thickness)
for (x,y) in st.session_state.hema_points:
    cv2.circle(marked_disp, (x,y), circle_radius, (0,0,255), line_thickness)
for (x,y) in st.session_state.bg_points:
    cv2.circle(marked_disp, (x,y), circle_radius, (255,255,0), line_thickness)
for (x,y) in st.session_state.manual_points:
    cv2.circle(marked_disp, (x,y), circle_radius, (0,255,0), line_thickness)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key="clickable_image", width=DISPLAY_WIDTH)

# -------------------- Klick-Logik --------------------
if coords is not None:
    x, y = coords["x"], coords["y"]
    if delete_mode:
        # Entferne Punkt in Nähe aus allen Lists
        for list_name in ["aec_points","hema_points","bg_points","manual_points"]:
            st.session_state[list_name] = [p for p in st.session_state[list_name] if not is_near(p,(x,y), r=circle_radius)]
    elif aec_mode:
        st.session_state.aec_points.append((x,y))
    elif hema_mode:
        st.session_state.hema_points.append((x,y))
    elif bg_mode:
        st.session_state.bg_points.append((x,y))
    else:
        st.session_state.manual_points.append((x,y))

# -------------------- Kalibrierung per Button --------------------
col_cal1, col_cal2 = st.columns([1,1])
with col_cal1:
    if st.button("⚡ Kalibrierung aus markierten Punkten"):
        # berechne HSV-Bereiche für jede Kategorie
        st.session_state.aec_hsv = compute_hsv_range(st.session_state.aec_points, hsv_disp, buffer_h=8, buffer_s=30, buffer_v=30)
        st.session_state.hema_hsv = compute_hsv_range(st.session_state.hema_points, hsv_disp, buffer_h=8, buffer_s=25, buffer_v=25)
        st.session_state.bg_hsv  = compute_hsv_range(st.session_state.bg_points, hsv_disp, buffer_h=6, buffer_s=20, buffer_v=30)
        st.success("Kalibrierung berechnet (AEC / Hämatoxylin / Hintergrund).")
with col_cal2:
    if st.button("🧹 Hintergrund-Punkte löschen"):
        st.session_state.bg_points = []
        st.info("Hintergrundpunkte gelöscht.")

# Show computed ranges (if present)
if st.session_state.aec_hsv:
    hmin,hmax,smin,smax,vmin,vmax = st.session_state.aec_hsv
    st.write(f"AEC HSV Bereich (geschätzt): H {hmin}-{hmax}, S {smin}-{smax}, V {vmin}-{vmax}")
if st.session_state.hema_hsv:
    hmin,hmax,smin,smax,vmin,vmax = st.session_state.hema_hsv
    st.write(f"Hämatoxylin HSV Bereich (geschätzt): H {hmin}-{hmax}, S {smin}-{smax}, V {vmin}-{vmax}")
if st.session_state.bg_hsv:
    hmin,hmax,smin,smax,vmin,vmax = st.session_state.bg_hsv
    st.write(f"Hintergrund HSV Bereich (geschätzt): H {hmin}-{hmax}, S {smin}-{smax}, V {vmin}-{vmax}")

# -------------------- Automatische Erkennung (nutzt die Kalibrierung) --------------------
if st.button("🤖 Auto-Erkennung (nutzt Kalibrierung)"):
    # Vorverarbeitung (Kontrast / Blur)
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc, (blur_kernel, blur_kernel), 0)
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

    # Erzeuge foreground-mask basierend auf Hintergrundkalibrierung (falls vorhanden)
    if st.session_state.bg_hsv:
        bhmin,bhmax,bsmin,bsmax,bvmin,bvmax = st.session_state.bg_hsv
        bg_mask = apply_hue_wrap(hsv_proc, bhmin, bhmax, bsmin, bsmax, bvmin, bvmax)
        fg_mask = cv2.bitwise_not(bg_mask)
    else:
        fg_mask = np.ones(hsv_proc.shape[:2], dtype=np.uint8) * 255

    kernel = np.ones((3,3), np.uint8)

    # --- AEC Erkennung (Hue-Wrap berücksichtigt) ---
    found_aec = []
    if st.session_state.aec_hsv:
        ahmin,ahmax,asmin,asmax,avmin,avmax = st.session_state.aec_hsv
        # wichtig: wenn Hue-Bereich tatsächlich wrapt (z.B. hmin>hmax), apply_hue_wrap behandelt das
        mask_aec = apply_hue_wrap(hsv_proc, ahmin, ahmax, asmin, asmax, avmin, avmax)
        # Anwenden der foreground-mask (Hintergrund ausschließen)
        mask_aec = cv2.bitwise_and(mask_aec, mask_aec, mask=fg_mask)
        mask_aec = cv2.morphologyEx(mask_aec, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_aec = cv2.morphologyEx(mask_aec, cv2.MORPH_CLOSE, kernel, iterations=1)
        found_aec = get_centers(mask_aec, min_area=int(min_area))
    st.session_state.aec_points = found_aec

    # --- Hämatoxylin Erkennung ---
    found_hema = []
    if st.session_state.hema_hsv:
        hhmin,hhmax,hsmin,hsmax,hvmin,hvmax = st.session_state.hema_hsv
        mask_hema = apply_hue_wrap(hsv_proc, hhmin, hhmax, hsmin, hsmax, hvmin, hvmax)
        mask_hema = cv2.bitwise_and(mask_hema, mask_hema, mask=fg_mask)
        mask_hema = cv2.morphologyEx(mask_hema, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_hema = cv2.morphologyEx(mask_hema, cv2.MORPH_CLOSE, kernel, iterations=1)
        found_hema = get_centers(mask_hema, min_area=int(min_area))
    st.session_state.hema_points = found_hema

    st.success(f"✅ {len(found_aec)} AEC, {len(found_hema)} Hämatoxylin erkannt (automatisch).")

# -------------------- Anzeige der Ergebnisse (Overlay aktualisiert) --------------------
overlay = image_disp.copy()
for (x,y) in st.session_state.aec_points:
    cv2.circle(overlay, (x,y), circle_radius, (255,0,0), line_thickness)
for (x,y) in st.session_state.hema_points:
    cv2.circle(overlay, (x,y), circle_radius, (0,0,255), line_thickness)
for (x,y) in st.session_state.bg_points:
    cv2.circle(overlay, (x,y), circle_radius, (255,255,0), line_thickness)
for (x,y) in st.session_state.manual_points:
    cv2.circle(overlay, (x,y), circle_radius, (0,255,0), line_thickness)

st.image(overlay, caption="Markierungen: Rot=AEC, Blau=Hämatoxylin, Gelb=Hintergrund, Grün=manuell", use_column_width=False, width=DISPLAY_WIDTH)

# -------------------- Gesamtanzahl & CSV Export --------------------
all_points = []
types = []
for p in st.session_state.aec_points:
    all_points.append(p); types.append("AEC")
for p in st.session_state.hema_points:
    all_points.append(p); types.append("Hämatoxylin")
for p in st.session_state.manual_points:
    all_points.append(p); types.append("manuell")

st.markdown(f"### 🔢 Gesamtanzahl Kerne: {len(all_points)}  (AEC={len(st.session_state.aec_points)} | Hema={len(st.session_state.hema_points)} | manuell={len(st.session_state.manual_points)})")

if all_points:
    df = pd.DataFrame(all_points, columns=["X_display","Y_display"])
    df["Type"] = types
    df["X_original"] = (df["X_display"] / scale).round().astype("Int64")
    df["Y_original"] = (df["Y_display"] / scale).round().astype("Int64")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne_calibrated.csv", mime="text/csv")
else:
    st.info("Keine Punkte zum Exportieren.")

# -------------------- Reset Buttons --------------------
colr1, colr2 = st.columns(2)
with colr1:
    if st.button("🧹 Automatische Punkte zurücksetzen"):
        st.session_state.aec_points = []
        st.session_state.hema_points = []
        st.success("Automatische Punkte entfernt.")
with colr2:
    if st.button("🔄 Alles zurücksetzen (inkl. Kalibrierung)"):
        for k in ["aec_points","hema_points","bg_points","manual_points","aec_hsv","hema_hsv","bg_hsv"]:
            st.session_state[k] = [] if "points" in k else None
        st.info("Alles gelöscht. Lade ein Bild neu, wenn nötig.")
