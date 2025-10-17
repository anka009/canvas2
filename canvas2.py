import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1)-np.array(p2)) < r

def get_centers(mask, min_area=50):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                centers.append((cx,cy))
    return centers

def compute_hsv_range(points, hsv_img, buffer_h=5, buffer_s=20, buffer_v=20):
    if not points:
        return None
    hsv_vals = np.array([hsv_img[y,x] for x,y in points])
    h_min = max(0,int(np.min(hsv_vals[:,0])-buffer_h))
    h_max = min(180,int(np.max(hsv_vals[:,0])+buffer_h))
    s_min = max(0,int(np.min(hsv_vals[:,1])-buffer_s))
    s_max = min(255,int(np.max(hsv_vals[:,1])+buffer_s))
    v_min = max(0,int(np.min(hsv_vals[:,2])-buffer_v))
    v_max = min(255,int(np.max(hsv_vals[:,2])+buffer_v))
    return (h_min,h_max,s_min,s_max,v_min,v_max)

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Interaktiver Zellkern-Zähler", layout="wide")
st.title("🧬 Interaktiver Zellkern-Zähler (Kalibrierung+Auto)")

# -------------------- Session State Defaults --------------------
keys_defaults = {
    "aec_points": [], "hema_points": [], "manual_points": [], "bg_points": [],
    "delete_mode": False, "last_file": None, "disp_width": 1400,
    "blur_kernel": 5, "alpha": 1.0, "circle_radius": 8, "line_thickness": 2,
    "aec_hsv": None, "hema_hsv": None, "bg_hsv": None
}
for k,v in keys_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","png","tif","tiff","jpeg"])

if uploaded_file:
    # Reset bei neuem Bild
    if uploaded_file.name != st.session_state.last_file:
        st.session_state.aec_points = []
        st.session_state.hema_points = []
        st.session_state.manual_points = []
        st.session_state.bg_points = []
        st.session_state.delete_mode = False
        st.session_state.last_file = uploaded_file.name

    # -------------------- Bildbreite-Slider --------------------
    colW1, colW2 = st.columns([2,1])
    with colW1:
        DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100, key="disp_width_slider")
        st.session_state.disp_width = DISPLAY_WIDTH
    with colW2:
        st.write("Breite anpassen")

    # -------------------- Bild vorbereiten --------------------
    image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
    H_orig, W_orig = image_orig.shape[:2]
    scale = DISPLAY_WIDTH / W_orig
    image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig*scale)), interpolation=cv2.INTER_AREA)
    hsv_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

    # -------------------- Slider für Parameter --------------------
    st.markdown("### ⚙️ Filter- und Erkennungsparameter")
    col1, col2 = st.columns(2)
    with col1:
        blur_kernel = st.slider("🔧 Blur", 1, 21, st.session_state.blur_kernel, step=2, key="blur_slider")
        st.session_state.blur_kernel = blur_kernel
    with col2:
        alpha = st.slider("🌗 Alpha", 0.1, 3.0, st.session_state.alpha, step=0.1, key="alpha_slider")
        st.session_state.alpha = alpha

    col3, col4 = st.columns(2)
    with col3:
        circle_radius = st.slider("⚪ Kreisradius", 3, 20, st.session_state.circle_radius, key="circle_slider")
        st.session_state.circle_radius = circle_radius
    with col4:
        line_thickness = st.slider("📏 Linienstärke", 1, 5, st.session_state.line_thickness, key="thickness_slider")
        st.session_state.line_thickness = line_thickness

    # -------------------- Checkbox-Modi --------------------
    colA, colB, colC = st.columns(3)
    with colA:
        st.session_state.delete_mode = st.checkbox("🗑️ Löschmodus aktivieren", key="delete_mode_cb")
    with colB:
        aec_mode = st.checkbox("🔴 AEC markieren", key="aec_mode_cb")
    with colC:
        hema_mode = st.checkbox("🔵 Hämatoxylin markieren", key="hema_mode_cb")
    bg_mode = st.checkbox("🖌 Hintergrund markieren", key="bg_mode_cb")

    # -------------------- Bildanzeige --------------------
    marked_disp = image_disp.copy()
    color_map = {
        "aec": (255,0,0),
        "hema": (0,0,255),
        "manual": (0,255,0),
        "bg": (255,255,0)
    }
    for (x,y) in st.session_state.aec_points:
        cv2.circle(marked_disp,(x,y),circle_radius,color_map["aec"],line_thickness)
    for (x,y) in st.session_state.hema_points:
        cv2.circle(marked_disp,(x,y),circle_radius,color_map["hema"],line_thickness)
    for (x,y) in st.session_state.manual_points:
        cv2.circle(marked_disp,(x,y),circle_radius,color_map["manual"],line_thickness)
    for (x,y) in st.session_state.bg_points:
        cv2.circle(marked_disp,(x,y),circle_radius,color_map["bg"],line_thickness)

    coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key="clickable_image", width=DISPLAY_WIDTH)

    # -------------------- Klick-Logik --------------------
    if coords is not None:
        x, y = coords["x"], coords["y"]
        if bg_mode:
            st.session_state.bg_points.append((x,y))
        elif st.session_state.delete_mode:
            for key in ["aec_points","hema_points","manual_points","bg_points"]:
                st.session_state[key] = [p for p in st.session_state[key] if not is_near(p,(x,y),r=circle_radius)]
        elif aec_mode:
            st.session_state.aec_points.append((x,y))
        elif hema_mode:
            st.session_state.hema_points.append((x,y))
        else:
            st.session_state.manual_points.append((x,y))

    # -------------------- Kalibrierungs-Button --------------------
    if st.button("⚡ Automatische Kalibrierung"):
        st.session_state.aec_hsv = compute_hsv_range(st.session_state.aec_points, hsv_disp)
        st.session_state.hema_hsv = compute_hsv_range(st.session_state.hema_points, hsv_disp)
        st.session_state.bg_hsv = compute_hsv_range(st.session_state.bg_points, hsv_disp)
        st.success("Kalibrierung abgeschlossen! 🔧")

    # -------------------- Auto-Erkennung --------------------
    if st.button("🤖 Auto-Erkennung starten"):
        mask_fg = np.ones(image_disp.shape[:2], dtype=np.uint8)*255
        if st.session_state.bg_hsv:
            h_min,h_max,s_min,s_max,v_min,v_max = st.session_state.bg_hsv
            mask_bg = cv2.inRange(hsv_disp, np.array([h_min,s_min,v_min]), np.array([h_max,s_max,v_max]))
            mask_fg = cv2.bitwise_not(mask_bg)

        kernel = np.ones((3,3),np.uint8)

        # AEC
        aec_points = []
        if st.session_state.aec_hsv:
            hmin,hmax,smin,smax,vmin,vmax = st.session_state.aec_hsv
            mask_aec = cv2.inRange(hsv_disp,np.array([hmin,smin,vmin]),np.array([hmax,smax,vmax]))
            mask_aec = cv2.bitwise_and(mask_aec, mask_aec, mask=mask_fg)
            mask_aec = cv2.morphologyEx(mask_aec, cv2.MORPH_OPEN,kernel,iterations=1)
            aec_points = get_centers(mask_aec, min_area=50)
        st.session_state.aec_points = aec_points

        # Hämatoxylin
        hema_points = []
        if st.session_state.hema_hsv:
            hmin,hmax,smin,smax,vmin,vmax = st.session_state.hema_hsv
            mask_hema = cv2.inRange(hsv_disp,np.array([hmin,smin,vmin]),np.array([hmax,smax,vmax]))
            mask_hema = cv2.bitwise_and(mask_hema, mask_hema, mask=mask_fg)
            mask_hema = cv2.morphologyEx(mask_hema, cv2.MORPH_OPEN,kernel,iterations=1)
            hema_points = get_centers(mask_hema, min_area=50)
        st.session_state.hema_points = hema_points

        st.success(f"✅ {len(aec_points)} AEC-Kerne, {len(hema_points)} Hämatoxylin-Kerne erkannt!")

    # -------------------- Reset --------------------
    if st.button("🧹 Alles zurücksetzen"):
        for key in ["aec_points","hema_points","manual_points","bg_points"]:
            st.session_state[key]=[]
        st.info("Alle Punkte gelöscht!")

    # -------------------- Gesamtanzahl --------------------
    all_points = st.session_state.aec_points + st.session_state.hema_points + st.session_state.manual_points
    st.markdown(f"### 🔢 Gesamtanzahl Kerne: {len(all_points)}")

    # -------------------- CSV Export --------------------
    df = pd.DataFrame(all_points, columns=["X_display","Y_display"])
    if not df.empty:
        df["X_original"] = (df["X_display"]/scale).round().astype("Int64")
        df["Y_original"] = (df["Y_display"]/scale).round().astype("Int64")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")
