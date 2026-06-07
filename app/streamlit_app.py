import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import json

API_URL = "http://fastapi:8080"

st.set_page_config(layout="wide")
st.title(" Detector de Caídas")

# =============================
# DASHBOARD (METRICAS)
# =============================

def fetch_stats():
    try:
        resp = requests.get(f"{API_URL}/dashboard/stats", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except:
        return None

stats = fetch_stats()

if stats:
    st.header("📊 Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Caídas hoy", stats.get("falls_today", 0))
    c2.metric("Caídas semana", stats.get("falls_week", 0))
    c3.metric("Alto riesgo", stats.get("high_risk_persons", 0))

    enabled = stats.get("analytics_enabled", False)
    c4.metric("Analytics", "ON" if enabled else "OFF")

    # métricas modelo
    model = stats.get("model", {})
    valid = model.get("splits", {}).get("valid", {})

    if valid:
        st.subheader("📈 Métricas del modelo")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Accuracy", f"{valid.get('accuracy', 0)*100:.1f}%")
        m2.metric("F1", f"{valid.get('f1_score', 0)*100:.1f}%")
        m3.metric("Precision", f"{valid.get('precision', 0)*100:.1f}%")
        m4.metric("Recall", f"{valid.get('recall', 0)*100:.1f}%")

# =============================
# INPUT DE IMÁGENES
# =============================
st.header("📷 Predicción de Imágenes")

files = st.file_uploader(
    "Subir imágenes",
    type=["jpg", "png"],
    accept_multiple_files=True
)

if files:
    for file in files:

        st.markdown("---")
        st.subheader(f"📷 {file.name}")

        image = Image.open(file)

        # =========================
        # REQUEST API
        # =========================
        file.seek(0)
        cnn = requests.post(f"{API_URL}/predict", files={"file": file}).json()

        file.seek(0)
        grad = requests.post(f"{API_URL}/gradcam", files={"file": file})

        # =========================
        # LAYOUT IMAGEN
        # =========================
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Imagen original", use_container_width=True)

        with col2:
            content_type = grad.headers.get("content-type", "")

            if grad.status_code == 200 and "image" in content_type:
                grad_img = Image.open(BytesIO(grad.content))
                st.image(grad_img, caption="Grad-CAM", use_container_width=True)

            else:
                st.warning("No se pudo generar Grad-CAM")

        # =========================
        # RESULTADOS
        # =========================
        st.subheader("🔍 Resultado")

        label = cnn.get("label", "N/A")
        confidence = cnn.get("confidence", 0)

        color = "green" if label == "fall" else "red"

        st.markdown(f"""
        **Predicción:** :{color}[{label}]  
        **Confianza:** {confidence*100:.2f}%
        """)

        if confidence < 0.7:
            st.warning("⚠️ Baja confianza en la predicción")

        st.json(cnn)

# =============================
# VIDEO
# =============================
st.header("🎥 Predicción de Video")

video_file = st.file_uploader("Subir video", type=["mp4"], key="video")

if video_file:
    st.video(video_file)

    if st.button("Procesar Video"):

        with st.spinner("Procesando video..."):

            files = {"file": video_file}

            response = requests.post(
                f"{API_URL}/predict/video",
                files=files
            )

            if response.status_code == 200:
                result = response.json()

                st.success("✅ Video procesado")

                # ✅ resumen
                st.write(f"Frames analizados: {result['total_frames']}")
                st.write(f"⚠️ Caídas detectadas: {result['falls_detected']}")

                st.subheader("📊 Resultados (preview)")

                st.write(result["preview"])

                # ✅ mostrar solo caídas
                falls = [r for r in result["preview"] if r["is_fall"]]

                if falls:
                    st.warning("⚠️ Caídas detectadas en preview")
                    st.write(falls)

            else:
                st.error("Error procesando video")