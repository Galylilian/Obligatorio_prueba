import os
from io import BytesIO

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8080")


def fetch_stats() -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/dashboard/stats", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


st.set_page_config(page_title="Detector de caidas", layout="wide")
st.title("Detector de caidas (Fall / Not Fall)")

stats = fetch_stats()
if stats:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Caidas hoy", stats.get("falls_today", 0))
    c2.metric("Caidas semana", stats.get("falls_week", 0))
    c3.metric("Personas alto riesgo", stats.get("high_risk_persons", 0))
    enabled = stats.get("analytics_enabled", False)
    c4.metric("Analytics", "ON" if enabled else "OFF")
    if enabled:
        st.caption(
            "Analytics activo. Dashboard Grafana: http://localhost:3000 "
            "(accuracy, F1, KPIs operacionales)."
        )
    else:
        st.warning("Analytics desactivado. Configura DATABASE_URL en .env o usa APP_ENV=development.")

    model = stats.get("model", {})
    valid = model.get("splits", {}).get("valid", {})
    if valid:
        st.subheader("Metricas del modelo (valid)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{valid.get('accuracy', 0) * 100:.1f}%")
        m2.metric("F1", f"{valid.get('f1_score', 0) * 100:.1f}%")
        m3.metric("Precision", f"{valid.get('precision', 0) * 100:.1f}%")
        m4.metric("Recall", f"{valid.get('recall', 0) * 100:.1f}%")

with st.sidebar:
    st.subheader("Registro analitico")
    person_id = st.text_input("person_id (opcional)", value="", placeholder="P042")
    source = st.text_input("source (opcional)", value="streamlit")

file = st.file_uploader("Subir imagen", type=["jpg", "png"])

if file:
    image = Image.open(file)
    st.image(image, caption="Imagen original", width="stretch")

    file.seek(0)
    data = {}
    if person_id.strip():
        data["person_id"] = person_id.strip()
    if source.strip():
        data["source"] = source.strip()

    cnn_resp = requests.post(f"{API_URL}/predict", files={"file": file}, data=data)
    cnn_result = cnn_resp.json()

    st.subheader("Resultados CNN")
    if "label" in cnn_result:
        label = cnn_result["label"]
        conf = cnn_result.get("confidence")
        if conf is not None:
            conf_pct = float(conf) * 100
            st.success(f"Prediccion: **{label}** (confianza: {conf_pct:.1f}%)")
            if conf_pct < 70:
                st.warning(
                    "Confianza baja: el modelo no esta seguro. "
                    "Proba con imagenes del dataset o del mismo tipo de camara."
                )
        else:
            st.success(f"Prediccion: **{label}**")
        if "probabilities" in cnn_result:
            st.bar_chart(cnn_result["probabilities"])
    st.json(cnn_result)

    file.seek(0)
    grad_resp = requests.post(f"{API_URL}/gradcam", files={"file": file})

    content_type = grad_resp.headers.get("content-type", "")

    if grad_resp.status_code == 200 and "image" in content_type:
        try:
            grad_img = Image.open(BytesIO(grad_resp.content))
            st.image(grad_img, caption="Grad-CAM", width="stretch")
        except Exception as e:
            st.error("Error al interpretar Grad-CAM")
            st.text(str(e))
    else:
        st.warning("No se pudo generar Grad-CAM")
        try:
            st.text(grad_resp.text)
        except Exception:
            st.text("Respuesta invalida del servidor")
