import os

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8080")

st.title("Detector de caídas (Fall / Not Fall)")

file = st.file_uploader("Subir imagen", type=["jpg", "png"])

if file:
    image = Image.open(file)
    st.image(image, caption="Imagen original", width="stretch")

    file.seek(0)
    cnn_resp = requests.post(
        f"{API_URL}/predict",
        files={"file": file}
    )
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
    grad_resp = requests.post(
        f"{API_URL}/gradcam",
        files={"file": file}
    )

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
            st.text("Respuesta inválida del servidor")
