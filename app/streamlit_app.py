import streamlit as st
import requests
from PIL import Image
from io import BytesIO

API_URL = "http://fastapi:8080"

st.set_page_config(layout="wide")

st.title("🛏️ Detector de paciente en camilla")

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
        # PREDICCIONES
        # =========================
        file.seek(0)
        cnn = requests.post(f"{API_URL}/predict", files={"file": file}).json()

        file.seek(0)
        yolo = requests.post(f"{API_URL}/predict_yolo", files={"file": file}).json()

        file.seek(0)
        grad = requests.post(f"{API_URL}/gradcam", files={"file": file})

        # =========================
        # LAYOUT
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
                st.warning("No se pudo generar GradCAM")

        # =========================
        # RESULTADOS
        # =========================
        st.markdown("### 🔍 Resultados")

        col1, col2 = st.columns(2)

        def pretty_result(result, title):
            pred = result.get("prediction", None)
            label = result.get("label", "N/A")

            color = "green" if pred == 1 else "red"

            st.markdown(f"""
            ### {title}
            - **Predicción:** :{color}[{label}]
            """)

        with col1:
            pretty_result(cnn, "🧠 CNN")

        with col2:
            pretty_result(yolo, "🤖 YOLO")