"""Streamlit demo. Run from the project root:

    streamlit run app.py

Uploaded images are only held in memory for the duration of the request --
nothing is written to disk here.
"""
import cv2
import numpy as np
import streamlit as st

from src.predict import Predictor

DISCLAIMER = (
    "This tool provides an AI-based prediction and is not definitive proof that "
    "media is real or manipulated. It was trained only on StyleGAN-generated faces "
    "and does NOT reliably detect diffusion-model images (Midjourney, Stable Diffusion, "
    "etc.) -- confirmed by manual testing. This is a portfolio demonstration, not a "
    "safety or forensic tool -- do not rely on it for real decisions about real images."
)

st.set_page_config(page_title="AI-Generated Media & Deepfake Detector")
st.title("AI-Generated Media & Deepfake Detector")
st.caption(DISCLAIMER)


@st.cache_resource
def get_predictor():
    return Predictor()


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded image", width=300)

    show_gradcam = st.checkbox("Show Grad-CAM explanation (which region influenced the prediction)")

    if st.button("Analyze Image"):
        try:
            predictor = get_predictor()
        except FileNotFoundError:
            st.error("No trained model found at models/best_model.pt yet. Run training first (see README).")
        else:
            with st.spinner("Analyzing..."):
                result = predictor.predict_with_gradcam(image_bgr) if show_gradcam else predictor.predict(image_bgr)

            if "error" in result:
                st.warning(result["error"])
            else:
                label = result["label"]
                if label == "MANIPULATED":
                    st.error(f"Prediction: POTENTIALLY MANIPULATED")
                elif label == "UNCERTAIN":
                    st.warning(f"Prediction: UNCERTAIN")
                else:
                    st.success(f"Prediction: REAL")
                st.write(f"Model confidence: {result['confidence'] * 100:.1f}%")

                st.subheader("Detected Face")
                st.image(result["face_rgb"], width=200)

                if show_gradcam and "gradcam_overlay" in result:
                    st.subheader("Grad-CAM: regions that influenced the prediction")
                    st.image(result["gradcam_overlay"], width=200)
                    st.caption("Heatmap shows where the model focused -- not proof that region is manipulated.")
