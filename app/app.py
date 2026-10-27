"""Fresh-versus-Rotten Fruit Classifier — Streamlit prototype.

Upload a photo OR use your webcam; the app shows the predicted fruit + freshness,
a confidence score, the full probability breakdown, and inference latency / FPS.
"""
import streamlit as st
from PIL import Image

from model_utils import load_model, predict

st.set_page_config(page_title="Fresh vs Rotten Fruit Classifier",
                   page_icon="🍎", layout="centered")

st.title("🍎 Fresh vs Rotten Fruit Classifier")
st.caption("MobileNetV3 · 6 classes (Apple / Banana / Orange × Fresh / Rotten) · "
           "Deep Learning Lab Capstone, Ramdeobaba University")


@st.cache_resource(show_spinner="Loading model…")
def _get_model():
    return load_model()


try:
    model, labels = _get_model()
    model_ready = True
except FileNotFoundError as e:
    model_ready = False
    st.error(str(e))
    st.info("Once trained, place `mobilenetv3_fresh_rotten.pt` and `labels.json` "
            "in the `app/models/` folder, then reload.")

st.subheader("Choose an input")
tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Use webcam"])

image = None
with tab_upload:
    up = st.file_uploader("Upload a fruit image", type=["png", "jpg", "jpeg"])
    if up is not None:
        image = Image.open(up)

with tab_camera:
    cam = st.camera_input("Take a photo of the fruit")
    if cam is not None:
        image = Image.open(cam)


def freshness_badge(pretty_label: str):
    if pretty_label.lower().startswith("fresh"):
        st.success(f"✅ Prediction: **{pretty_label}**")
    else:
        st.warning(f"⚠️ Prediction: **{pretty_label}**")


if image is not None and model_ready:
    st.image(image, caption="Input image", use_container_width=True)
    with st.spinner("Classifying…"):
        pretty, conf, prob_map, latency_ms = predict(model, labels, image)

    freshness_badge(pretty)
    st.metric("Confidence", f"{conf * 100:.1f}%")

    st.markdown("**Class probabilities**")
    for name, p in sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True):
        st.write(name)
        st.progress(min(max(p, 0.0), 1.0))

    fps = 1000.0 / latency_ms if latency_ms > 0 else float("inf")
    st.caption(f"Inference latency: {latency_ms:.1f} ms  ·  ~{fps:.1f} FPS (CPU)")
elif image is not None and not model_ready:
    st.image(image, caption="Input image (model not loaded)", use_container_width=True)

with st.expander("About this prototype"):
    st.markdown(
        "- **Model:** MobileNetV3-Large fine-tuned via transfer learning.\n"
        "- **Preprocessing:** resize → center-crop 224 → ImageNet normalization.\n"
        "- **Classes:** Fresh/Rotten × Apple/Banana/Orange.\n"
        "- Trained on the Kaggle *Fruits fresh and rotten* dataset "
        "(70/15/15 split) and validated on locally photographed fruit."
    )
