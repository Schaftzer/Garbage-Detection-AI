
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("Simple Garbage AI Scanner")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload gambar sampah",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Asli", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name, conf=0.25)

    result_img = results[0].plot()
    st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.subheader("Terdeteksi:")
        for c in boxes.cls.tolist():
            st.write("-", model.names[int(c)])
    else:
        st.write("Tidak ada sampah terdeteksi")
    