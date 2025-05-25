import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import torch
import ultralytics.nn.tasks as tasks

# ------------------ Allow safe loading of DetectionModel ------------------
torch.serialization.add_safe_globals([tasks.DetectionModel])

# ------------------ Constants ------------------
MODEL_PATH = "rust-corrosion.pt"

# ------------------ Streamlit Page Config ------------------
st.set_page_config(page_title="YOLOv8 Inference App", layout="centered")
st.title("🔍 YOLOv8 Image Detector")

# ------------------ Model Loading with Streamlit Cache ------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ------------------ Image Input Method ------------------
input_method = st.radio("Choose Image Input Method:", ("📤 Upload", "📷 Camera"))

image = None  # Initialize image variable

if input_method == "📤 Upload":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if image_file:
        image = Image.open(image_file)

elif input_method == "📷 Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(camera_image)

# ------------------ Perform Inference ------------------
if image is not None:
    st.image(image, caption="🖼️ Input Image", use_column_width=True)

    with st.spinner("🔍 Detecting objects..."):
        results = model(image)
        result_img = results[0].plot()  # Get image with boxes drawn

    st.image(result_img, caption="🧠 Detection Result", use_column_width=True)

    # Save detection result for download
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        Image.fromarray(result_img).save(tmp.name)
        st.download_button(
            label="📥 Download Output",
            data=open(tmp.name, "rb").read(),
            file_name="detected.jpg",
            mime="image/jpeg"
        )

else:
    st.info("Please provide an image using Upload or Camera.")