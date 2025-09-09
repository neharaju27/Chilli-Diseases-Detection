import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # replace with your trained chilli model

model = load_model()

# List of chilli disease/pest classes
disease_classes = [
    'Anthracnose_and_Mosaic_Virus', 'Aphids', 'Armyworm', 'Bacterial_Leaf_Spot',
    'Caterpillar', 'Cercospora_Leaf_Spot', 'Fusarium_Wilt', 'Leafcurl_Virus',
    'Mites', 'Powdery_Mildew', 'Thirps', 'Whitefly'
]

st.title("🌶️ Chilli Plant Pest & Disease Detection")
st.info("ℹ️ **Note:** This model can detect only these diseases/pests: " + ", ".join(disease_classes))

# Image uploader
uploaded_file = st.file_uploader("Upload a Chilli Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image (Original)", use_column_width=True)

    # Save temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    st.write("🔍 Detecting diseases/pests...")

    # Run YOLO inference
    results = model.predict(source=temp_file.name, conf=0.25, save=False)

    # Display results
    for r in results:
        # Annotated image (BGR by default)
        im_array = r.plot()
        # Convert to RGB for Streamlit
        im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

        st.image(im_array, caption="Detected Diseases/Pests", use_column_width=True)

        # Extract detected class names
        detected = [disease_classes[int(c)] for c in r.boxes.cls.cpu().numpy()]
        st.subheader("🦠 Detected Diseases/Pests:")
        if detected:
            st.write(", ".join(set(detected)))
        else:
            st.write("✅ No disease/pest detected!")

    # Cleanup temp file
    os.remove(temp_file.name)
