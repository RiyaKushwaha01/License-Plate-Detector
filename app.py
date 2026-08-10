import streamlit as st
from PIL import Image
import os

from predict import detect_plate
from preprocess import read_plate

st.title("License Plate Recognition")

uploaded = st.file_uploader(
    "Upload Car Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:

    image = Image.open(uploaded)

    st.image(image)

    os.makedirs("uploads", exist_ok=True)

    path = os.path.join("uploads", uploaded.name)

    image.save(path)

    crops = detect_plate(path)

    if len(crops) == 0:

        st.error("No plate detected")

    else:

        for crop in crops:

            st.image(crop)

            plate = read_plate(crop)

            st.success(f"License Plate : {plate}")