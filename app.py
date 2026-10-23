
import streamlit as st
import cv2
import numpy as np
import re
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import easyocr


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

st.set_page_config(
    page_title="License Plate Recognition",
    page_icon="🚘",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_yolo_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Place best.pt inside the models folder."
        )
    return YOLO(str(MODEL_PATH))


@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(["en"], gpu=False)


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Enlarge the plate to help OCR read small characters
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    return enhanced


# --------------------------------------------------
# OCR
# --------------------------------------------------

def read_plate(plate, reader):
    processed = preprocess_plate(plate)

    # Allow common alphanumeric plate characters
    ocr_results = reader.readtext(
        processed,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    # Keep text regions in their left-to-right order
    ocr_results = sorted(
        ocr_results,
        key=lambda item: min(point[0] for point in item[0])
    )

    accepted = [
        (text, float(confidence))
        for _, text, confidence in ocr_results
        if confidence >= 0.20
    ]

    raw_text = "".join(text for text, _ in accepted)
    cleaned_text = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()

    if accepted:
        ocr_confidence = sum(
            confidence for _, confidence in accepted
        ) / len(accepted)
    else:
        ocr_confidence = 0.0

    return cleaned_text, processed, ocr_confidence


# --------------------------------------------------
# YOLO DETECTION
# --------------------------------------------------

def detect_plates(image_rgb, model, reader, confidence):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(
        source=image_bgr,
        conf=confidence,
        imgsz=640,
        verbose=False
    )

    annotated = image_rgb.copy()
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().tolist()
            )

            # Keep coordinates within image boundaries
            height, width = image_bgr.shape[:2]
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            plate_crop = image_bgr[y1:y2, x1:x2].copy()
            if plate_crop.size == 0:
                continue

            plate_text, processed, ocr_confidence = read_plate(
                plate_crop, reader
            )

            detection_confidence = float(box.conf[0].item())
            class_id = int(box.cls[0].item())
            class_name = model.names.get(class_id, "License plate")

            # Draw detection on the original image
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 200, 0),
                3
            )

            label = plate_text if plate_text else "Plate detected"
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 0),
                2,
                cv2.LINE_AA
            )

            detections.append({
                "text": plate_text,
                "detection_confidence": detection_confidence,
                "ocr_confidence": ocr_confidence,
                "crop": cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB),
                "processed": processed,
                "box": (x1, y1, x2, y2),
                "class_name": class_name
            })

    return annotated, detections


# --------------------------------------------------
# STREAMLIT INTERFACE
# --------------------------------------------------

st.title("🚘 License Plate Recognition System")

st.write(
    "Upload a car image to detect the license plate "
    "and extract its registration characters using YOLO and OCR."
)

with st.sidebar:
    st.header("Settings")
    confidence = st.slider(
        "YOLO detection confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )
    st.caption(
        "Lower confidence may detect more plates but can also "
        "produce false detections."
    )

uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(image)

        st.subheader("Uploaded Image")
        st.image(image_rgb, use_container_width=True)

        if st.button("Detect License Plate", type="primary"):
            with st.spinner("Loading models and processing image..."):
                model = load_yolo_model()
                reader = load_ocr_model()

                annotated, detections = detect_plates(
                    image_rgb,
                    model,
                    reader,
                    confidence
                )

            st.subheader("Detection Results")

            left, right = st.columns(2)

            with left:
                st.markdown("**Detected image**")
                st.image(annotated, use_container_width=True)

            with right:
                st.markdown("**Recognized plate number(s)**")

                if not detections:
                    st.warning(
                        "No license plate detected. Try a clearer "
                        "image or lower the detection confidence."
                    )

                for index, detection in enumerate(detections, start=1):
                    st.markdown(f"### Plate {index}")

                    if detection["text"]:
                        st.success(detection["text"])
                    else:
                        st.warning(
                            "Plate detected, but the number could not "
                            "be read confidently. Check the crop below."
                        )

                    st.metric(
                        "YOLO detection confidence",
                        f"{detection['detection_confidence']:.1%}"
                    )

                    st.caption(
                        f"OCR confidence (average): "
                        f"{detection['ocr_confidence']:.1%}"
                    )

                    st.image(
                        detection["crop"],
                        caption="Cropped license plate",
                        use_container_width=True
                    )

                    with st.expander("View preprocessed plate"):
                        st.image(
                            detection["processed"],
                            caption="Image sent to OCR",
                            clamp=True,
                            use_container_width=True
                        )

                    st.download_button(
                        "Download cropped plate",
                        data=cv2.imencode(
                            ".jpg",
                            cv2.cvtColor(
                                detection["crop"],
                                cv2.COLOR_RGB2BGR
                            )
                        )[1].tobytes(),
                        file_name=f"license_plate_{index}.jpg",
                        mime="image/jpeg",
                        key=f"download_{index}"
                    )

    except Exception as error:
        st.error(f"An error occurred: {error}")
        st.info(
            "Check that models/best.pt exists and that all "
            "dependencies are installed."
        )

st.divider()
st.caption(
    "Note: OCR results may be incorrect for blurry, angled, "
    "or low-resolution plates. Verify the extracted number."
)