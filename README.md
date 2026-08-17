# 🚗 License Plate Recognition System

An end-to-end **License Plate Recognition (LPR)** application that detects vehicle license plates from car images using **YOLO** and extracts the license plate number using **OCR**.

The application provides a simple **Streamlit web interface** where users can upload a car image and receive the detected license plate number as the output.

---

## 📌 Project Overview

The system follows a two-stage computer vision pipeline:

```text
Car Image
    ↓
YOLO License Plate Detection
    ↓
License Plate Crop
    ↓
Image Preprocessing
    ↓
OCR
    ↓
License Plate Number
```

For example:

```text
Input:
car.jpg

        ↓

YOLO detects:
License Plate

        ↓

OCR reads:
DL01AB1234
```

---

## ✨ Features

* 🚗 Upload car images through a web interface
* 🔍 Detect license plates using YOLO
* ✂️ Automatically crop detected license plates
* 🔤 Extract license plate text using OCR
* 📊 Display detection results
* 🖼️ Display the detected license plate
* 📈 Show detection/OCR confidence
* 💻 Simple Streamlit interface
* 🔄 Supports multiple image formats

Supported formats:

```text
.jpg
.jpeg
.png
```

---

## 🛠️ Technologies Used

| Technology          | Purpose                        |
| ------------------- | ------------------------------ |
| Python              | Main programming language      |
| YOLO                | License plate detection        |
| OpenCV              | Image processing               |
| EasyOCR / PaddleOCR | License plate text recognition |
| Streamlit           | Web application                |
| NumPy               | Numerical operations           |
| Pillow              | Image handling                 |
| Pandas              | Data processing                |

---

## 📂 Project Structure

```text
License-Plate-Recognition/
│
├── app.py
├── predict.py
├── preprocess.py
├── utils.py
├── requirements.txt
├── README.md
│
├── models/
│   └── best.pt
│
├── uploads/
│
├── cropped/
│
└── output/
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/License-Plate-Recognition.git
```

Navigate to the project:

```bash
cd License-Plate-Recognition
```

---

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Example `requirements.txt`:

```text
ultralytics
opencv-python
easyocr
streamlit
pillow
numpy
pandas
```

---

# 🤖 YOLO Model Training

The license plate detection model was trained using a custom annotated dataset.

### Dataset preparation

The dataset contains images of vehicles with bounding-box annotations around the license plates.

Example:

```text
Car Image
┌─────────────────────────────┐
│                             │
│        🚗                   │
│             ┌──────────┐    │
│             │ DL01AB12 │    │
│             └──────────┘    │
│                             │
└─────────────────────────────┘
```

The dataset can be prepared and annotated using **Roboflow** and exported in YOLO format.

---

## 🧠 Training

YOLO can be trained using Google Colab with GPU acceleration.

Example:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16
)
```

After training, the best model is generated at:

```text
runs/detect/train/weights/best.pt
```

Copy the trained model into:

```text
models/best.pt
```

---

# 🔍 License Plate Detection

The trained YOLO model detects license plates in the uploaded vehicle image.

Example:

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model(image)
```

The detected bounding box is then used to crop the license plate.

---

# 🔤 OCR

After detecting the plate, the cropped image is passed to an OCR model.

Example using EasyOCR:

```python
import easyocr

reader = easyocr.Reader(["en"])

result = reader.readtext(plate_image)

for detection in result:
    print(detection[1])
```

The OCR output is then cleaned using post-processing techniques.

---

# 🖥️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

Upload a vehicle image:

```text
📤 Upload Car Image

        ↓

🔍 Detect License Plate

        ↓

✂️ Crop Plate

        ↓

🔤 OCR

        ↓

✅ License Plate: DL01AB1234
```

---

# 📊 Example Output

### Input

A vehicle image containing a visible license plate.

### Detection

```text
License Plate Detected
Confidence: 0.94
```

### OCR Result

```text
DL01AB1234
```

---

# 🚀 Future Improvements

The project can be extended with:

* [ ] Multiple image upload
* [ ] Folder-based batch processing
* [ ] Video license plate recognition
* [ ] Real-time webcam detection
* [ ] Improved OCR using PaddleOCR
* [ ] Automatic image preprocessing
* [ ] Confidence-based filtering
* [ ] Save results to CSV/Excel
* [ ] Vehicle tracking
* [ ] Number plate history/database
* [ ] FastAPI REST API
* [ ] Docker deployment
* [ ] Cloud deployment

---

# 🎯 Use Cases

This system can be used as a foundation for:

* 🚘 Automatic Number Plate Recognition (ANPR)
* 🅿️ Smart parking systems
* 🚧 Toll booth automation
* 🏢 Vehicle entry/exit systems
* 🏭 Industrial vehicle monitoring
* 🔐 Access-control systems
* 🚦 Traffic monitoring

---

# 📈 Machine Learning Pipeline

```text
                    Dataset
                       │
                       ▼
                Data Annotation
                       │
                       ▼
                   Roboflow
                       │
                       ▼
                 YOLO Dataset
                       │
                       ▼
                 Model Training
                       │
                       ▼
                    YOLO
                       │
                       ▼
              License Plate Detection
                       │
                       ▼
                Image Cropping
                       │
                       ▼
               Image Preprocessing
                       │
                       ▼
                     OCR
                       │
                       ▼
              Text Post-Processing
                       │
                       ▼
             License Plate Number
```

---

# 🧪 Model Evaluation

The YOLO model can be evaluated using:

* Precision
* Recall
* mAP50
* mAP50-95
* Confusion Matrix

Example:

```text
Precision : XX
Recall    : XX
mAP@50    : XX
mAP@50-95 : XX
```

Replace the placeholder values with your actual model evaluation results after training.

---

# 🔮 Future Architecture

The final version of the application can be deployed as:

```text
             User
               │
               ▼
        Streamlit / Web UI
               │
               ▼
          Image Upload
               │
               ▼
        YOLO Detection
               │
               ▼
       License Plate Crop
               │
               ▼
             OCR
               │
               ▼
       Post Processing
               │
               ▼
        Plate Number
               │
               ▼
        Database / API
```

---

# 👩‍💻 Author

**Riya Kushwaha**

Data Science | Machine Learning | Computer Vision

---

## ⭐ If you find this project useful

Feel free to ⭐ star the repository and contribute to improving the project.
