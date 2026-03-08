# Emotion Face Detection System 🎭

This project is an **Emotion Face Detection System** built using Machine Learning and OpenCV, with a beautiful and interactive user interface powered by Streamlit.

## Features
- **Face Detection:** Uses OpenCV's Haar Cascades (`haarcascade_frontalface_default.xml`) to accurately identify faces in real-time.
- **Emotion Recognition:** Uses **DeepFace** (a powerful Machine Learning framework) to classify the emotion of detected faces into 7 categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
- **Web Interface:** Fully integrated with Streamlit to provide two modes:
  - **Live Camera Input:** Capture an image directly from your webcam.
  - **File Upload:** Upload existing images (JPG, PNG) for analysis.

## Required Technologies
- Python 3.7+
- Streamlit
- OpenCV
- DeepFace
- TensorFlow (Backend for DeepFace)

## Installation Instructions

1. **Install Dependencies**
   If you have Python installed, you can install the required packages directly via:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   Start the Streamlit web application:
   ```bash
   streamlit run app.py
   ```

3. **Access the Application**
   The UI will open in your default browser at `http://localhost:8501`. 

> *Note: The first time you run an emotion detection query, DeepFace will automatically download the pre-trained neural network weights (~17MB). Subsequent runs will be much faster!*
