import streamlit as st
import numpy as np
import tempfile
import time
import os
from pathlib import Path

# Try to import OpenCV first. Ultralytics imports cv2 internally, so a missing cv2
# can raise during the ultralytics import. We handle both imports gracefully
# and present clear messages in the Streamlit UI.
try:
    import cv2
except Exception as e:
    cv2 = None
    cv2_import_error = e
else:
    cv2_import_error = None

# Try to import Ultralytics' YOLO. If this fails, capture the error and
# show guidance rather than letting the app crash with an opaque traceback.
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    ultralytics_import_error = e
else:
    ultralytics_import_error = None

# Inform users via Streamlit if key dependencies are missing (don't raise here
# so the app can start and show a friendly message).
try:
    if cv2 is None:
        st.error("OpenCV (cv2) is not installed. Install it with: pip install opencv-python")
    if YOLO is None:
        st.error("Ultralytics failed to import. Install it with: pip install ultralytics (and ensure opencv-python is installed).")
except Exception:
    # If Streamlit UI isn't available at import time, skip UI messaging.
    pass

# Load the YOLO model only if the package imported correctly. If the model
# file is missing or loading fails, capture that and show a helpful message
# at runtime rather than crashing the app during import.
model = None
MODEL_PATH = Path("trained_model.pt")
if YOLO is not None:
    if MODEL_PATH.exists():
        try:
            model = YOLO(str(MODEL_PATH))
        except Exception as e:
            model = None
            try:
                st.error(f"Failed to load YOLO model '{MODEL_PATH}': {e}")
            except Exception:
                pass
    else:
        try:
            st.warning(f"Model file not found at '{MODEL_PATH}'. Place your .pt model there or update the path.")
        except Exception:
            pass

def run_image_detection(conf_threshold):
    st.subheader("Image Detection Mode")
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        if cv2 is None:
            st.error("OpenCV (cv2) is required for image decoding. Install opencv-python and restart the app.")
            return
        if model is None:
            st.error("YOLO model is not available. Ensure 'ultralytics' is installed and 'trained_model.pt' exists in the app folder.")
            return
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = model.predict(source=image, conf=conf_threshold, show=False)
        annotated_image = results[0].plot()
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, channels="BGR", caption="Uploaded Image", width=300)
        with col2:
            st.image(annotated_image, channels="BGR", caption="Detection Result", width=300)

def run_video_detection(conf_threshold):
    st.subheader("Video Detection Mode")
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    if uploaded_video:
        if cv2 is None:
            st.error("OpenCV (cv2) is required for video processing. Install opencv-python and restart the app.")
            return
        if model is None:
            st.error("YOLO model is not available. Ensure 'ultralytics' is installed and 'trained_model.pt' exists in the app folder.")
            return
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())
        cap = cv2.VideoCapture(temp_file.name)
        video_placeholder = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=conf_threshold, show=False)
            annotated_frame = results[0].plot()
            video_placeholder.image(annotated_frame, channels="BGR")
            time.sleep(0.03)
        cap.release()

def run_webcam_detection(conf_threshold):
    st.subheader("Webcam Live Detection")
    
    # Import streamlit-webrtc and av (make sure these are in your requirements.txt)
    try:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    except Exception:
        st.error("Package 'streamlit-webrtc' is required for webcam streaming. Install it with: pip install streamlit-webrtc")
        return
    try:
        import av
    except Exception:
        st.error("Package 'av' is required for webcam streaming. Install it with: pip install av")
        return

    if model is None:
        st.error("YOLO model is not available. Ensure 'ultralytics' is installed and 'trained_model.pt' exists in the app folder.")
        return

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.conf_threshold = conf_threshold

        def transform(self, frame):
            # Convert frame to an OpenCV image
            if cv2 is None:
                # If cv2 missing, return original frame as-is
                return frame.to_ndarray(format="bgr24")
            img = frame.to_ndarray(format="bgr24")
            # Run YOLO detection on the frame
            results = model.predict(source=img, conf=self.conf_threshold, show=False)
            annotated_frame = results[0].plot()
            return annotated_frame

    # Explicitly set media stream constraints to enable video and disable audio
    webrtc_streamer(
        key="webcam",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

def main():
    st.set_page_config(page_title="YOLO Object Detection", layout="wide")
    st.markdown(
        "<h1 style='text-align: center; color: blue;'>YOLO Object Detection App</h1>",
        unsafe_allow_html=True
    )
    
    st.sidebar.header("Configuration Panel")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    mode = st.sidebar.radio("Select Detection Mode", ["Image", "Video", "Webcam"])
    
    if mode == "Image":
        run_image_detection(conf_threshold)
    elif mode == "Video":
        run_video_detection(conf_threshold)
    elif mode == "Webcam":
        run_webcam_detection(conf_threshold)

if __name__ == '__main__':
    main()
