import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO

# Load the YOLO model (adjust the path as needed)
model = YOLO(r"trained_model.pt")

def run_image_detection(conf_threshold):
    st.subheader("Image Detection Mode")
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        # Convert the uploaded image to a NumPy array
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Run YOLO detection on the image
        results = model.predict(source=image, conf=conf_threshold, show=False)
        annotated_image = results[0].plot()
        
        # Display original and detection images side by side with reduced width
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, channels="BGR", caption="Uploaded Image", width=300)
        with col2:
            st.image(annotated_image, channels="BGR", caption="Detection Result", width=300)

def run_video_detection(conf_threshold):
    st.subheader("Video Detection Mode")
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    if uploaded_video:
        # Save the uploaded video to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())
        cap = cv2.VideoCapture(temp_file.name)
        video_placeholder = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Run YOLO detection on the current frame
            results = model.predict(source=frame, conf=conf_threshold, show=False)
            annotated_frame = results[0].plot()
            video_placeholder.image(annotated_frame, channels="BGR")
            time.sleep(0.03)  # Control frame rate
        cap.release()

def run_webcam_detection(conf_threshold):
    st.subheader("Webcam Live Detection")
    # Use streamlit-webrtc for client-side webcam access
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    import av

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            # Set the confidence threshold from the current slider value
            self.conf_threshold = conf_threshold

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model.predict(source=img, conf=self.conf_threshold, show=False)
            annotated_frame = results[0].plot()
            return annotated_frame

    webrtc_streamer(key="webcam", video_transformer_factory=lambda: VideoTransformer())

def main():
    st.set_page_config(page_title="YOLO Object Detection", layout="wide")
    st.markdown(
        "<h1 style='text-align: center; color: blue;'>YOLO Object Detection App</h1>",
        unsafe_allow_html=True
    )
    
    # Sidebar configuration with header and controls
    st.sidebar.header("Configuration Panel")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    mode = st.sidebar.radio("Select Detection Mode", ["Image", "Video", "Webcam"])
    
    # Run detection based on the selected mode
    if mode == "Image":
        run_image_detection(conf_threshold)
    elif mode == "Video":
        run_video_detection(conf_threshold)
    elif mode == "Webcam":
        run_webcam_detection(conf_threshold)

if __name__ == '__main__':
    main()
