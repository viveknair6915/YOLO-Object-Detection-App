import os
import sys
import time
import tempfile
from pathlib import Path
import numpy as np
import streamlit as st

# ==============================================================================
# 1. Streamlit Page Configuration (Must be the very first Streamlit command)
# ==============================================================================
st.set_page_config(
    page_title="YOLO Object Detection App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. Dependency Management & Diagnostics
# ==============================================================================
dependency_errors = []

# OpenCV check
try:
    import cv2
except Exception as e:
    cv2 = None
    dependency_errors.append(
        f"**OpenCV (`cv2`) failed to import**: `{e}`\n\n"
        "To fix, run:\n```bash\npython -m pip install opencv-python\n```"
    )

# Ultralytics check
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    dependency_errors.append(
        f"**Ultralytics failed to import**: `{e}`\n\n"
        "To fix, run:\n```bash\npython -m pip install ultralytics\n```"
    )

# WebRTC and PyAV check (used for webcam mode)
webrtc_available = False
webrtc_error_message = None
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    webrtc_available = True
except Exception as e:
    webrtc_available = False
    webrtc_error_message = (
        f"**WebRTC / PyAV module not available**: `{e}`\n\n"
        "To enable webcam streaming, install the required packages:\n"
        "```bash\npython -m pip install streamlit-webrtc av\n```"
    )

# ==============================================================================
# 3. Model Loading with Streamlit Resource Caching
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "trained_model.pt"

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path_str: str):
    """Loads and caches the YOLO detection model to prevent reloading on reruns."""
    path = Path(model_path_str)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights file not found at '{path}'. "
            "Please ensure 'trained_model.pt' exists in the project root."
        )
    if YOLO is None:
        raise RuntimeError("Ultralytics library is not installed or failed to import.")
    
    # Initialize YOLO with checkpoint
    loaded_model = YOLO(str(path))
    return loaded_model

# ==============================================================================
# 4. Mode 1: Image Detection
# ==============================================================================
def run_image_detection(model, conf_threshold: float):
    st.subheader("🖼️ Image Detection Mode")
    st.write("Upload an image (`JPG`, `JPEG`, or `PNG`) to detect objects.")

    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_image is not None:
        if cv2 is None:
            st.error("OpenCV is required to process images. Please install `opencv-python`.")
            return

        try:
            # Read file bytes and decode safely
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image_bgr is None:
                st.error("Failed to decode uploaded image. The file may be corrupt or in an unsupported format.")
                return

            with st.spinner("Running YOLO object detection..."):
                results = model.predict(source=image_bgr, conf=conf_threshold, verbose=False)
                result = results[0]
                annotated_bgr = result.plot()

            # Convert BGR to RGB for accurate Streamlit rendering
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Image**")
                st.image(image_rgb, use_container_width=True)

            with col2:
                st.markdown("**Detection Result**")
                st.image(annotated_rgb, use_container_width=True)

            # Detection statistics and feedback
            boxes = result.boxes
            num_detections = len(boxes) if boxes is not None else 0

            if num_detections == 0:
                st.info(
                    f"ℹ️ No objects detected above confidence threshold **{conf_threshold:.2f}**. "
                    "Try lowering the confidence threshold slider in the sidebar."
                )
            else:
                st.success(f"🎯 Successfully detected **{num_detections}** object(s)!")
                
                # Show breakdown in an expander
                with st.expander("📊 View Detection Breakdown", expanded=True):
                    detection_rows = []
                    for idx, box in enumerate(boxes, start=1):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = model.names.get(cls_id, f"Class {cls_id}")
                        detection_rows.append({
                            "#": idx,
                            "Class": cls_name,
                            "Confidence": f"{conf:.2%}"
                        })
                    st.table(detection_rows)

                # Download button for annotated image
                success, buffer = cv2.imencode(".jpg", annotated_bgr)
                if success:
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=buffer.tobytes(),
                        file_name=f"detected_{uploaded_image.name}",
                        mime="image/jpeg"
                    )

        except Exception as e:
            st.error(f"Error during image inference: {e}")

# ==============================================================================
# 5. Mode 2: Video Detection
# ==============================================================================
def run_video_detection(model, conf_threshold: float):
    st.subheader("🎥 Video Detection Mode")
    st.write("Upload a video file (`MP4`, `MOV`, `AVI`, or `MKV`) to process frame by frame.")

    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mkv"],
        help="Supported formats: MP4, MOV, AVI, MKV"
    )

    if uploaded_video is not None:
        if cv2 is None:
            st.error("OpenCV is required to process videos. Please install `opencv-python`.")
            return

        # Write uploaded video bytes to a secure temporary file
        temp_video = None
        temp_path = None
        cap = None
        try:
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_video.write(uploaded_video.read())
            temp_video.flush()
            temp_path = temp_video.name
            temp_video.close()

            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                st.error("Could not open the video file. It may be corrupt or encoded with an unsupported codec.")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            
            st.info(f"Video loaded: ~{total_frames} frames @ {fps:.1f} FPS.")

            start_col, stop_col = st.columns([1, 1])
            stop_processing = stop_col.button("🛑 Stop Video Processing", key="stop_video_btn")

            progress_bar = st.progress(0)
            status_text = st.empty()
            video_placeholder = st.empty()

            processed_count = 0
            start_time = time.time()

            while cap.isOpened():
                if stop_processing:
                    st.warning("Video processing stopped by user.")
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                processed_count += 1
                results = model.predict(source=frame, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()

                # Convert BGR to RGB for clean rendering in Streamlit
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(annotated_rgb, use_container_width=True)

                if total_frames > 0:
                    progress_val = min(processed_count / total_frames, 1.0)
                    progress_bar.progress(progress_val)
                    status_text.text(f"Processing frame {processed_count}/{total_frames} ({(processed_count / total_frames):.1%})")
                else:
                    status_text.text(f"Processed {processed_count} frames...")

            elapsed = max(time.time() - start_time, 0.001)
            processing_fps = processed_count / elapsed
            st.success(f"Processing complete: {processed_count} frames in {elapsed:.1f}s ({processing_fps:.1f} FPS).")

        except Exception as e:
            st.error(f"Error during video processing: {e}")
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

# ==============================================================================
# 6. Mode 3: Webcam Live Detection
# ==============================================================================
def run_webcam_detection(model, conf_threshold: float):
    st.subheader("📹 Webcam Live Detection Mode")
    
    st.markdown(
        """
        > [!NOTE]
        > **Webcam Streaming:** Uses browser-based WebRTC. When you click **START**, 
        > your browser will prompt you to grant camera permissions.
        >
        > *If running on a remote cloud server (e.g. Streamlit Community Cloud), 
        > video is captured directly by your client browser and sent to the server for inference.*
        """
    )

    if not webrtc_available:
        st.error(webrtc_error_message or "WebRTC packages are not available.")
        st.info("Ensure `streamlit-webrtc` and `av` are installed in your environment.")
        return

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        """Processes each incoming WebRTC video frame through YOLO."""
        try:
            img_bgr = frame.to_ndarray(format="bgr24")
            results = model.predict(source=img_bgr, conf=conf_threshold, verbose=False)
            annotated_bgr = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")
        except Exception:
            # In case of any inference glitch, return the original frame cleanly
            return frame

    webrtc_streamer(
        key="yolo_webcam_streamer",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# ==============================================================================
# 7. Main Application
# ==============================================================================
def main():
    st.markdown(
        "<h1 style='text-align: center; color: #1E88E5;'>🎯 YOLO Object Detection App</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: #888;'>Real-time AI Object Detection powered by Ultralytics YOLO & Streamlit</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # If critical dependencies are missing, show diagnostic errors and stop gracefully
    if dependency_errors:
        st.error("### ⚠️ Missing Required Dependencies")
        for err in dependency_errors:
            st.markdown(err)
            st.divider()
        st.stop()

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration Panel")

    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score required for an object to be detected and displayed."
    )

    mode = st.sidebar.radio(
        "Select Detection Mode",
        options=["Image", "Video", "Webcam"],
        index=0,
        help="Choose whether to run inference on an uploaded image, a video, or live webcam feed."
    )

    # Load model with caching
    try:
        model = load_yolo_model(str(DEFAULT_MODEL_PATH))
    except Exception as e:
        st.error(f"❌ Failed to load YOLO model: {e}")
        st.info("Check that `trained_model.pt` exists in your project directory and has not been corrupted.")
        st.stop()

    # Sidebar model info expander
    with st.sidebar.expander("ℹ️ Model Details", expanded=False):
        num_classes = len(model.names) if hasattr(model, "names") else 0
        st.write(f"**Weights File:** `{DEFAULT_MODEL_PATH.name}`")
        st.write(f"**Task:** `{getattr(model, 'task', 'detect')}`")
        st.write(f"**Classes Count:** `{num_classes}`")
        if num_classes > 0:
            st.caption(f"Sample classes: {', '.join(list(model.names.values())[:6])}...")

    # Run selected mode
    if mode == "Image":
        run_image_detection(model, conf_threshold)
    elif mode == "Video":
        run_video_detection(model, conf_threshold)
    elif mode == "Webcam":
        run_webcam_detection(model, conf_threshold)

if __name__ == "__main__":
    main()
