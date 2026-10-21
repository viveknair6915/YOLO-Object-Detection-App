"""Unit and integration test suite for YOLO Object Detection App."""

import unittest
import os
import tempfile
from pathlib import Path
import numpy as np

class TestYOLOApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent
        cls.model_path = cls.base_dir / "trained_model.pt"

    def test_01_dependency_imports(self):
        """Verifies that all core and optional dependencies import without error."""
        import cv2
        import ultralytics
        from ultralytics import YOLO
        import torch
        import torchvision
        import PIL
        import streamlit
        import streamlit_webrtc
        import av

        self.assertIsNotNone(cv2.__version__)
        self.assertIsNotNone(ultralytics.__version__)
        self.assertIsNotNone(torch.__version__)

    def test_02_model_checkpoint_exists(self):
        """Verifies that trained_model.pt exists and is valid size (> 1MB)."""
        self.assertTrue(self.model_path.is_file(), f"Model file not found at {self.model_path}")
        size_bytes = self.model_path.stat().st_size
        self.assertGreater(size_bytes, 1024 * 1024, "Model file is unexpectedly small.")

    def test_03_model_loading(self):
        """Verifies that trained_model.pt can be initialized with Ultralytics YOLO."""
        from ultralytics import YOLO
        model = YOLO(str(self.model_path))
        self.assertIsNotNone(model)
        self.assertEqual(model.task, "detect")
        self.assertEqual(len(model.names), 80, "Expected 80 classes for COCO trained model.")

    def test_04_synthetic_image_inference(self):
        """Verifies end-to-end inference on a synthetic image."""
        from ultralytics import YOLO
        model = YOLO(str(self.model_path))
        synthetic_img = np.zeros((320, 320, 3), dtype=np.uint8)
        results = model.predict(source=synthetic_img, conf=0.25, verbose=False)
        self.assertEqual(len(results), 1)
        annotated = results[0].plot()
        self.assertEqual(annotated.shape, (320, 320, 3))
        self.assertEqual(annotated.dtype, np.uint8)

    def test_05_real_asset_inference(self):
        """Verifies inference on existing screenshot asset."""
        import cv2
        from ultralytics import YOLO
        model = YOLO(str(self.model_path))
        asset_path = self.base_dir / "Screenshots" / "image2.jpeg"
        if not asset_path.exists():
            self.skipTest(f"{asset_path} not found.")

        img = cv2.imread(str(asset_path))
        self.assertIsNotNone(img, "Failed to read image2.jpeg with OpenCV")
        results = model.predict(source=img, conf=0.1, verbose=False)
        self.assertGreater(len(results[0].boxes), 0, "Expected at least one detection on image2.jpeg at conf=0.1")

    def test_06_video_processing_pipeline(self):
        """Verifies video creation, frame-by-frame decoding, inference, and cleanup."""
        import cv2
        from ultralytics import YOLO
        model = YOLO(str(self.model_path))

        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_path = temp_video.name
        temp_video.close()

        try:
            # Create a 5-frame synthetic video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_path, fourcc, 10.0, (160, 120))
            for i in range(5):
                frame = np.full((120, 160, 3), fill_value=(i * 30, i * 20, i * 10), dtype=np.uint8)
                writer.write(frame)
            writer.release()

            # Process frame by frame
            cap = cv2.VideoCapture(temp_path)
            self.assertTrue(cap.isOpened(), "Could not open synthetic test video.")
            frames_read = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(source=frame, conf=0.5, verbose=False)
                annotated = results[0].plot()
                self.assertEqual(annotated.shape, frame.shape)
                frames_read += 1
            cap.release()
            self.assertEqual(frames_read, 5)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == "__main__":
    unittest.main()
