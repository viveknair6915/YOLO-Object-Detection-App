"""Diagnostic and environment validation script for the YOLO Object Detection App.

Verifies:
1. Active Python interpreter and version.
2. Imports and versions of all critical libraries (OpenCV, Ultralytics, PyTorch, Torchvision, NumPy, Pillow, Streamlit, streamlit-webrtc, av).
3. Existence and integrity of trained_model.pt.
4. Model initialization via Ultralytics YOLO.
5. End-to-end inference on synthetic test image and existing screenshot assets.
6. Synthetic video read and inference pipeline test.
"""

import sys
import os
from pathlib import Path
import tempfile

def print_section(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def main():
    all_passed = True
    print_section("1. Python Environment Information")
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Platform          : {sys.platform}")
    print(f"Current Directory : {os.getcwd()}")

    print_section("2. Package Imports and Version Verification")
    packages = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV (cv2)"),
        ("PIL", "Pillow (PIL)"),
        ("torch", "PyTorch"),
        ("torchvision", "Torchvision"),
        ("ultralytics", "Ultralytics"),
        ("streamlit", "Streamlit"),
        ("streamlit_webrtc", "Streamlit-WebRTC"),
        ("av", "PyAV (av)"),
    ]

    imported_modules = {}
    for mod_name, display_name in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"[PASS] {display_name:<20}: {ver}")
            imported_modules[mod_name] = mod
        except Exception as e:
            print(f"[FAIL] {display_name:<20}: FAILED to import ({e})")
            all_passed = False

    # Check for OpenCV distribution conflicts
    cv2_mod = imported_modules.get("cv2")
    if cv2_mod:
        cv2_file = getattr(cv2_mod, "__file__", "unknown")
        print(f"       OpenCV location: {cv2_file}")

    print_section("3. Model Weights Validation")
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "trained_model.pt"

    if not model_path.exists():
        print(f"[FAIL] trained_model.pt not found at: {model_path}")
        all_passed = False
        return 1
    
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"[PASS] Model file found: {model_path} ({file_size_mb:.2f} MB)")

    print_section("4. Model Initialization & Metadata")
    yolo_model = None
    if "ultralytics" in imported_modules:
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(str(model_path))
            num_classes = len(yolo_model.names) if hasattr(yolo_model, "names") else 0
            task = getattr(yolo_model, "task", "unknown")
            print(f"[PASS] Model loaded successfully!")
            print(f"       Architecture/Task: {task}")
            print(f"       Class Count      : {num_classes}")
            if num_classes > 0:
                sample_classes = list(yolo_model.names.values())[:5]
                print(f"       Sample Classes   : {sample_classes} ...")
        except Exception as e:
            print(f"[FAIL] Failed to load model weights with YOLO: {e}")
            all_passed = False
    else:
        print("[SKIP] Skipping model load test because ultralytics import failed.")
        all_passed = False

    print_section("5. Image Inference Smoke Test")
    if yolo_model and cv2_mod:
        np_mod = imported_modules.get("numpy")
        try:
            # 5a. Synthetic blank image test
            synthetic_img = np_mod.zeros((320, 320, 3), dtype=np_mod.uint8)
            res_synth = yolo_model.predict(source=synthetic_img, conf=0.25, verbose=False)
            boxes_synth = len(res_synth[0].boxes)
            print(f"[PASS] Synthetic image inference: OK (Detections: {boxes_synth})")

            # 5b. Real asset image test if exists
            test_asset = base_dir / "Screenshots" / "image2.jpeg"
            if test_asset.exists():
                asset_img = cv2_mod.imread(str(test_asset))
                res_asset = yolo_model.predict(source=asset_img, conf=0.1, verbose=False)
                boxes_asset = len(res_asset[0].boxes)
                print(f"[PASS] Real asset test ({test_asset.name}): OK (Detections: {boxes_asset})")
                for box in res_asset[0].boxes[:3]:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = yolo_model.names.get(cls_id, str(cls_id))
                    print(f"       Detected: {cls_name} (conf: {conf:.2%})")
            else:
                print(f"[INFO] Asset {test_asset} not found, skipped.")
        except Exception as e:
            print(f"[FAIL] Image inference test failed: {e}")
            all_passed = False

    print_section("6. Video Pipeline Smoke Test")
    if yolo_model and cv2_mod and imported_modules.get("numpy"):
        np_mod = imported_modules.get("numpy")
        temp_video = None
        try:
            # Create a small 5-frame synthetic video
            temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_video_path = temp_video.name
            temp_video.close()

            fourcc = cv2_mod.VideoWriter_fourcc(*"mp4v")
            writer = cv2_mod.VideoWriter(temp_video_path, fourcc, 10.0, (160, 120))
            for i in range(5):
                frame = np_mod.full((120, 160, 3), fill_value=(i * 40, i * 30, i * 20), dtype=np_mod.uint8)
                writer.write(frame)
            writer.release()

            # Now read back and run inference
            cap = cv2_mod.VideoCapture(temp_video_path)
            frames_processed = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                _ = yolo_model.predict(source=frame, conf=0.5, verbose=False)
                frames_processed += 1
            cap.release()

            print(f"[PASS] Synthetic video pipeline test: OK (Processed {frames_processed} frames)")
        except Exception as e:
            print(f"[FAIL] Video processing pipeline test failed: {e}")
            all_passed = False
        finally:
            if temp_video and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except Exception:
                    pass

    print_section("Summary")
    if all_passed:
        print("[SUCCESS] All environment, model, and inference diagnostics PASSED!")
        return 0
    else:
        print("[ERROR] One or more diagnostic checks failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
