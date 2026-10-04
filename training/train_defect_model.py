"""
Train Defect Detection Model — YOLOv8 training on converted dataset.

Usage:
    python training/train_defect_model.py \
        --data-dir data/training_dataset \
        --epochs 50 \
        --batch-size 16 \
        --output-dir models/
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 defect detector")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset dir with dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--model-base", type=str, default="yolov8n.pt", help="Base model")
    parser.add_argument("--output-dir", type=str, default="models/")
    parser.add_argument("--device", type=str, default="", help="cpu or cuda device")
    args = parser.parse_args()

    data_yaml = Path(args.data_dir) / "dataset.yaml"
    if not data_yaml.exists():
        print(f"ERROR: dataset.yaml not found in {args.data_dir}")
        print("Run dataset_converter.py first to generate the dataset.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"Loading base model: {args.model_base}")
    model = YOLO(args.model_base)

    print(f"Training on: {data_yaml}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, Img: {args.img_size}")

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        project=args.output_dir,
        name="defect_detector",
        device=args.device or None,
        save=True,
        plots=True,
        verbose=True,
    )

    # Copy best model
    output_path = Path(args.output_dir)
    best_model = output_path / "defect_detector" / "weights" / "best.pt"
    if best_model.exists():
        import shutil
        dest = output_path / "defect_detector.pt"
        shutil.copy2(str(best_model), str(dest))
        print(f"\n[OK] Best model saved to: {dest}")
        print(f"Copy this to models/ and restart the app to use it.")
    else:
        print(f"\n[!] best.pt not found at expected path. Check {output_path}/defect_detector/")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
