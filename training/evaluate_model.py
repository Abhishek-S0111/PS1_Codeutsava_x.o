"""
Evaluate Model — run mAP / precision / recall on test set.

Usage:
    python training/evaluate_model.py \
        --model-path models/defect_detector.pt \
        --test-data data/training_dataset/images/test \
        --test-labels data/training_dataset/labels/test
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Evaluate defect detection model")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-yaml", type=str, default="", help="dataset.yaml path")
    parser.add_argument("--test-data", type=str, default="")
    parser.add_argument("--test-labels", type=str, default="")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        sys.exit(1)

    model = YOLO(str(model_path))

    if args.data_yaml:
        data_yaml = args.data_yaml
    else:
        # Try to find dataset.yaml
        candidates = [
            ROOT / "data" / "training_dataset" / "dataset.yaml",
            Path(args.test_data).parent.parent / "dataset.yaml" if args.test_data else None,
        ]
        data_yaml = None
        for c in candidates:
            if c and c.exists():
                data_yaml = str(c)
                break

    if data_yaml:
        print(f"Evaluating on: {data_yaml}")
        results = model.val(
            data=data_yaml,
            split="test",
            conf=args.conf,
            iou=args.iou,
            verbose=True,
        )
        print("\n=== Evaluation Results ===")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall: {results.box.mr:.4f}")
    else:
        print("No dataset.yaml found. Running inference on test images instead.")
        if args.test_data and Path(args.test_data).exists():
            import cv2
            test_dir = Path(args.test_data)
            images = list(test_dir.glob("*.*"))
            print(f"Running on {len(images)} test images ...")
            for img_path in images[:20]:
                results = model(str(img_path), conf=args.conf, verbose=False)
                n_dets = sum(len(r.boxes) for r in results if r.boxes is not None)
                print(f"  {img_path.name}: {n_dets} detection(s)")
        else:
            print("Provide --data-yaml or --test-data to evaluate.")


if __name__ == "__main__":
    main()
