import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detect_plate_tiled import detect_single, load_detector, resolve_device


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Mine hard-negative tiles from an existing tiled training set.")
    parser.add_argument("--weights", type=str, required=True, help="Model weights used for mining")
    parser.add_argument("--images", type=str, required=True, help="Tiled train images directory")
    parser.add_argument("--labels", type=str, required=True, help="Tiled train labels directory")
    parser.add_argument("--img-size", type=int, default=800, help="Inference image size")
    parser.add_argument("--conf-thres", type=float, default=0.05, help="Confidence threshold for mining")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="Per-tile NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    parser.add_argument("--max-hard-negatives", type=int, default=200, help="Maximum number of mined tiles to keep")
    parser.add_argument("--base-train-txt", type=str, required=True, help="Base tiled train txt to extend")
    parser.add_argument("--output-hardneg-txt", type=str, required=True, help="Output txt for mined hard-negative tiles")
    parser.add_argument("--output-mixed-train-txt", type=str, required=True, help="Output txt for mixed train list")
    parser.add_argument("--output-json", type=str, required=True, help="Output summary JSON path")
    parser.add_argument("--repeat-hard-negatives", type=int, default=3, help="Extra repeats per mined hard-negative tile")
    return parser.parse_args()


def label_is_empty(label_path):
    return (not label_path.exists()) or label_path.stat().st_size == 0


def load_base_train_paths(base_train_txt):
    return [line.strip() for line in Path(base_train_txt).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    args = parse_args()
    images_root = Path(args.images)
    labels_root = Path(args.labels)
    output_hardneg_txt = Path(args.output_hardneg_txt)
    output_mixed_train_txt = Path(args.output_mixed_train_txt)
    output_json = Path(args.output_json)

    device = resolve_device(args.device)
    model = load_detector(args.weights, device)

    candidates = []
    image_paths = sorted([p for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES])
    empty_tiles = 0

    for image_path in image_paths:
        label_path = (labels_root / image_path.relative_to(images_root)).with_suffix(".txt")
        if not label_is_empty(label_path):
            continue

        empty_tiles += 1
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        detections = detect_single(model, device, image, args.img_size, args.conf_thres, args.iou_thres)
        if detections:
            candidates.append(
                {
                    "image": str(image_path.resolve()),
                    "label": str(label_path.resolve()),
                    "detections": len(detections),
                    "max_conf": max(det["conf"] for det in detections),
                }
            )

    candidates.sort(key=lambda item: (item["max_conf"], item["detections"]), reverse=True)
    selected = candidates[: args.max_hard_negatives]
    selected_paths = [item["image"] for item in selected]

    base_paths = load_base_train_paths(args.base_train_txt)
    mixed_paths = list(base_paths)
    for path in selected_paths:
        mixed_paths.extend([path] * args.repeat_hard_negatives)

    output_hardneg_txt.parent.mkdir(parents=True, exist_ok=True)
    output_mixed_train_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    output_hardneg_txt.write_text("\n".join(selected_paths) + ("\n" if selected_paths else ""), encoding="utf-8")
    output_mixed_train_txt.write_text("\n".join(mixed_paths) + ("\n" if mixed_paths else ""), encoding="utf-8")

    summary = {
        "weights": args.weights,
        "images_scanned": len(image_paths),
        "empty_tiles_scanned": empty_tiles,
        "hard_negative_candidates": len(candidates),
        "hard_negatives_selected": len(selected),
        "repeat_hard_negatives": args.repeat_hard_negatives,
        "base_train_samples": len(base_paths),
        "mixed_train_samples": len(mixed_paths),
        "top_candidates": selected[:20],
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved hard negatives to {output_hardneg_txt}")
    print(f"Saved mixed train list to {output_mixed_train_txt}")
    print(f"Saved summary to {output_json}")


if __name__ == "__main__":
    main()
