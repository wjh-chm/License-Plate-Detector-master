import argparse
import gc
import json
from pathlib import Path

import cv2

from detect_plate_tiled import cleanup_device, load_detector, resolve_device, detect_tiled


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
cv2.setNumThreads(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tiled plate detector with IoU@0.5 precision/recall.")
    parser.add_argument("--weights", type=str, required=True, help="Model weights")
    parser.add_argument("--images", type=str, required=True, help="Validation images directory")
    parser.add_argument("--labels", type=str, required=True, help="Validation labels directory")
    parser.add_argument("--img-size", type=int, default=800, help="Inference image size for each tile")
    parser.add_argument("--conf-thres", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="Per-tile NMS IoU threshold")
    parser.add_argument("--merge-iou-thres", type=float, default=0.5, help="Global merge NMS IoU threshold")
    parser.add_argument("--match-iou-thres", type=float, default=0.5, help="IoU threshold for TP matching")
    parser.add_argument("--tile-width", type=int, default=192, help="Tile width")
    parser.add_argument("--tile-height", type=int, default=192, help="Tile height")
    parser.add_argument("--overlap-x", type=int, default=64, help="Horizontal overlap")
    parser.add_argument("--overlap-y", type=int, default=64, help="Vertical overlap")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--save-json", type=str, default="", help="Optional path to save summary JSON")
    return parser.parse_args()


def yolo_row_to_xyxy(parts, width, height):
    cx = float(parts[1]) * width
    cy = float(parts[2]) * height
    bw = float(parts[3]) * width
    bh = float(parts[4]) * height
    return [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt_boxes(label_path, width, height):
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            boxes.append(yolo_row_to_xyxy(parts, width, height))
    return boxes


def main():
    args = parse_args()
    images_root = Path(args.images)
    labels_root = Path(args.labels)

    device = resolve_device(args.device)
    model = load_detector(args.weights, device)

    tp = 0
    fp = 0
    fn = 0
    details = []

    image_paths = sorted([p for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES])
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        label_path = (labels_root / image_path.relative_to(images_root)).with_suffix(".txt")
        gt_boxes = load_gt_boxes(label_path, width, height)

        detections, _ = detect_tiled(
            model,
            device,
            image,
            args.img_size,
            args.conf_thres,
            args.iou_thres,
            args.merge_iou_thres,
            args.tile_width,
            args.tile_height,
            args.overlap_x,
            args.overlap_y,
        )

        preds = sorted(detections, key=lambda d: d["conf"], reverse=True)
        matched = set()
        image_tp = 0
        image_fp = 0
        for pred in preds:
            pred_box = pred["bbox"]
            best_iou = 0.0
            best_idx = -1
            for idx, gt_box in enumerate(gt_boxes):
                if idx in matched:
                    continue
                iou = box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= args.match_iou_thres:
                matched.add(best_idx)
                tp += 1
                image_tp += 1
            else:
                fp += 1
                image_fp += 1

        image_fn = len(gt_boxes) - len(matched)
        fn += image_fn
        details.append(
            {
                "image": str(image_path),
                "gt": len(gt_boxes),
                "pred": len(preds),
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
            }
        )
        del image, detections, preds, gt_boxes
        gc.collect()
        cleanup_device(device)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    summary = {
        "weights": args.weights,
        "images": len(image_paths),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "tile_width": args.tile_width,
        "tile_height": args.tile_height,
        "overlap_x": args.overlap_x,
        "overlap_y": args.overlap_y,
        "details": details,
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))
    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to {save_path}")


if __name__ == "__main__":
    main()
