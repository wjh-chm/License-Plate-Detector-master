import argparse
import gc
import json
from pathlib import Path

import cv2

try:
    import yaml
except ImportError:  # pragma: no cover - fallback only used in minimal environments
    yaml = None

from detect_plate_tiled import cleanup_device, detect_tiled, load_detector, normalize_args, resolve_device


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
cv2.setNumThreads(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tiled plate detector with IoU@0.5 precision/recall.")
    parser.add_argument("--weights", type=str, required=True, help="Model weights")
    parser.add_argument("--images", type=str, default="", help="Validation images directory, image list txt, or single image")
    parser.add_argument("--labels", type=str, default="", help="Validation labels directory")
    parser.add_argument("--data", type=str, default="", help="Dataset yaml. Uses the val split and auto-derives label paths.")
    parser.add_argument("--img-size", type=int, default=800, help="Inference image size for each tile")
    parser.add_argument("--conf-thres", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="Per-tile NMS IoU threshold")
    parser.add_argument("--merge-iou-thres", type=float, default=0.5, help="Global merge NMS IoU threshold")
    parser.add_argument("--match-iou-thres", type=float, default=0.5, help="IoU threshold for TP matching")
    parser.add_argument("--tile-size", type=int, default=0, help="Square tile size alias for tile-width and tile-height")
    parser.add_argument("--tile-width", type=int, default=192, help="Tile width")
    parser.add_argument("--tile-height", type=int, default=192, help="Tile height")
    parser.add_argument("--tile-overlap", type=int, default=-1, help="Symmetric overlap alias for overlap-x and overlap-y")
    parser.add_argument("--overlap-x", type=int, default=64, help="Horizontal overlap")
    parser.add_argument("--overlap-y", type=int, default=64, help="Vertical overlap")
    parser.add_argument(
        "--profile",
        type=str,
        default="fast_stable",
        choices=["fast_stable", "high_recall"],
        help="Deployment profile. fast_stable uses proposal_plus_sparse, high_recall uses fixed sliding windows.",
    )
    parser.add_argument(
        "--proposal-mode",
        type=str,
        default="proposal_plus_sparse",
        choices=["none", "proposal_only", "proposal_plus_sparse"],
        help="Tile generation mode",
    )
    parser.add_argument("--proposal-weights", type=str, default="", help="Optional weights for the proposal model")
    parser.add_argument("--proposal-conf", type=float, default=0.03, help="Confidence threshold for whole-image proposals")
    parser.add_argument("--proposal-expand-ratio", type=float, default=2.0, help="Region expansion ratio around proposal boxes")
    parser.add_argument("--max-tiles-per-image", type=int, default=8, help="Maximum number of proposal-driven tiles per image")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--save-json", type=str, default="", help="Optional path to save summary JSON")
    return parser.parse_args()


def require_dataset_input(args):
    if args.data or args.images:
        return
    raise ValueError("Either --data or --images must be provided.")


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
        if len(parts) >= 5:
            boxes.append(yolo_row_to_xyxy(parts, width, height))
    return boxes


def load_yaml_file(path):
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)

    data = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("'\"")
    return data


def resolve_data_value(value, config_dir):
    if isinstance(value, list):
        return [resolve_data_value(item, config_dir) for item in value]
    path = Path(value)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


def collect_image_paths(source):
    src = Path(source)
    if src.is_file() and src.suffix.lower() != ".txt":
        return [src.resolve()], src
    if src.is_file() and src.suffix.lower() == ".txt":
        image_paths = []
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            path = Path(line)
            if not path.is_absolute():
                path = (src.parent / path).resolve()
            image_paths.append(path)
        return image_paths, src
    if src.is_dir():
        return sorted([p.resolve() for p in src.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]), src.resolve()
    raise FileNotFoundError(f"Input source not found: {source}")


def infer_label_path_from_image(image_path):
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def resolve_dataset(args):
    if args.data:
        data_path = Path(args.data).resolve()
        data_cfg = load_yaml_file(data_path)
        val_entry = data_cfg.get("val")
        if not val_entry:
            raise ValueError(f"No 'val' entry found in data config: {data_path}")
        val_source = resolve_data_value(val_entry, data_path.parent)
        if isinstance(val_source, list):
            image_paths = []
            for item in val_source:
                paths, _ = collect_image_paths(item)
                image_paths.extend(paths)
            image_paths = sorted(image_paths)
            images_root = None
        else:
            image_paths, images_root = collect_image_paths(val_source)
        return image_paths, images_root, None, str(data_path)

    image_paths, images_root = collect_image_paths(args.images)
    labels_root = Path(args.labels).resolve() if args.labels else None
    return image_paths, images_root, labels_root, str(Path(args.images).resolve())


def build_label_path(image_path, images_root, labels_root):
    if labels_root is not None and images_root is not None and images_root.is_dir():
        return (labels_root / image_path.relative_to(images_root)).with_suffix(".txt")
    if labels_root is not None:
        return (labels_root / image_path.stem).with_suffix(".txt")
    return infer_label_path_from_image(image_path)


def get_scene_name(image_path, images_root):
    if images_root is None or not images_root.is_dir():
        return image_path.parent.name
    try:
        relative = image_path.relative_to(images_root)
    except ValueError:
        return image_path.parent.name
    if len(relative.parts) > 1:
        return relative.parts[0]
    return image_path.parent.name


def make_metric_bucket():
    return {"images": 0, "tp": 0, "fp": 0, "fn": 0, "tiles": 0, "latency_ms": 0.0}


def finalize_metric_bucket(bucket):
    tp = bucket["tp"]
    fp = bucket["fp"]
    fn = bucket["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    images = bucket["images"]
    return {
        "images": images,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "avg_tiles_per_image": bucket["tiles"] / images if images else 0.0,
        "avg_latency_ms": bucket["latency_ms"] / images if images else 0.0,
    }


def main():
    args = normalize_args(parse_args())
    require_dataset_input(args)

    image_paths, images_root, labels_root, dataset_name = resolve_dataset(args)
    device = resolve_device(args.device)
    model = load_detector(args.weights, device)
    proposal_model = model
    if args.proposal_mode != "none" and args.proposal_weights and args.proposal_weights != args.weights:
        proposal_model = load_detector(args.proposal_weights, device)

    tp = 0
    fp = 0
    fn = 0
    total_tiles = 0
    total_latency_ms = 0.0
    max_tiles_per_image = 0
    details = []
    subset_buckets = {}

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        label_path = build_label_path(image_path, images_root, labels_root)
        gt_boxes = load_gt_boxes(label_path, width, height)

        detections, tiles, meta = detect_tiled(
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
            proposal_mode=args.proposal_mode,
            proposal_model=proposal_model,
            proposal_conf=args.proposal_conf,
            proposal_expand_ratio=args.proposal_expand_ratio,
            max_tiles_per_image=args.max_tiles_per_image,
            return_meta=True,
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
        tile_count = len(tiles)
        latency_ms = meta["total_latency_ms"]
        total_tiles += tile_count
        total_latency_ms += latency_ms
        max_tiles_per_image = max(max_tiles_per_image, tile_count)

        scene_name = get_scene_name(image_path, images_root)
        subset_bucket = subset_buckets.setdefault(scene_name, make_metric_bucket())
        subset_bucket["images"] += 1
        subset_bucket["tp"] += image_tp
        subset_bucket["fp"] += image_fp
        subset_bucket["fn"] += image_fn
        subset_bucket["tiles"] += tile_count
        subset_bucket["latency_ms"] += latency_ms

        details.append(
            {
                "image": str(image_path),
                "scene": scene_name,
                "label": str(label_path),
                "gt": len(gt_boxes),
                "pred": len(preds),
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
                "tile_count": tile_count,
                "latency_ms": latency_ms,
                "proposal_count": meta["proposal_count"],
                "tile_count_before_limit": meta["tile_count_before_limit"],
                "tile_source_breakdown": meta["tile_source_breakdown"],
            }
        )
        del image, detections, preds, gt_boxes
        gc.collect()
        cleanup_device(device)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    subset_metrics = {name: finalize_metric_bucket(bucket) for name, bucket in sorted(subset_buckets.items())}
    summary = {
        "weights": args.weights,
        "dataset": dataset_name,
        "images": len(details),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "img_size": args.img_size,
        "tile_width": args.tile_width,
        "tile_height": args.tile_height,
        "overlap_x": args.overlap_x,
        "overlap_y": args.overlap_y,
        "proposal_mode": args.proposal_mode,
        "profile": args.profile,
        "proposal_weights": args.proposal_weights or args.weights,
        "proposal_conf": args.proposal_conf,
        "proposal_expand_ratio": args.proposal_expand_ratio,
        "max_tiles_per_image_limit": args.max_tiles_per_image,
        "avg_tiles_per_image": total_tiles / len(details) if details else 0.0,
        "max_tiles_per_image": max_tiles_per_image,
        "avg_latency_ms": total_latency_ms / len(details) if details else 0.0,
        "subset_metrics": subset_metrics,
        "details": details,
    }

    print(json.dumps({k: v for k, v in summary.items() if k not in {"details", "subset_metrics"}}, indent=2))
    if subset_metrics:
        print(json.dumps({"subset_metrics": subset_metrics}, indent=2))
    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to {save_path}")


if __name__ == "__main__":
    main()
