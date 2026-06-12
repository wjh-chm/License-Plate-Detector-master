# -*- coding: utf-8 -*-
"""
Sliding-window plate detection for small targets.
Runs plate detection on overlapping image tiles, maps detections back to the
full image, and applies a final global NMS.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import cv2
import torch
import torchvision

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_plate, scale_coords


cv2.setNumThreads(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Plate detection with tiled inference")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="YOLO weights path")
    parser.add_argument("--source", type=str, default="test", help="Image file or directory")
    parser.add_argument("--img-size", type=int, default=800, help="Inference image size for each tile")
    parser.add_argument("--conf-thres", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="Per-tile NMS IoU threshold")
    parser.add_argument("--merge-iou-thres", type=float, default=0.3, help="Global merge NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--save-dir", type=str, default="runs/plate_detect_tiled", help="Output directory")
    parser.add_argument("--save-vis", action="store_true", help="Save visualized detection images")
    parser.add_argument("--json-dir", type=str, default="", help="Directory to save per-image json files")
    parser.add_argument("--tile-size", type=int, default=0, help="Square tile size alias for tile-width and tile-height")
    parser.add_argument("--tile-width", type=int, default=192, help="Tile width in pixels")
    parser.add_argument("--tile-height", type=int, default=192, help="Tile height in pixels")
    parser.add_argument("--tile-overlap", type=int, default=-1, help="Symmetric overlap alias for overlap-x and overlap-y")
    parser.add_argument("--overlap-x", type=int, default=64, help="Horizontal overlap in pixels")
    parser.add_argument("--overlap-y", type=int, default=64, help="Vertical overlap in pixels")
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
    return parser.parse_args()


def normalize_args(args):
    if getattr(args, "tile_size", 0):
        args.tile_width = args.tile_size
        args.tile_height = args.tile_size
    if getattr(args, "tile_overlap", -1) >= 0:
        args.overlap_x = args.tile_overlap
        args.overlap_y = args.tile_overlap
    if getattr(args, "max_tiles_per_image", 0) == 0:
        args.max_tiles_per_image = -1
    if getattr(args, "profile", "") == "high_recall":
        args.proposal_mode = "none"
    elif getattr(args, "profile", "") == "fast_stable" and not getattr(args, "proposal_mode", ""):
        args.proposal_mode = "proposal_plus_sparse"
    return args


def resolve_device(device_arg):
    if device_arg.lower().startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, fallback to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def get_image_paths(source):
    src = Path(source)
    if src.is_file():
        return [src]
    if src.is_dir():
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted([p for p in src.rglob("*") if p.suffix.lower() in suffixes])
    raise FileNotFoundError(f"Source not found: {source}")


def sliding_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]

    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def generate_tile_boxes(width, height, tile_width, tile_height, overlap_x, overlap_y):
    stride_x = max(1, tile_width - overlap_x)
    stride_y = max(1, tile_height - overlap_y)
    x_starts = sliding_starts(width, tile_width, stride_x)
    y_starts = sliding_starts(height, tile_height, stride_y)

    tiles = []
    for top in y_starts:
        for left in x_starts:
            tiles.append((left, top, min(left + tile_width, width), min(top + tile_height, height)))
    return tiles


def generate_sparse_tile_boxes(width, height, tile_width, tile_height, overlap_x, overlap_y):
    stride_x = max(1, (tile_width - overlap_x) * 2)
    stride_y = max(1, (tile_height - overlap_y) * 2)
    x_starts = sliding_starts(width, tile_width, stride_x)
    y_starts = sliding_starts(height, tile_height, stride_y)

    tiles = []
    for top in y_starts:
        for left in x_starts:
            tiles.append((left, top, min(left + tile_width, width), min(top + tile_height, height)))
    return tiles


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]
    coords[:, [1, 3, 5, 7]] -= pad[1]
    coords[:, :8] /= gain
    coords[:, [0, 2, 4, 6]].clamp_(0, img0_shape[1])
    coords[:, [1, 3, 5, 7]].clamp_(0, img0_shape[0])
    return coords


def load_detector(weights, device):
    model = attempt_load(weights, map_location=device)
    model.eval()
    return model


def cleanup_device(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def detect_single(model, device, image, img_size, conf_thres, iou_thres):
    img0 = image
    h0, w0 = img0.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=int(model.stride.max()))
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()

    tensor = torch.from_numpy(img).to(device).float() / 255.0
    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(tensor)[0]
        pred = non_max_suppression_plate(pred, conf_thres, iou_thres)

    detections = []
    for det in pred:
        if not len(det):
            continue
        det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], image.shape).round()
        det[:, 5:13] = scale_coords_landmarks(tensor.shape[2:], det[:, 5:13], image.shape).round()

        for row in det:
            detections.append(
                {
                    "bbox": [int(v.item()) for v in row[:4]],
                    "conf": float(row[4].item()),
                    "landmarks": [int(v.item()) for v in row[5:13]],
                    "class_id": int(row[13].item()),
                }
            )
    del tensor, pred
    return detections


def shift_detection(det, offset_x, offset_y):
    bbox = det["bbox"]
    landmarks = det["landmarks"]
    return {
        "bbox": [bbox[0] + offset_x, bbox[1] + offset_y, bbox[2] + offset_x, bbox[3] + offset_y],
        "conf": det["conf"],
        "landmarks": [
            landmarks[0] + offset_x,
            landmarks[1] + offset_y,
            landmarks[2] + offset_x,
            landmarks[3] + offset_y,
            landmarks[4] + offset_x,
            landmarks[5] + offset_y,
            landmarks[6] + offset_x,
            landmarks[7] + offset_y,
        ],
        "class_id": det["class_id"],
    }


def global_nms(detections, iou_thres):
    if not detections:
        return []

    boxes = torch.tensor([det["bbox"] for det in detections], dtype=torch.float32)
    scores = torch.tensor([det["conf"] for det in detections], dtype=torch.float32)
    keep = torchvision.ops.nms(boxes, scores, iou_thres)
    return [detections[i] for i in keep.tolist()]


def expand_box(box, width, height, expand_ratio):
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(1.0, (x2 - x1) * expand_ratio)
    bh = max(1.0, (y2 - y1) * expand_ratio)
    left = max(0, int(round(cx - bw / 2.0)))
    top = max(0, int(round(cy - bh / 2.0)))
    right = min(width, int(round(cx + bw / 2.0)))
    bottom = min(height, int(round(cy + bh / 2.0)))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


def add_tile_record(tile_map, tile, score, source):
    box = tuple(int(v) for v in tile)
    if box[2] <= box[0] or box[3] <= box[1]:
        return
    record = tile_map.get(box)
    if record is None:
        tile_map[box] = {"box": box, "score": float(score), "sources": {source}}
        return
    record["score"] = max(record["score"], float(score))
    record["sources"].add(source)


def boxes_intersect(box_a, box_b):
    return min(box_a[2], box_b[2]) > max(box_a[0], box_b[0]) and min(box_a[3], box_b[3]) > max(box_a[1], box_b[1])


def build_proposal_tiles(proposal_detections, candidate_tiles, width, height, expand_ratio):
    tile_map = {}
    for det in proposal_detections:
        region = expand_box(det["bbox"], width, height, expand_ratio)
        for tile in candidate_tiles:
            if boxes_intersect(tile, region):
                add_tile_record(tile_map, tile, det["conf"], "proposal")
    return tile_map


def select_tiles(
    width,
    height,
    tile_width,
    tile_height,
    overlap_x,
    overlap_y,
    proposal_mode,
    proposal_detections,
    proposal_expand_ratio,
    max_tiles_per_image,
):
    if proposal_mode == "none":
        tiles = generate_tile_boxes(width, height, tile_width, tile_height, overlap_x, overlap_y)
        meta = {
            "proposal_count": 0,
            "tile_count_before_limit": len(tiles),
            "tile_count_after_limit": len(tiles),
            "tile_source_breakdown": {"sliding": len(tiles)},
            "tiles_limited": False,
        }
        return tiles, meta

    baseline_tiles = generate_tile_boxes(width, height, tile_width, tile_height, overlap_x, overlap_y)
    tile_map = build_proposal_tiles(proposal_detections, baseline_tiles, width, height, proposal_expand_ratio)

    if proposal_mode == "proposal_plus_sparse":
        for tile in generate_sparse_tile_boxes(width, height, tile_width, tile_height, overlap_x, overlap_y):
            add_tile_record(tile_map, tile, 0.0, "sparse")

    tiles_before_limit = len(tile_map)
    tile_records = sorted(
        tile_map.values(),
        key=lambda item: (-item["score"], -(item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]), item["box"]),
    )
    if max_tiles_per_image and max_tiles_per_image > 0:
        tile_records = tile_records[:max_tiles_per_image]

    tiles = [record["box"] for record in tile_records]
    tile_source_breakdown = {"proposal": 0, "sparse": 0}
    for record in tile_records:
        for source in record["sources"]:
            tile_source_breakdown[source] = tile_source_breakdown.get(source, 0) + 1

    meta = {
        "proposal_count": len(proposal_detections),
        "tile_count_before_limit": tiles_before_limit,
        "tile_count_after_limit": len(tiles),
        "tile_source_breakdown": tile_source_breakdown,
        "tiles_limited": len(tiles) < tiles_before_limit,
    }
    return tiles, meta


def detect_tiled(
    model,
    device,
    image,
    img_size,
    conf_thres,
    iou_thres,
    merge_iou_thres,
    tile_width,
    tile_height,
    overlap_x,
    overlap_y,
    proposal_mode="none",
    proposal_model=None,
    proposal_conf=0.02,
    proposal_expand_ratio=2.0,
    max_tiles_per_image=16,
    return_meta=False,
):
    height, width = image.shape[:2]
    proposal_model = proposal_model or model
    proposal_detections = []

    t0 = time.perf_counter()
    proposal_latency_ms = 0.0
    if proposal_mode != "none":
        proposal_start = time.perf_counter()
        proposal_detections = detect_single(proposal_model, device, image, img_size, proposal_conf, iou_thres)
        proposal_latency_ms = (time.perf_counter() - proposal_start) * 1000.0

    tiles, tile_meta = select_tiles(
        width,
        height,
        tile_width,
        tile_height,
        overlap_x,
        overlap_y,
        proposal_mode,
        proposal_detections,
        proposal_expand_ratio,
        max_tiles_per_image,
    )

    merged = []
    tile_start = time.perf_counter()
    for left, top, right, bottom in tiles:
        tile = image[top:bottom, left:right]
        detections = detect_single(model, device, tile, img_size, conf_thres, iou_thres)
        for det in detections:
            merged.append(shift_detection(det, left, top))
        del detections, tile
    tile_latency_ms = (time.perf_counter() - tile_start) * 1000.0

    merged = global_nms(merged, merge_iou_thres)
    cleanup_device(device)
    meta = {
        "proposal_mode": proposal_mode,
        "proposal_latency_ms": proposal_latency_ms,
        "tile_latency_ms": tile_latency_ms,
        "total_latency_ms": (time.perf_counter() - t0) * 1000.0,
        **tile_meta,
    }
    if return_meta:
        return merged, tiles, meta
    return merged, tiles


def draw_detections(image, detections):
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{det['conf']:.2f}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        lms = det["landmarks"]
        for i in range(4):
            px, py = int(lms[2 * i]), int(lms[2 * i + 1])
            cv2.circle(vis, (px, py), 2, (0, 255, 255), -1)
    return vis


def unique_json_path(json_dir, stem):
    p = json_dir / f"{stem}.json"
    if not p.exists():
        return p
    i = 1
    while True:
        p = json_dir / f"{stem}_{i}.json"
        if not p.exists():
            return p
        i += 1


def main():
    args = normalize_args(parse_args())
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    json_dir = Path(args.json_dir) if args.json_dir else (save_dir / "detections")
    json_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model = load_detector(args.weights, device)
    proposal_model = model
    if args.proposal_mode != "none" and args.proposal_weights and args.proposal_weights != args.weights:
        proposal_model = load_detector(args.proposal_weights, device)

    image_paths = get_image_paths(args.source)
    index_records = []

    print(f"Detector device: {device}")
    print(f"Images: {len(image_paths)}")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

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
        out_vis = ""

        if args.save_vis:
            vis = draw_detections(image, detections)
            vis_path = save_dir / f"{image_path.stem}_det.jpg"
            cv2.imwrite(str(vis_path), vis)
            out_vis = str(vis_path.resolve())

        record = {
            "image": str(image_path.resolve()),
            "image_name": image_path.name,
            "visualization": out_vis,
            "tile_count": len(tiles),
            "tile_shape": [args.tile_width, args.tile_height],
            "tile_overlap": [args.overlap_x, args.overlap_y],
            "profile": args.profile,
            "proposal_mode": args.proposal_mode,
            "proposal_meta": meta,
            "detections": detections,
        }

        out_json = unique_json_path(json_dir, image_path.stem)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        index_records.append(
            {
                "image": str(image_path.resolve()),
                "json": str(out_json.resolve()),
                "visualization": out_vis,
                "tile_count": len(tiles),
                "detections": len(detections),
                "profile": args.profile,
                "proposal_mode": args.proposal_mode,
                "latency_ms": meta["total_latency_ms"],
            }
        )

        print(
            f"Processed: {image_path} | profile={args.profile} | mode={args.proposal_mode} | proposals={meta['proposal_count']} "
            f"| tiles={len(tiles)} | dets={len(detections)} | latency_ms={meta['total_latency_ms']:.1f} | json={out_json.name}"
        )

    index_path = save_dir / "detections_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_records, f, ensure_ascii=False, indent=2)

    print(f"Per-image json dir: {json_dir}")
    print(f"Index json: {index_path}")


if __name__ == "__main__":
    main()
