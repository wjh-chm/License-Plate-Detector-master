# -*- coding: utf-8 -*-
"""
检测脚本：
1) 读取图片/目录
2) 每张图片导出一个检测 JSON
3) 可选导出可视化检测图（用于人工检查和矫正）
"""

import argparse
import json
from pathlib import Path

import cv2
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_plate, scale_coords


def parse_args():
    parser = argparse.ArgumentParser(description="Plate detection with per-image JSON + visualization")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="YOLO weights path")
    parser.add_argument("--source", type=str, default="imgs", help="Image file or directory")
    parser.add_argument("--img-size", type=int, default=800, help="Inference image size")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--save-dir", type=str, default="runs/plate_detect", help="Output directory")
    parser.add_argument("--save-vis", action="store_true", help="Save visualized detection images")
    parser.add_argument("--json-dir", type=str, default="", help="Directory to save per-image json files")
    return parser.parse_args()


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


def detect_single(model, device, image, img_size, conf_thres, iou_thres):
    img0 = image.copy()
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
            x1, y1, x2, y2 = [int(v.item()) for v in row[:4]]
            conf = float(row[4].item())
            landmarks = [int(v.item()) for v in row[5:13]]
            cls_id = int(row[13].item())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "landmarks": landmarks,
                "class_id": cls_id,
            })

    return detections


def draw_detections(image, detections):
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{det['conf']:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    json_dir = Path(args.json_dir) if args.json_dir else (save_dir / "detections")
    json_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model = load_detector(args.weights, device)

    image_paths = get_image_paths(args.source)
    index_records = []

    print(f"Detector device: {device}")
    print(f"Images: {len(image_paths)}")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        detections = detect_single(model, device, image, args.img_size, args.conf_thres, args.iou_thres)
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
            "detections": detections,
        }

        out_json = unique_json_path(json_dir, image_path.stem)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        index_records.append({
            "image": str(image_path.resolve()),
            "json": str(out_json.resolve()),
            "visualization": out_vis,
            "detections": len(detections),
        })

        print(f"Processed: {image_path} | dets={len(detections)} | json={out_json.name}")

    index_path = save_dir / "detections_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_records, f, ensure_ascii=False, indent=2)

    print(f"Per-image json dir: {json_dir}")
    print(f"Index json: {index_path}")


if __name__ == "__main__":
    main()
