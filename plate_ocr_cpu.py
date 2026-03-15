# -*- coding: utf-8 -*-
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_plate, scale_coords


PLATE_CHARS_REGEX = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]")


def parse_args():
    parser = argparse.ArgumentParser(description="Detect license plates and run OCR on CPU.")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Detector weights path")
    parser.add_argument("--source", type=str, default="imgs", help="Image path or directory")
    parser.add_argument("--img-size", type=int, default=800, help="Inference image size")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--ocr-engine", type=str, default="auto", choices=["auto", "tesseract", "easyocr"],
                        help="OCR backend")
    parser.add_argument("--tesseract-cmd", type=str, default="", help="Optional tesseract executable path")
    parser.add_argument("--tesseract-lang", type=str, default="eng", help="Tesseract language, e.g. eng or chi_sim+eng")
    parser.add_argument("--easyocr-lang", type=str, default="en", help="EasyOCR language code, e.g. en or ch_sim")
    parser.add_argument("--save-dir", type=str, default="runs/plate_ocr_cpu", help="Output directory")
    return parser.parse_args()


def load_detector(weights_path):
    device = torch.device("cpu")
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    return model, device


def get_image_paths(source):
    src = Path(source)
    if src.is_file():
        return [src]
    if src.is_dir():
        valid_suffix = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted([p for p in src.rglob("*") if p.suffix.lower() in valid_suffix])
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


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def rectify_plate(image, landmarks):
    pts = np.array(landmarks, dtype=np.float32).reshape(4, 2)
    rect = order_points(pts)
    dst = np.array([[0, 0], [239, 0], [239, 79], [0, 79]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (240, 80))


def preprocess_for_ocr(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    denoise = cv2.bilateralFilter(up, 7, 50, 50)
    _, th = cv2.threshold(denoise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [up, th]


def normalize_plate_text(text):
    cleaned = text.strip().upper().replace(" ", "")
    return PLATE_CHARS_REGEX.sub("", cleaned)


def init_tesseract(lang, tesseract_cmd):
    try:
        import pytesseract
    except ImportError:
        return None, "pytesseract is not installed"

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    return {"module": pytesseract, "lang": lang}, None


def run_tesseract(ocr_state, image):
    pytesseract = ocr_state["module"]
    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    try:
        data = pytesseract.image_to_data(image, lang=ocr_state["lang"], config=config,
                                         output_type=pytesseract.Output.DICT)
    except Exception:
        return "", 0.0
    texts = []
    confs = []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        txt = txt.strip()
        if not txt:
            continue
        try:
            c = float(conf)
        except Exception:
            continue
        if c >= 0:
            texts.append(txt)
            confs.append(c)
    if not texts:
        return "", 0.0
    text = normalize_plate_text("".join(texts))
    if not text:
        return "", 0.0
    return text, float(np.mean(confs) / 100.0)


def init_easyocr(lang):
    try:
        import easyocr
    except ImportError:
        return None, "easyocr is not installed"

    # Backward compatibility for old torch versions that do not support
    # torch.load(..., weights_only=...).
    try:
        import inspect
        sig = inspect.signature(torch.load)
        if "weights_only" not in sig.parameters:
            _orig_torch_load = torch.load

            def _torch_load_compat(*args, **kwargs):
                kwargs.pop("weights_only", None)
                return _orig_torch_load(*args, **kwargs)

            torch.load = _torch_load_compat
    except Exception:
        pass

    reader = easyocr.Reader([lang], gpu=False, verbose=False)
    return {"reader": reader}, None


def run_easyocr(ocr_state, image):
    reader = ocr_state["reader"]
    try:
        out = reader.readtext(image, detail=1, paragraph=False,
                              allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    except Exception:
        return "", 0.0
    if not out:
        return "", 0.0
    best = max(out, key=lambda x: x[2])
    text = normalize_plate_text(best[1])
    if not text:
        return "", 0.0
    return text, float(best[2])


def init_ocr(engine, tesseract_lang, tesseract_cmd, easyocr_lang):
    errors = []
    if engine in ("auto", "tesseract"):
        state, err = init_tesseract(tesseract_lang, tesseract_cmd)
        if state is not None:
            return "tesseract", state
        errors.append(err)
        if engine == "tesseract":
            raise RuntimeError("; ".join([e for e in errors if e]))
    if engine in ("auto", "easyocr"):
        state, err = init_easyocr(easyocr_lang)
        if state is not None:
            return "easyocr", state
        errors.append(err)
        if engine == "easyocr":
            raise RuntimeError("; ".join([e for e in errors if e]))
    raise RuntimeError("No OCR backend available: " + "; ".join([e for e in errors if e]))


def run_ocr(backend, state, plate_img):
    candidates = [plate_img] + preprocess_for_ocr(plate_img)
    best_text = ""
    best_score = 0.0
    for cand in candidates:
        if backend == "tesseract":
            text, score = run_tesseract(state, cand)
        else:
            text, score = run_easyocr(state, cand)
        if text and (score > best_score or (score == best_score and len(text) > len(best_text))):
            best_text = text
            best_score = score
    return best_text, best_score


def detect_plates(model, device, bgr_image, img_size, conf_thres, iou_thres):
    img0 = bgr_image.copy()
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
        det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], bgr_image.shape).round()
        det[:, 5:13] = scale_coords_landmarks(tensor.shape[2:], det[:, 5:13], bgr_image.shape).round()
        for row in det:
            x1, y1, x2, y2 = [int(v.item()) for v in row[:4]]
            conf = float(row[4].item())
            landmarks = [int(v.item()) for v in row[5:13]]
            cls_id = int(row[13].item())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "landmarks": landmarks,
                "class_id": cls_id
            })
    return detections


def draw_result(image, det, text, score):
    x1, y1, x2, y2 = det["bbox"]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f'{text if text else "N/A"} {score:.2f}'
    cv2.putText(image, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    lms = np.array(det["landmarks"], dtype=np.int32).reshape(4, 2)
    for p in lms:
        cv2.circle(image, tuple(p), 2, (0, 255, 255), -1)


def safe_crop(image, bbox):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model, device = load_detector(args.weights)
    ocr_backend, ocr_state = init_ocr(
        args.ocr_engine, args.tesseract_lang, args.tesseract_cmd, args.easyocr_lang
    )
    print(f"OCR backend: {ocr_backend}")
    print(f"Running detector on device: {device}")

    image_paths = get_image_paths(args.source)
    if not image_paths:
        print("No images found.")
        return

    all_results = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        detections = detect_plates(model, device, image, args.img_size, args.conf_thres, args.iou_thres)
        vis = image.copy()
        image_records = []

        for det in detections:
            crop = safe_crop(image, det["bbox"])
            warped = rectify_plate(image, det["landmarks"])

            text_candidates = []
            if crop is not None and crop.size > 0:
                text_candidates.append(run_ocr(ocr_backend, ocr_state, crop))
            if warped is not None and warped.size > 0:
                text_candidates.append(run_ocr(ocr_backend, ocr_state, warped))

            if text_candidates:
                best_text, best_score = max(text_candidates, key=lambda x: (x[1], len(x[0])))
            else:
                best_text, best_score = "", 0.0

            draw_result(vis, det, best_text, best_score)
            image_records.append({
                "bbox": det["bbox"],
                "conf": det["conf"],
                "landmarks": det["landmarks"],
                "class_id": det["class_id"],
                "plate_text": best_text,
                "ocr_score": round(best_score, 4)
            })

        out_img = save_dir / f"{image_path.stem}_ocr.jpg"
        cv2.imwrite(str(out_img), vis)
        print(f"Processed: {image_path} | plates={len(image_records)} | saved={out_img}")
        all_results.append({
            "image": str(image_path),
            "output_image": str(out_img),
            "plates": image_records
        })

    out_json = save_dir / "ocr_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved OCR json: {out_json}")


if __name__ == "__main__":
    main()
