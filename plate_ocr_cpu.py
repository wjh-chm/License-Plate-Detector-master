# -*- coding: utf-8 -*-
"""
OCR脚本：
1) 读取 detect_plate.py 输出的检测 JSON（bbox/landmarks）
2) 按检测框裁剪并做 OCR
3) 输出 OCR 可视化图和结果 JSON
"""

import argparse
import json
import os
import re
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PLATE_CHARS_REGEX = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]")
CN_PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
CN_PLATE_PATTERN = re.compile(rf"[{CN_PROVINCES}][A-Z][A-Z0-9]{{5,6}}")
_TEXT_FONT = None


def _find_cjk_font():
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for fp in candidates:
        if Path(fp).exists():
            return fp
    return None


def _get_text_font(size=24):
    global _TEXT_FONT
    if _TEXT_FONT is not None:
        return _TEXT_FONT
    font_path = _find_cjk_font()
    if font_path is None:
        _TEXT_FONT = ImageFont.load_default()
    else:
        _TEXT_FONT = ImageFont.truetype(font_path, size=size)
    return _TEXT_FONT


def parse_args():
    parser = argparse.ArgumentParser(description="OCR from detection JSON")
    parser.add_argument("--det-json", type=str, required=True, help="Detection JSON file, index JSON, or per-image JSON directory")
    parser.add_argument("--ocr-engine", type=str, default="paddleocr", choices=["paddleocr", "auto", "easyocr", "tesseract"], help="OCR backend")

    parser.add_argument("--paddle-lang", type=str, default="ch", help="PaddleOCR language")
    parser.add_argument("--paddle-use-gpu", action="store_true", help="Use GPU in PaddleOCR")
    parser.add_argument("--paddle-use-angle-cls", action="store_true", help="Enable angle classifier")
    parser.add_argument("--paddle-model-root", type=str, default="runs/paddle_worker_models", help="PaddleOCR offline model root")

    parser.add_argument("--easyocr-lang", type=str, default="en", help="EasyOCR language")
    parser.add_argument("--easyocr-gpu", action="store_true", help="Use GPU in EasyOCR")

    parser.add_argument("--tesseract-cmd", type=str, default="", help="Optional tesseract executable path")
    parser.add_argument("--tesseract-lang", type=str, default="eng", help="Tesseract language")

    parser.add_argument("--keywords", type=str, default="", help="Comma-separated keywords")
    parser.add_argument("--save-dir", type=str, default="runs/plate_ocr_from_det", help="Output directory")
    return parser.parse_args()


def load_detection_records(det_json):
    path = Path(det_json)
    if not path.exists():
        raise FileNotFoundError(f"Detection JSON not found: {det_json}")

    # 1) Directory input: load all per-image json files.
    if path.is_dir():
        records = []
        for jp in sorted(path.glob("*.json")):
            with jp.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "image" in data and "detections" in data:
                records.append(data)
        return records

    # 2) File input.
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 2.1) Index format from detect_plate.py: each item has field "json".
    if isinstance(data, list) and data and isinstance(data[0], dict) and "json" in data[0]:
        records = []
        for item in data:
            jp = Path(item.get("json", ""))
            if not jp.exists():
                continue
            with jp.open("r", encoding="utf-8") as f:
                one = json.load(f)
            if isinstance(one, dict) and "image" in one and "detections" in one:
                records.append(one)
        return records

    # 2.2) Backward-compatible old format: list of full records.
    if isinstance(data, list):
        return data

    # 2.3) New per-image single record format.
    if isinstance(data, dict) and "image" in data and "detections" in data:
        return [data]

    # 2.4) Also support index object: {"items": [...]}.
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        records = []
        for item in data["items"]:
            jp = Path(item.get("json", ""))
            if not jp.exists():
                continue
            with jp.open("r", encoding="utf-8") as f:
                one = json.load(f)
            if isinstance(one, dict) and "image" in one and "detections" in one:
                records.append(one)
        return records

    raise ValueError("Unsupported detection JSON format")


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
    return [plate_img, up, th]


def normalize_plate_text(text):
    cleaned = text.strip().upper().replace(" ", "")
    return PLATE_CHARS_REGEX.sub("", cleaned)


def extract_cn_plate_candidate(text):
    if not text:
        return ""
    norm = normalize_plate_text(text)
    hits = CN_PLATE_PATTERN.findall(norm)
    if not hits:
        return ""
    hits = sorted(hits, key=len, reverse=True)
    return hits[0]


def is_infer_model_dir(path):
    p = Path(path)
    if not p.is_dir():
        return False
    has_params = (p / "inference.pdiparams").exists()
    has_model = (p / "inference.pdmodel").exists() or (p / "inference.json").exists()
    return has_params and has_model


def extract_role_archives(base_dir, role):
    base = Path(base_dir)
    if not base.exists():
        return

    for tar_path in sorted(base.glob("*.tar")):
        name = tar_path.name.lower()
        if role not in name:
            continue
        try:
            with tarfile.open(tar_path, "r") as tf:
                tf.extractall(path=base)
            print(f"Extracted {role} archive: {tar_path}")
        except Exception as e:
            print(f"Warning: failed to extract {tar_path}: {e}")


def pick_model_dir(base_dir, role):
    base = Path(base_dir)
    if not base.exists():
        return None

    # Exact directory first: <base>/<role>/inference.*
    role_dir = base / role
    if is_infer_model_dir(role_dir):
        return role_dir

    # Then nested under role directory.
    if role_dir.exists():
        nested = [p for p in role_dir.rglob("*") if is_infer_model_dir(p)]
        if nested:
            nested.sort(key=lambda p: len(p.parts))
            return nested[0]

    # Finally search under base and prefer folder names that include role.
    candidates = [p for p in base.rglob("*") if is_infer_model_dir(p)]
    if not candidates:
        return None

    role_candidates = [p for p in candidates if role in p.name.lower() or f"_{role}_" in p.name.lower()]
    if not role_candidates:
        return None
    return role_candidates[0]


def resolve_paddle_model_dirs(model_root):
    root = Path(model_root)
    root.mkdir(parents=True, exist_ok=True)

    # Auto-extract tar archives from role subdirectories only.
    for role in ("det", "rec", "cls"):
        extract_role_archives(root / role, role)

    det_dir = pick_model_dir(root, "det")
    rec_dir = pick_model_dir(root, "rec")
    cls_dir = pick_model_dir(root, "cls")
    return det_dir, rec_dir, cls_dir


def init_paddleocr(lang, use_gpu, use_angle_cls, model_root):
    # Compatibility for Python < 3.9 where stdlib zoneinfo is missing.
    if sys.version_info < (3, 9):
        try:
            import zoneinfo  # noqa: F401
        except ImportError:
            try:
                from backports import zoneinfo as backports_zoneinfo
                sys.modules["zoneinfo"] = backports_zoneinfo
            except Exception:
                pass

    try:
        import paddleocr
    except ImportError:
        return None, "paddleocr is not installed"

    _, rec_dir, cls_dir = resolve_paddle_model_dirs(model_root)
    if rec_dir is None:
        return None, (
            f"Offline PaddleOCR model incomplete under {model_root}. "
            f"Need rec model directory containing inference files."
        )

    print(f"Using rec model: {rec_dir}")
    if use_angle_cls:
        print("Warning: rec-only mode does not use cls. --paddle-use-angle-cls will be ignored.")

    # PaddleOCR 3.x+: use TextRecognition to avoid det/cls dependency.
    text_recognition_cls = getattr(paddleocr, "TextRecognition", None)
    if text_recognition_cls is not None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        device = "gpu" if use_gpu else "cpu"
        rec = text_recognition_cls(
            model_name="PP-OCRv5_mobile_rec",
            model_dir=str(rec_dir),
            device=device,
        )
        return {
            "reader": rec,
            "backend": "text_recognition",
            "rec_model_dir": str(rec_dir),
            "cls_model_dir": "",
            "use_angle_cls": False,
        }, None

    # PaddleOCR 2.x fallback.
    try:
        from paddleocr import PaddleOCR
    except Exception as e:
        return None, f"paddleocr import failed: {e}"

    kwargs = dict(
        det=False,
        rec=True,
        use_angle_cls=False,
        lang=lang,
        use_gpu=use_gpu,
        det_model_dir=str(rec_dir),
        rec_model_dir=str(rec_dir),
    )
    ocr = PaddleOCR(**kwargs)
    return {
        "reader": ocr,
        "backend": "paddleocr_legacy",
        "rec_model_dir": str(rec_dir),
        "cls_model_dir": "",
        "use_angle_cls": False,
    }, None


def run_paddleocr(state, image):
    reader = state["reader"]
    if state.get("backend") == "text_recognition":
        try:
            result = reader.predict(image)
        except Exception:
            return "", 0.0
        if not result:
            return "", 0.0
        texts = []
        confs = []
        for entry in result:
            if not isinstance(entry, dict):
                continue
            txt = str(entry.get("rec_text", "")).strip()
            conf = float(entry.get("rec_score", 0.0))
            if txt:
                texts.append(txt)
                confs.append(conf)
        if not texts:
            return "", 0.0
        merged = normalize_plate_text("".join(texts))
        if not merged:
            return "", 0.0
        return merged, float(np.mean(confs)) if confs else 0.0

    try:
        result = reader.ocr(image, det=False, rec=True, cls=state.get("use_angle_cls", False))
    except Exception:
        return "", 0.0
    if not result:
        return "", 0.0

    texts = []
    confs = []

    # rec-only output may be:
    # 1) [[text, score], ...]
    # 2) [[[text, score], ...]]
    entries = result
    if (
        isinstance(result, list)
        and len(result) == 1
        and isinstance(result[0], list)
    ):
        nested = result[0]
        if nested and isinstance(nested[0], (list, tuple)) and len(nested[0]) >= 2:
            entries = nested

    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        txt = str(entry[0]).strip()
        try:
            conf = float(entry[1])
        except Exception:
            conf = 0.0
        if txt:
            texts.append(txt)
            confs.append(conf)

    if not texts:
        return "", 0.0
    merged = normalize_plate_text("".join(texts))
    if not merged:
        return "", 0.0
    return merged, float(np.mean(confs)) if confs else 0.0


def init_easyocr(lang, use_gpu):
    try:
        import easyocr
    except ImportError:
        return None, "easyocr is not installed"

    reader = easyocr.Reader([lang], gpu=use_gpu, verbose=False)
    return {"reader": reader}, None


def run_easyocr(state, image):
    reader = state["reader"]
    try:
        out = reader.readtext(image, detail=1, paragraph=False)
    except Exception:
        return "", 0.0
    if not out:
        return "", 0.0
    merged = normalize_plate_text("".join([x[1] for x in out]))
    if not merged:
        return "", 0.0
    return merged, float(np.mean([x[2] for x in out]))


def init_tesseract(lang, tesseract_cmd):
    try:
        import pytesseract
    except ImportError:
        return None, "pytesseract is not installed"

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    return {"module": pytesseract, "lang": lang}, None


def run_tesseract(state, image):
    pytesseract = state["module"]
    config = "--oem 3 --psm 7"
    try:
        data = pytesseract.image_to_data(image, lang=state["lang"], config=config,
                                         output_type=pytesseract.Output.DICT)
    except Exception:
        return "", 0.0

    texts, confs = [], []
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
    merged = normalize_plate_text("".join(texts))
    if not merged:
        return "", 0.0
    return merged, float(np.mean(confs) / 100.0)


def init_ocr(args):
    errors = []

    if args.ocr_engine in ("paddleocr", "auto"):
        state, err = init_paddleocr(args.paddle_lang, args.paddle_use_gpu, args.paddle_use_angle_cls, args.paddle_model_root)
        if state is not None:
            return "paddleocr", state
        errors.append(err)
        if args.ocr_engine == "paddleocr":
            raise RuntimeError("; ".join([e for e in errors if e]))

    if args.ocr_engine in ("easyocr", "auto"):
        state, err = init_easyocr(args.easyocr_lang, args.easyocr_gpu)
        if state is not None:
            return "easyocr", state
        errors.append(err)
        if args.ocr_engine == "easyocr":
            raise RuntimeError("; ".join([e for e in errors if e]))

    if args.ocr_engine in ("tesseract", "auto"):
        state, err = init_tesseract(args.tesseract_lang, args.tesseract_cmd)
        if state is not None:
            return "tesseract", state
        errors.append(err)
        if args.ocr_engine == "tesseract":
            raise RuntimeError("; ".join([e for e in errors if e]))

    raise RuntimeError("No OCR backend available: " + "; ".join([e for e in errors if e]))


def run_ocr(engine, state, plate_img):
    best_text = ""
    best_score = 0.0

    for cand in preprocess_for_ocr(plate_img):
        if engine == "paddleocr":
            text, score = run_paddleocr(state, cand)
        elif engine == "easyocr":
            text, score = run_easyocr(state, cand)
        else:
            text, score = run_tesseract(state, cand)

        if text and (score > best_score or (score == best_score and len(text) > len(best_text))):
            best_text, best_score = text, score

    return best_text, best_score


def safe_crop(image, bbox):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def draw_result(image, bbox, text, score):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f'{text if text else "N/A"} {score:.2f}'
    try:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = _get_text_font(size=24)
        tx = x1
        ty = max(0, y1 - 28)
        draw.text((tx, ty), label, font=font, fill=(0, 255, 0))
        image[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(image, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def parse_keywords(keywords_str):
    if not keywords_str:
        return []
    return [k.strip().upper().replace(" ", "") for k in keywords_str.split(",") if k.strip()]


def contains_keywords(text, keywords):
    if not keywords or not text:
        return False
    upper_text = text.upper().replace(" ", "")
    return any(k in upper_text for k in keywords)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    det_records = load_detection_records(args.det_json)
    ocr_engine, ocr_state = init_ocr(args)
    keywords = parse_keywords(args.keywords)

    print(f"OCR backend: {ocr_engine}")
    print(f"Detection records: {len(det_records)}")

    all_results = []
    keyword_hits = []

    for item in det_records:
        img_path = Path(item.get("image", ""))
        detections = item.get("detections", [])

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Skip unreadable image: {img_path}")
            continue

        vis = image.copy()
        ocr_plates = []

        for det in detections:
            bbox = det.get("bbox", [])
            landmarks = det.get("landmarks", [])

            crop = safe_crop(image, bbox)
            warped = rectify_plate(image, landmarks) if len(landmarks) == 8 else None

            text_candidates = []
            if crop is not None and crop.size > 0:
                text_candidates.append(run_ocr(ocr_engine, ocr_state, crop))
            if warped is not None and warped.size > 0:
                text_candidates.append(run_ocr(ocr_engine, ocr_state, warped))

            if text_candidates:
                best_text, best_score = max(text_candidates, key=lambda x: (x[1], len(x[0])))
            else:
                best_text, best_score = "", 0.0

            plate_candidate = extract_cn_plate_candidate(best_text)
            query_text = plate_candidate if plate_candidate else best_text
            hit = contains_keywords(query_text, keywords)

            draw_text = plate_candidate if plate_candidate else best_text
            draw_result(vis, bbox, draw_text, best_score)

            record = {
                "bbox": bbox,
                "conf": det.get("conf", 0.0),
                "landmarks": landmarks,
                "class_id": det.get("class_id", -1),
                "plate_text": best_text,
                "plate_candidate": plate_candidate,
                "is_cn_plate_like": bool(plate_candidate),
                "keyword_hit": hit,
                "ocr_score": round(best_score, 4),
            }
            ocr_plates.append(record)

            if hit:
                keyword_hits.append({
                    "image": str(img_path),
                    "plate": query_text,
                    "score": round(best_score, 4),
                    "bbox": bbox,
                })

        out_img = save_dir / f"{img_path.stem}_ocr.jpg"
        cv2.imwrite(str(out_img), vis)

        per_image_result = {
            "image": str(img_path),
            "source_detection_record": item,
            "output_image": str(out_img),
            "plates": ocr_plates,
        }
        per_img_json = save_dir / f"{img_path.stem}_ocr.json"
        with per_img_json.open("w", encoding="utf-8") as f:
            json.dump(per_image_result, f, ensure_ascii=False, indent=2)

        all_results.append(per_image_result)

        print(f"Processed OCR: {img_path} | dets={len(detections)} | saved={out_img} | json={per_img_json}")

    out_json = save_dir / "ocr_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved OCR json: {out_json}")

    if keywords:
        hit_json = save_dir / "keyword_hits.json"
        with hit_json.open("w", encoding="utf-8") as f:
            json.dump(keyword_hits, f, ensure_ascii=False, indent=2)
        print(f"Saved keyword hits: {hit_json} | hits={len(keyword_hits)}")


if __name__ == "__main__":
    main()
