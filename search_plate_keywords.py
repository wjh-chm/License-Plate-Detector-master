# -*- coding: utf-8 -*-
import argparse
import json
import re
from pathlib import Path


CN_PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
CN_PLATE_PATTERN = re.compile(rf"[{CN_PROVINCES}][A-Z][A-Z0-9]{{5,6}}")


def normalize_text(text):
    text = text or ""
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", text.upper())


def extract_cn_plate(text):
    norm = normalize_text(text)
    hits = CN_PLATE_PATTERN.findall(norm)
    if not hits:
        return ""
    hits = sorted(hits, key=len, reverse=True)
    return hits[0]


def parse_keywords(keywords_str):
    if not keywords_str:
        return []
    return [normalize_text(x) for x in keywords_str.split(",") if x.strip()]


def contains_keywords(text, keywords):
    if not keywords:
        return False
    norm = normalize_text(text)
    return any(k in norm for k in keywords)


def load_records(input_json):
    with Path(input_json).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def search_records(records, keywords, plate_only):
    matches = []
    for item in records:
        image = item.get("image", "")
        for plate in item.get("plates", []):
            raw_text = plate.get("plate_text", "")
            candidate = plate.get("plate_candidate", "") or extract_cn_plate(raw_text)
            text_for_search = candidate if candidate else raw_text

            if plate_only and not candidate:
                continue

            keyword_hit = contains_keywords(text_for_search, keywords) if keywords else True
            if not keyword_hit:
                continue

            matches.append({
                "image": image,
                "plate_text": raw_text,
                "plate_candidate": candidate,
                "bbox": plate.get("bbox", []),
                "ocr_score": plate.get("ocr_score", 0.0),
            })
    return matches


def main():
    parser = argparse.ArgumentParser(description="Search OCR results by Chinese plate format and keywords.")
    parser.add_argument("--input-json", type=str, default="runs/plate_ocr_gpu/ocr_results.json", help="OCR result json path")
    parser.add_argument("--keywords", type=str, default="", help="Comma separated keywords, e.g. 粤B,京A")
    parser.add_argument("--plate-only", action="store_true", help="Only keep China plate-format candidates")
    parser.add_argument("--output-json", type=str, default="runs/plate_ocr_gpu/keyword_search_results.json", help="Search output json")
    args = parser.parse_args()

    records = load_records(args.input_json)
    keywords = parse_keywords(args.keywords)
    matches = search_records(records, keywords, args.plate_only)

    out = {
        "input_json": args.input_json,
        "keywords": keywords,
        "plate_only": args.plate_only,
        "total_matches": len(matches),
        "matches": matches,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved search result: {out_path}")
    print(f"Total matches: {len(matches)}")


if __name__ == "__main__":
    main()
