import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a tiled plate dataset with YOLO-style 13-column labels.")
    parser.add_argument(
        "--source-root",
        type=str,
        default="D:/project/dataloader/exports/fusion_round1_quad_dataset",
        help="Dataset root containing images/train, images/val, labels/train, labels/val",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="D:/project/dataloader/exports/fusion_round1_quad_dataset_tiles_192_o64",
        help="Output tiled dataset root",
    )
    parser.add_argument("--tile-width", type=int, default=192, help="Tile width in pixels")
    parser.add_argument("--tile-height", type=int, default=192, help="Tile height in pixels")
    parser.add_argument("--overlap-x", type=int, default=64, help="Horizontal overlap in pixels")
    parser.add_argument("--overlap-y", type=int, default=64, help="Vertical overlap in pixels")
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.7,
        help="Minimum retained bbox area ratio for keeping a tiled label",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=2.0,
        help="Minimum clipped box width/height in pixels to keep",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the output directory before regenerating tiles",
    )
    return parser.parse_args()


def sliding_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]

    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def load_label_rows(label_path):
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []

    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        rows.append([float(x) for x in parts])
    return rows


def normalized_to_xyxy(row, image_width, image_height):
    cx = row[1] * image_width
    cy = row[2] * image_height
    bw = row[3] * image_width
    bh = row[4] * image_height
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return [x1, y1, x2, y2]


def clip_box_to_tile(box, left, top, right, bottom):
    x1 = max(box[0], left)
    y1 = max(box[1], top)
    x2 = min(box[2], right)
    y2 = min(box[3], bottom)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def tile_label_rows(rows, image_width, image_height, left, top, tile_width, tile_height, min_area_ratio, min_box_size):
    right = left + tile_width
    bottom = top + tile_height
    tiled_rows = []

    for row in rows:
        cls_id = int(row[0])
        full_box = normalized_to_xyxy(row, image_width, image_height)
        clipped_box = clip_box_to_tile(full_box, left, top, right, bottom)
        if clipped_box is None:
            continue

        full_area = box_area(full_box)
        clipped_area = box_area(clipped_box)
        if full_area <= 0:
            continue
        if clipped_area / full_area < min_area_ratio:
            continue

        clipped_width = clipped_box[2] - clipped_box[0]
        clipped_height = clipped_box[3] - clipped_box[1]
        if clipped_width < min_box_size or clipped_height < min_box_size:
            continue

        local_x1 = clipped_box[0] - left
        local_y1 = clipped_box[1] - top
        local_x2 = clipped_box[2] - left
        local_y2 = clipped_box[3] - top

        cx = (local_x1 + local_x2) / 2.0 / tile_width
        cy = (local_y1 + local_y2) / 2.0 / tile_height
        bw = clipped_width / tile_width
        bh = clipped_height / tile_height
        x1 = local_x1 / tile_width
        y1 = local_y1 / tile_height
        x2 = local_x2 / tile_width
        y2 = local_y1 / tile_height
        x3 = local_x2 / tile_width
        y3 = local_y2 / tile_height
        x4 = local_x1 / tile_width
        y4 = local_y2 / tile_height

        tiled_rows.append(
            [
                cls_id,
                cx,
                cy,
                bw,
                bh,
                x1,
                y1,
                x2,
                y2,
                x3,
                y3,
                x4,
                y4,
            ]
        )

    return tiled_rows


def write_label_rows(label_path, rows):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        label_path.write_text("", encoding="utf-8")
        return

    lines = []
    for row in rows:
        line = [str(int(row[0]))] + [f"{value:.6f}" for value in row[1:]]
        lines.append(" ".join(line))
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_split(source_root, output_root, split, tile_width, tile_height, overlap_x, overlap_y, min_area_ratio, min_box_size):
    image_root = source_root / "images" / split
    label_root = source_root / "labels" / split
    out_image_root = output_root / "images" / split
    out_label_root = output_root / "labels" / split
    out_image_root.mkdir(parents=True, exist_ok=True)
    out_label_root.mkdir(parents=True, exist_ok=True)

    stride_x = max(1, tile_width - overlap_x)
    stride_y = max(1, tile_height - overlap_y)

    summary = {
        "images": 0,
        "tiles": 0,
        "positive_tiles": 0,
        "empty_tiles": 0,
        "labels": 0,
    }

    for image_path in sorted(p for p in image_root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        rows = load_label_rows(label_path)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size

            x_starts = sliding_starts(width, tile_width, stride_x)
            y_starts = sliding_starts(height, tile_height, stride_y)

            for top in y_starts:
                for left in x_starts:
                    right = min(left + tile_width, width)
                    bottom = min(top + tile_height, height)

                    tile = image.crop((left, top, right, bottom))
                    if tile.size != (tile_width, tile_height):
                        padded = Image.new("RGB", (tile_width, tile_height))
                        padded.paste(tile, (0, 0))
                        tile = padded

                    tiled_rows = tile_label_rows(
                        rows,
                        width,
                        height,
                        left,
                        top,
                        tile_width,
                        tile_height,
                        min_area_ratio,
                        min_box_size,
                    )

                    sequence_dir = relative.parent
                    tile_stem = f"{image_path.stem}__x{left}_y{top}"
                    out_image_path = out_image_root / sequence_dir / f"{tile_stem}{image_path.suffix.lower()}"
                    out_label_path = out_label_root / sequence_dir / f"{tile_stem}.txt"
                    out_image_path.parent.mkdir(parents=True, exist_ok=True)

                    tile.save(out_image_path)
                    write_label_rows(out_label_path, tiled_rows)

                    summary["tiles"] += 1
                    if tiled_rows:
                        summary["positive_tiles"] += 1
                        summary["labels"] += len(tiled_rows)
                    else:
                        summary["empty_tiles"] += 1

        summary["images"] += 1

    return summary


def write_split_txt(output_root, split):
    image_root = output_root / "images" / split
    txt_path = output_root / f"{split}.txt"
    lines = [str(path.resolve()) for path in sorted(image_root.rglob("*")) if path.suffix.lower() in IMAGE_SUFFIXES]
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_dataset_yaml(output_root):
    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"train: {output_root.as_posix()}/images/train",
                f"val: {output_root.as_posix()}/images/val",
                "",
                "nc: 1",
                "names: ['plate']",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)

    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    report = {
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "tile_width": args.tile_width,
        "tile_height": args.tile_height,
        "overlap_x": args.overlap_x,
        "overlap_y": args.overlap_y,
        "min_area_ratio": args.min_area_ratio,
        "min_box_size": args.min_box_size,
        "splits": {},
    }

    for split in ("train", "val"):
        summary = process_split(
            source_root,
            output_root,
            split,
            args.tile_width,
            args.tile_height,
            args.overlap_x,
            args.overlap_y,
            args.min_area_ratio,
            args.min_box_size,
        )
        write_split_txt(output_root, split)
        report["splits"][split] = summary

    write_dataset_yaml(output_root)

    report_path = output_root / "tile_export_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Summary written to {report_path}")


if __name__ == "__main__":
    main()
