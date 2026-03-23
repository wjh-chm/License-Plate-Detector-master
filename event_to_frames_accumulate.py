# -*- coding: utf-8 -*-
"""
Method 2:
Within each time window, accumulate event counts per pixel,
then normalize the whole frame to 0-255 as new brightness values.
Generate one frame per window, and save one every N frames (default N=5).
"""

import argparse
import os

import cv2
import h5py
import numpy as np


def load_events_from_h5(h5_file):
    with h5py.File(h5_file, "r") as f:
        if "events" not in f:
            raise KeyError("Dataset 'events' not found in h5 file.")
        return f["events"][:]


def _get_valid_xy(events, width, height):
    x = events[:, 1].astype(np.int32)
    y = events[:, 2].astype(np.int32)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[valid], y[valid]


def make_accumulate_window_frame(window_events, width, height, acc_threshold):
    counts = np.zeros((height, width), dtype=np.float32)
    if len(window_events) == 0:
        return counts.astype(np.uint8)

    x, y = _get_valid_xy(window_events, width, height)
    np.add.at(counts, (y, x), 1.0)

    if acc_threshold > 0:
        counts = np.minimum(counts, float(acc_threshold))

    max_val = float(np.max(counts))
    if max_val <= 0:
        return np.zeros((height, width), dtype=np.uint8)

    norm = (counts / max_val) * 255.0
    return norm.astype(np.uint8)


def process_events_to_frames(events, window_ms, width, height, acc_threshold, max_frames=None):
    if len(events) == 0:
        return []

    timestamps = events[:, 0]
    t_min = int(np.min(timestamps))
    t_max = int(np.max(timestamps))
    window_us = int(window_ms * 1000)
    total_windows = int(np.ceil((t_max - t_min) / float(window_us)))
    if max_frames is not None and max_frames > 0:
        total_windows = min(total_windows, int(max_frames))

    frames = []
    for i in range(total_windows):
        start_us = t_min + i * window_us
        end_us = start_us + window_us
        mask = (timestamps >= start_us) & (timestamps < end_us)
        window_events = events[mask]
        frame = make_accumulate_window_frame(window_events, width, height, acc_threshold)
        frames.append((i + 1, end_us, frame))
    return frames


def save_every_n_frames(frames, output_dir, save_every=5):
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for frame_idx, ts_us, frame in frames:
        if frame_idx % save_every != 0:
            continue
        ts_ms = ts_us / 1000.0
        out_name = f"frame_{frame_idx:06d}_{ts_ms:.1f}ms.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, frame)
        saved += 1
    return saved


def main():
    parser = argparse.ArgumentParser(description="Event to frame (accumulate + normalize method).")
    parser.add_argument("--h5-file", type=str, required=True, help="Input h5 file path")
    parser.add_argument("--window-ms", type=float, default=20, help="Time window size in milliseconds")
    parser.add_argument("--width", type=int, default=346, help="Sensor width")
    parser.add_argument("--height", type=int, default=260, help="Sensor height")
    parser.add_argument("--acc-threshold", type=float, default=0.0, help="Per-pixel accumulation clip threshold, 0 means no clip")
    parser.add_argument("--max-frames", type=int, default=None, help="Max generated frames")
    parser.add_argument("--save-every", type=int, default=5, help="Save one frame every N generated frames")
    parser.add_argument("--output-dir", type=str, default="event_frames_accumulate", help="Output directory")
    args = parser.parse_args()

    if args.save_every <= 0:
        raise ValueError("--save-every must be > 0")

    print("Loading events...")
    events = load_events_from_h5(args.h5_file)
    print(f"Total events: {len(events)}")
    print(f"Window size: {args.window_ms} ms")
    print(f"Accumulation threshold: {args.acc_threshold}")

    frames = process_events_to_frames(
        events=events,
        window_ms=args.window_ms,
        width=args.width,
        height=args.height,
        acc_threshold=args.acc_threshold,
        max_frames=args.max_frames,
    )
    saved = save_every_n_frames(frames, args.output_dir, args.save_every)

    print(f"Generated frames: {len(frames)}")
    print(f"Saved frames (every {args.save_every}): {saved}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
