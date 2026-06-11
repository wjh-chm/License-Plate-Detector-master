# -*- coding: utf-8 -*-
import argparse

import h5py
import numpy as np


TS_HINTS = ("timestamp", "time", "ts", "t")
TS_UNIT_TO_MS = {
    "s": 1000.0,
    "ms": 1.0,
    "us": 0.001,
    "ns": 0.000001,
}


def is_numeric_dataset(ds):
    return isinstance(ds, h5py.Dataset) and np.issubdtype(ds.dtype, np.number)


def is_time_like_name(name):
    lower = name.lower()
    return any(h in lower for h in TS_HINTS)


def collect_numeric_datasets(h5f):
    items = []

    def visitor(name, obj):
        if is_numeric_dataset(obj):
            items.append((name, obj))

    h5f.visititems(visitor)
    return items


def pick_timestamp_dataset(h5f):
    numeric = collect_numeric_datasets(h5f)
    if not numeric:
        return None, []

    preferred = []
    fallback = []
    matrix_candidates = []
    for name, ds in numeric:
        if ds.ndim == 1:
            if is_time_like_name(name):
                preferred.append((name, ds))
            else:
                fallback.append((name, ds))
        elif ds.ndim == 2 and ds.shape[1] >= 2:
            matrix_candidates.append((name, ds))

    if preferred:
        preferred.sort(key=lambda x: x[1].shape[0], reverse=True)
        return preferred[0], numeric
    if fallback:
        fallback.sort(key=lambda x: x[1].shape[0], reverse=True)
        return fallback[0], numeric
    if matrix_candidates:
        matrix_candidates.sort(key=lambda x: x[1].shape[0], reverse=True)
        return matrix_candidates[0], numeric

    return None, numeric


def pick_timestamp_from_event_matrix(ds):
    if ds.ndim != 2 or ds.shape[1] < 2:
        return None

    # Use uniform sampling across the whole dataset to detect time-like columns.
    n = min(int(ds.shape[0]), 5000)
    if n <= 0:
        return None
    if int(ds.shape[0]) == n:
        sample = ds[:n]
    else:
        idx = np.linspace(0, int(ds.shape[0]) - 1, n, dtype=np.int64)
        sample = ds[idx]
    if sample.size == 0:
        return None
    sample = np.asarray(sample, dtype=np.float64)

    best_col = None
    best_score = -1.0
    for col in range(sample.shape[1]):
        v = sample[:, col]
        if v.size < 3:
            continue
        dif = np.diff(v)
        mono_ratio = float(np.mean(dif >= 0))
        spread = float(np.std(v))
        if mono_ratio < 0.85 or spread <= 0:
            continue
        score = mono_ratio + min(spread / 1e6, 1.0) * 0.01
        if score > best_score:
            best_score = score
            best_col = col

    if best_col is None:
        return None
    return best_col


def estimate_frames(timestamps, frame_ms, ts_unit, time_start_ms=None, time_end_ms=None):
    t = np.asarray(timestamps).astype(np.float64).reshape(-1)
    if t.size == 0:
        return None

    t_min = float(np.min(t))
    t_max = float(np.max(t))
    if t_max < t_min:
        t_min, t_max = t_max, t_min

    full_span_ms = (t_max - t_min) * TS_UNIT_TO_MS[ts_unit]

    window_start_ms = 0.0 if time_start_ms is None else max(0.0, float(time_start_ms))
    window_end_ms = full_span_ms if time_end_ms is None else min(full_span_ms, float(time_end_ms))
    if window_end_ms < window_start_ms:
        window_end_ms = window_start_ms

    window_ms = max(0.0, window_end_ms - window_start_ms)
    full_frames = int(window_ms // frame_ms)
    remain_ms = window_ms - (full_frames * frame_ms)

    return {
        "t_min": t_min,
        "t_max": t_max,
        "full_span_ms": full_span_ms,
        "window_start_ms": window_start_ms,
        "window_end_ms": window_end_ms,
        "window_ms": window_ms,
        "frame_ms": frame_ms,
        "full_frames": full_frames,
        "remain_ms": remain_ms,
    }


def explore_h5_structure(h5_file, frame_ms, ts_unit, time_start_ms=None, time_end_ms=None):
    print(f"探索H5文件: {h5_file}")
    print("=" * 60)

    with h5py.File(h5_file, "r") as f:
        def print_structure(name, obj):
            print(f"{name}: {type(obj).__name__}")
            if isinstance(obj, h5py.Dataset):
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                #print(f"前十行{obj[:-1:10]}")
                if obj.shape and len(obj.shape) <= 2 and np.prod(obj.shape) <= 20:
                    print(f"  Data: {obj[:]}")
            elif isinstance(obj, h5py.Group):
                print(f"  Keys: {list(obj.keys())}")

        f.visititems(print_structure)

        print("\n" + "=" * 60)
        print("按时间估算可生成帧数")
        print("=" * 60)

        picked, numeric = pick_timestamp_dataset(f)
        if picked is None:
            print("未找到可用的数值型时间戳数据集。")
            if numeric:
                print("可选数值数据集:")
                for name, ds in numeric:
                    print(f"  - {name} | shape={ds.shape} dtype={ds.dtype}")
            return

        ts_name, ts_ds = picked
        ts_data = ts_ds[:]
        ts_desc = ts_name
        if ts_ds.ndim == 2:
            ts_col = pick_timestamp_from_event_matrix(ts_ds)
            if ts_col is None:
                print(f"检测到二维数据集 {ts_name}，但未识别到时间戳列。")
                return
            ts_data = ts_ds[:, ts_col]
            ts_desc = f"{ts_name}[:, {ts_col}]"

        info = estimate_frames(
            ts_data,
            frame_ms=frame_ms,
            ts_unit=ts_unit,
            time_start_ms=time_start_ms,
            time_end_ms=time_end_ms,
        )
        if info is None:
            print(f"时间戳数据集为空: {ts_desc}")
            return

        print(f"使用时间戳数据集: {ts_desc}")
        print(f"时间单位: {ts_unit}")
        print(f"原始时间范围: [{info['t_min']}, {info['t_max']}]")
        print(f"总时长: {info['full_span_ms']:.3f} ms")
        print(f"时间约束窗口: [{info['window_start_ms']:.3f}, {info['window_end_ms']:.3f}] ms")
        print(f"按每帧 {info['frame_ms']} ms 估算，可生成整帧数: {info['full_frames']}")
        print(f"剩余不足一帧时间: {info['remain_ms']:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="探索H5文件结构并按时间估算帧数")
    parser.add_argument("--h5_file", type=str, required=True, help="H5文件路径")
    parser.add_argument("--frame-ms", type=float, default=20.0, help="每帧时间窗口(ms)")
    parser.add_argument("--ts-unit", type=str, default="us", choices=["s", "ms", "us", "ns"], help="时间戳单位")
    parser.add_argument("--time-start-ms", type=float, default=None, help="时间窗口起点(ms, 相对首事件)")
    parser.add_argument("--time-end-ms", type=float, default=None, help="时间窗口终点(ms, 相对首事件)")
    args = parser.parse_args()

    explore_h5_structure(
        args.h5_file,
        frame_ms=args.frame_ms,
        ts_unit=args.ts_unit,
        time_start_ms=args.time_start_ms,
        time_end_ms=args.time_end_ms,
    )
