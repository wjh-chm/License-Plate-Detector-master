import argparse
import os
import cv2
import numpy as np
import math


def find_green_rects(img):
    # 找到图像中绿色矩形（标注框）的外接矩形列表
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 绿色范围（宽容度较大以兼容不同渲染）
    lower = np.array([35, 50, 50])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # 膨胀/闭运算填充断裂的线条
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # 忽略噪声
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # 忽略过瘦或过扁的轮廓
        if w < 10 or h < 10:
            continue
        rects.append((x, y, w, h))
    # 如果没有找到，返回空列表
    return rects


def generate_video(img_path, out_path=None, duration=3.0, fps=30, amp=0.6):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {img_path}')

    h, w = img.shape[:2]
    if out_path is None:
        base = os.path.splitext(img_path)[0]
        out_path = base + '_light.mp4'

    rects = find_green_rects(img)
    if not rects:
        print('No green rectangles found; trying edge-based fallback to locate boxes.')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            x, y, ww, hh = cv2.boundingRect(cnt)
            rects.append((x, y, ww, hh))

    # 如果仍为空，直接使用整个图像作为 ROI
    if not rects:
        rects = [(0, 0, w, h)]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    frames = max(1, int(duration * fps))

    for i in range(frames):
        t = i / float(frames - 1) if frames > 1 else 0.0
        # alpha 从 1 -> 1+amp -> 1 （半周期的正弦）
        alpha = 1.0 + amp * math.sin(math.pi * t)
        frame = img.copy()
        for (x, y, rw, rh) in rects:
            # 截取 ROI 并调整亮度
            roi = frame[y:y+rh, x:x+rw]
            # 使用 convertScaleAbs 保持类型与饱和度
            bright = cv2.convertScaleAbs(roi, alpha=alpha, beta=0)
            frame[y:y+rh, x:x+rw] = bright

        writer.write(frame)

    writer.release()
    print('Saved video:', out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image', '-i', type=str, default='imgs/9_r.jpg', help='input image path')
    p.add_argument('--out', '-o', type=str, default=None, help='output video path')
    p.add_argument('--duration', type=float, default=3.0, help='video duration seconds')
    p.add_argument('--fps', type=int, default=30, help='frames per second')
    p.add_argument('--amp', type=float, default=0.6, help='brightness amplitude (multiplier delta)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    img_path = args.image
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.getcwd(), img_path)
    try:
        generate_video(img_path, out_path=args.out, duration=args.duration, fps=args.fps, amp=args.amp)
    except Exception as e:
        print('Error:', e)
