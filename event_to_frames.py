# -*- coding: UTF-8 -*-
import h5py
import numpy as np
import argparse
import cv2
import os

def load_events_from_h5(h5_file):
    """
    从H5文件加载事件数据
    返回: events数组，形状为(N, 4)，每行为[x, y, timestamp, polarity]
    """
    with h5py.File(h5_file, 'r') as f:
        events = f['events'][:]
    return events

def create_event_frames(events, time_window_ms=20, width=346, height=260, separate_polarity=False, max_frames=None):
    """
    将事件数据转换为事件帧序列
    
    参数:
        events: 事件数据，形状为(N, 4)，每行为[时间(微秒), x, y, 极性]
        time_window_ms: 时间窗口大小（毫秒）
        width: 传感器宽度 (DVS346为346)
        height: 传感器高度 (DVS346为260)
        separate_polarity: 是否区分极性（True时输出双通道，False时输出单通道）
        max_frames: 最大生成帧数（None表示不限制）
    
    返回:
        event_frames: 事件帧列表，每个帧的形状根据separate_polarity决定
        frame_timestamps: 每个帧对应的时间戳
    """
    if len(events) == 0:
        print("警告: 没有事件数据")
        return [], []
    
    # 提取时间戳并计算时间窗口
    timestamps = events[:, 0]  # 第0列是时间（微秒）
    min_time = np.min(timestamps)
    max_time = np.max(timestamps)
    
    # 转换为秒
    total_duration_sec = (max_time - min_time) / 1000000
    print(f"事件总数: {len(events)}")
    print(f"时间范围: {min_time} - {max_time} (总时长: {total_duration_sec:.2f}秒)")
    print(f"时间窗口: {time_window_ms}ms")
    
    # 计算时间窗口数量（将时间窗口转换为微秒）
    time_window_us = time_window_ms * 1000
    total_duration_us = max_time - min_time
    num_windows = int(np.ceil(total_duration_us / time_window_us))
    
    # 如果指定了最大帧数，限制生成的帧数
    if max_frames is not None and max_frames > 0:
        num_windows = min(num_windows, max_frames)
        print(f"限制生成帧数为: {max_frames}")
    
    print(f"将生成 {num_windows} 个事件帧")
    
    event_frames = []
    frame_timestamps = []
    
    # 按时间窗口处理事件
    for i in range(num_windows):
        window_start = min_time + i * time_window_us
        window_end = window_start + time_window_us
        
        # 提取当前时间窗口内的事件
        mask = (timestamps >= window_start) & (timestamps < window_end)
        window_events = events[mask]
        
        # 创建空帧（所有像素初始为0）
        if separate_polarity:
            frame = np.zeros((height, width, 2), dtype=np.uint8)
        else:
            frame = np.zeros((height, width), dtype=np.uint8)
        
        # 如果有事件，将对应像素设置为255
        if len(window_events) > 0:
            if separate_polarity:
                # 区分极性：ON和OFF事件分别处理
                # ON事件（polarity=1）
                on_events = window_events[window_events[:, 3] == 1]
                if len(on_events) > 0:
                    x_coords = on_events[:, 1].astype(int)  # 第1列是x
                    y_coords = on_events[:, 2].astype(int)  # 第2列是y
                    # 确保坐标在图像范围内
                    valid_mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
                    x_coords = x_coords[valid_mask]
                    y_coords = y_coords[valid_mask]
                    # 将有事件的像素设为255
                    frame[y_coords, x_coords, 0] = 255
                
                # OFF事件（polarity=0）
                off_events = window_events[window_events[:, 3] == 0]
                if len(off_events) > 0:
                    x_coords = off_events[:, 1].astype(int)  # 第1列是x
                    y_coords = off_events[:, 2].astype(int)  # 第2列是y
                    valid_mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
                    x_coords = x_coords[valid_mask]
                    y_coords = y_coords[valid_mask]
                    # 将有事件的像素设为255
                    frame[y_coords, x_coords, 1] = 255
            else:
                # 不区分极性：所有事件都设为255
                x_coords = window_events[:, 1].astype(int)  # 第1列是x
                y_coords = window_events[:, 2].astype(int)  # 第2列是y
                valid_mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
                x_coords = x_coords[valid_mask]
                y_coords = y_coords[valid_mask]
                # 将有事件的像素设为255
                frame[y_coords, x_coords] = 255
        
        event_frames.append(frame)
        frame_timestamps.append(window_end)
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{num_windows} 帧...")
    
    print(f"事件帧生成完成！")
    return event_frames, frame_timestamps

def visualize_event_frames(event_frames, output_dir, frame_timestamps, separate_polarity=False):
    """
    可视化事件帧并保存为图片
    
    参数:
        event_frames: 事件帧列表
        output_dir: 输出目录
        frame_timestamps: 每个帧的时间戳（微秒）
        separate_polarity: 是否区分极性
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始保存事件帧到: {output_dir}")
    
    for i, frame in enumerate(event_frames):
        if separate_polarity:
            # 双通道：分别显示ON和OFF事件
            on_frame = frame[:, :, 0]
            off_frame = frame[:, :, 1]
            
            # 创建彩色可视化：红色=ON，蓝色=OFF
            color_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            color_frame[:, :, 2] = on_frame  # 红色通道 (ON事件)
            color_frame[:, :, 0] = off_frame  # 蓝色通道 (OFF事件)
            
            # 添加时间戳信息（转换为毫秒）
            timestamp_ms = frame_timestamps[i] / 1000
            cv2.putText(color_frame, f"Frame {i+1} - {timestamp_ms:.1f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            output_path = os.path.join(output_dir, f"frame_{i+1:06d}.png")
            cv2.imwrite(output_path, color_frame)
        else:
            # 单通道：显示事件
            # 帧已经是0-255范围，不需要归一化
            display_frame = frame.astype(np.uint8)
            
            # 添加时间戳信息（转换为毫秒）
            timestamp_ms = frame_timestamps[i] / 1000
            cv2.putText(display_frame, f"Frame {i+1} - {timestamp_ms:.1f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            output_path = os.path.join(output_dir, f"frame_{i+1:06d}.png")
            cv2.imwrite(output_path, display_frame)
    
    print(f"所有事件帧已保存到: {output_dir}")

def save_event_frames_as_video(event_frames, output_video_path, fps=30, frame_timestamps=None, separate_polarity=False):
    """
    将事件帧保存为视频文件
    
    参数:
        event_frames: 事件帧列表
        output_video_path: 输出视频路径
        fps: 帧率
        frame_timestamps: 每个帧的时间戳
        separate_polarity: 是否区分极性
    """
    if len(event_frames) == 0:
        print("没有事件帧可保存")
        return
    
    # 获取第一帧的尺寸
    first_frame = event_frames[0]
    if separate_polarity:
        height, width = first_frame.shape[:2]
        is_color = True
    else:
        height, width = first_frame.shape
        is_color = False
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), is_color)
    
    print(f"开始保存视频到: {output_video_path}")
    
    for i, frame in enumerate(event_frames):
        if separate_polarity:
            # 双通道：创建彩色可视化
            on_frame = frame[:, :, 0]
            off_frame = frame[:, :, 1]
            
            on_normalized = cv2.normalize(on_frame, None, 0, 255, cv2.NORM_MINMAX)
            off_normalized = cv2.normalize(off_frame, None, 0, 255, cv2.NORM_MINMAX)
            
            color_frame = np.zeros((height, width, 3), dtype=np.uint8)
            color_frame[:, :, 2] = on_normalized
            color_frame[:, :, 0] = off_normalized
            
            if frame_timestamps:
                timestamp_ms = frame_timestamps[i]
                cv2.putText(color_frame, f"Frame {i+1} - {timestamp_ms}ms", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            video_writer.write(color_frame)
        else:
            # 单通道：创建灰度可视化
            normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            normalized = normalized.astype(np.uint8)
            
            if frame_timestamps:
                timestamp_ms = frame_timestamps[i]
                cv2.putText(normalized, f"Frame {i+1} - {timestamp_ms}ms", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            video_writer.write(normalized)
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(event_frames)} 帧...")
    
    video_writer.release()
    print(f"视频已保存: {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description='将事件数据转换为事件帧序列')
    parser.add_argument('--h5_file', type=str, required=True, help='H5文件路径')
    parser.add_argument('--time_window', type=int, default=20, help='时间窗口大小（毫秒）')
    parser.add_argument('--width', type=int, default=346, help='传感器宽度 (DVS346为346)')
    parser.add_argument('--height', type=int, default=260, help='传感器高度 (DVS346为260)')
    parser.add_argument('--separate_polarity', action='store_true', help='是否区分极性')
    parser.add_argument('--max_frames', type=int, default=None, help='最大生成帧数（None表示不限制）')
    parser.add_argument('--output_dir', type=str, default='event_frames', help='输出目录')
    parser.add_argument('--output_video', type=str, default=None, help='输出视频路径（可选）')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("事件帧生成程序")
    print("=" * 60)
    
    # 加载事件数据
    print(f"\n加载事件数据: {args.h5_file}")
    events = load_events_from_h5(args.h5_file)
    
    # 创建事件帧
    print(f"\n生成事件帧...")
    event_frames, frame_timestamps = create_event_frames(
        events, 
        time_window_ms=args.time_window, 
        width=args.width, 
        height=args.height, 
        separate_polarity=args.separate_polarity,
        max_frames=args.max_frames
    )
    
    # 保存事件帧为图片
    print(f"\n保存事件帧为图片...")
    visualize_event_frames(
        event_frames, 
        args.output_dir, 
        frame_timestamps, 
        separate_polarity=args.separate_polarity
    )
    
    # 可选：保存为视频
    if args.output_video:
        print(f"\n保存事件帧为视频...")
        save_event_frames_as_video(
            event_frames, 
            args.output_video, 
            fps=args.fps, 
            frame_timestamps=frame_timestamps, 
            separate_polarity=args.separate_polarity
        )
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"事件帧总数: {len(event_frames)}")
    print(f"输出目录: {args.output_dir}")
    if args.output_video:
        print(f"输出视频: {args.output_video}")

if __name__ == '__main__':
    main()
