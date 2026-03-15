import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# The path to your HDF5 event file
# This variable is available from the kernel state from a previous cell execution
# If you restart the runtime, you might need to re-run the `v2e` conversion cell
h5_filepath = 'D:\\project\\License-Plate-Detector-master\\imgs\\mp4\\9-black.h5'

# Load the events data
try:
    with h5py.File(h5_filepath, 'r') as f:
        events = f['events'][()]
    print(f"Successfully loaded {events.shape[0]} events from {h5_filepath}")
except Exception as e:
    print(f"Error loading HDF5 file: {e}")
    events = None # Ensure events is None if loading fails

if events is not None:
    # Events are typically (timestamp_us, x, y, polarity)
    # Timestamps are in microseconds
    timestamps_us = events[:, 0]
    x_coords = events[:, 1]
    y_coords = events[:, 2]
    polarities = events[:, 3]

    # Infer sensor resolution from event coordinates
    # The output_mode from v2e was 'dvs346', which corresponds to 346x260 (width x height)
    # v2e events are (t,x,y,p)
    sensor_width = int(x_coords.max() + 1)
    sensor_height = int(y_coords.max() + 1)

    # Use the known DVS346 resolution as a fallback or for consistency
    # if the inferred max coordinates are not reliable for the full sensor size.
    # For dvs346, it's 346x260 (width x height)
    if output_mode == 'dvs346':
        sensor_width = 346
        sensor_height = 260

    print(f"Inferred sensor resolution: {sensor_width}x{sensor_height} (WxH)")
    print(f"First 5 events:\n{events[:5]}")
else:
    print("No events loaded. Please ensure the HDF5 file exists and is valid.")

if events is None:
    print("Cannot proceed: No events data available.")
else:
    time_window_ms = 20 #@param {type:"number"} # 每个事件帧的时间窗口长度 (毫秒)
    distinguish_polarity = False #@param {type:"boolean"} # 是否区分事件极性

    print(f"事件窗口长度: {time_window_ms} ms")
    print(f"是否区分极性: {distinguish_polarity}")

    # Convert time window to microseconds
    time_window_us = time_window_ms * 1000

    # Calculate the total duration of events
    total_duration_us = timestamps_us[-1] - timestamps_us[0]
    print(f"总事件持续时间: {total_duration_us / 1e6:.2f} 秒")

    # Calculate number of time windows
    num_time_windows = int(np.ceil(total_duration_us / time_window_us))
    print(f"将生成的事件帧数量: {num_time_windows}")

    # Initialize accumulated frames
    if distinguish_polarity:
        # Store ON and OFF counts separately
        # Shape: (num_windows, height, width, 2) where last dim is [ON_count, OFF_count]
        accumulated_frames = np.zeros((num_time_windows, sensor_height, sensor_width, 2), dtype=np.int32)
    else:
        # Store total counts
        # Shape: (num_windows, height, width)
        accumulated_frames = np.zeros((num_time_windows, sensor_height, sensor_width), dtype=np.int32)

    # Process events
    print("正在处理事件并生成事件帧...")
    for i in range(events.shape[0]):
        t, x, y, p = events[i]

        # Normalize timestamp relative to the first event
        relative_t = t - timestamps_us[0]

        # Determine which time window this event falls into
        window_idx = int(relative_t // time_window_us)

        if 0 <= y < sensor_height and 0 <= x < sensor_width and 0 <= window_idx < num_time_windows:
            if distinguish_polarity:
                if p == 1: # ON event
                    accumulated_frames[window_idx, y, x, 0] += 1
                elif p == -1: # OFF event
                    accumulated_frames[window_idx, y, x, 1] += 1
            else:
                accumulated_frames[window_idx, y, x] += 1

    print("事件帧生成完成。")
    print(f"生成的事件帧序列形状: {accumulated_frames.shape}")

if 'accumulated_frames' not in locals() or accumulated_frames is None:
    print("No event frames to visualize. Please ensure previous steps ran successfully.")
else:
    num_frames_to_display = min(5, num_time_windows) # Display up to 5 frames

    plt.figure(figsize=(num_frames_to_display * 4, 8))

    for i in range(num_frames_to_display):
        plt.subplot(2, num_frames_to_display, i + 1)
        frame = accumulated_frames[i]
        if distinguish_polarity:
            # If distinguishing polarity, sum ON and OFF for display, or show one channel
            display_frame = frame[:, :, 0] + frame[:, :, 1] # Sum ON+OFF for grayscale display
            plt.title(f"Frame {i} (ON+OFF)")
        else:
            display_frame = frame
            plt.title(f"Frame {i}")
        plt.imshow(display_frame, cmap='gray', origin='lower')
        plt.colorbar()
        plt.axis('off')

    # Display last few frames as well
    if num_time_windows > num_frames_to_display:
        for i in range(num_frames_to_display):
            if num_time_windows - num_frames_to_display + i >= num_frames_to_windows:
                break
            frame_idx = num_time_windows - num_frames_to_display + i
            plt.subplot(2, num_frames_to_display, num_frames_to_display + i + 1)
            frame = accumulated_frames[frame_idx]
            if distinguish_polarity:
                display_frame = frame[:, :, 0] + frame[:, :, 1]
                plt.title(f"Frame {frame_idx} (ON+OFF)")
            else:
                display_frame = frame
                plt.title(f"Frame {frame_idx}")
            plt.imshow(display_frame, cmap='gray', origin='lower')
            plt.colorbar()
            plt.axis('off')

    plt.tight_layout()
    plt.show()
