import cv2
import numpy as np
import os
import time
from tqdm import tqdm

def process_video_for_shapes_and_steps(input_video_path, output_folder):
    # --- 1. Ensure the output folder exists ---
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder '{output_folder}' ensured.")
    except Exception as e:
        print(f"Error creating output folder '{output_folder}': {e}")
        return

    cap = cv2.VideoCapture(input_video_path)

    # --- 2. Check if video was opened successfully ---
    if not cap.isOpened():
        print(f"Error: Could not open input video '{input_video_path}'. Please check the path and file integrity.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        print(f"Warning: Input video '{input_video_path}' has 0 or negative frames. No processing will occur.")
        cap.release()
        return

    print(f"Input Video Properties: {frame_count} frames, {fps:.2f} FPS, {frame_width}x{frame_height} resolution.")

    # Define the codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # --- 3. Define all video steps and their properties ---
    video_writers = {}
    steps_config = {
        "original": {"filename": "original_video.mp4", "is_color": True},
        "hsv_mask": {"filename": "hsv_mask_video.mp4", "is_color": False},
        "isolated_trapezoid": {"filename": "isolated_trapezoid_video.mp4", "is_color": False},
        "combined_mask": {"filename": "combined_mask_video.mp4", "is_color": True}, # Storing the visual combined mask
        "outlined_shapes": {"filename": "outlined_shapes_video.mp4", "is_color": True} # Final output with outlines and labels
    }

    # --- 4. Initialize all video writers ---
    print("Initializing video writers...")
    all_writers_opened = True
    for step_name, config in steps_config.items():
        output_path = os.path.join(output_folder, config["filename"])
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=config["is_color"])

        if not writer.isOpened():
            print(f"CRITICAL ERROR: Could not open video writer for '{step_name}' at '{output_path}'.")
            all_writers_opened = False
            break
        else:
            print(f"Writer '{step_name}' opened successfully at '{output_path}'.")
            video_writers[step_name] = writer

    if not all_writers_opened:
        for writer_key in video_writers:
            if video_writers[writer_key].isOpened():
                video_writers[writer_key].release()
        cap.release()
        print("Aborting due to critical video writer error.")
        return

    print(f"Processing '{input_video_path}' with {frame_count} frames at {fps:.2f} FPS...")
    start_time = time.time()

    # Define colors for drawing (BGR format)
    red = (0, 0, 255) # BGR for red
    white = (255, 255, 255) # BGR for white
    green_contour = (0, 255, 0) # BGR for green
    blue_text = (255, 0, 0) # BGR for blue

    # HSV color range for the first mask (from your Jupyter notebook)
    lower_hsv_mask1 = np.array([0, 20, 20])
    upper_hsv_mask1 = np.array([255, 255, 255])

    # HSV color range for the second mask (from your Jupyter notebook)
    lower_hsv_mask2 = np.array([0, 0, 0])
    upper_hsv_mask2 = np.array([255, 5, 255])

    # Contour filtering parameters for isolated trapezoid
    min_area_trap = 5000
    max_area_trap = 30000
    min_aspect_ratio_trap = 0.5
    max_aspect_ratio_trap = 1.7
    area_threshold_final_shapes = 1000 # Minimum contour area for the final outlined shapes

    # --- 5. Main Frame Processing Loop ---
    for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"Warning: Reached end of video early at frame {frame_idx}.")
            break

        # Write original frame
        if "original" in video_writers:
            video_writers["original"].write(frame_bgr)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # Keep an RGB version for combined_mask display

        # --- Process for Cleaned HSV Mask (hsv_binaries from Jupyter) ---
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hsv_mask_initial = cv2.inRange(frame_hsv, lower_hsv_mask1, upper_hsv_mask1)
        blurred_hsv = cv2.GaussianBlur(hsv_mask_initial, (11, 11), 0)
        _, clean_hsv_mask = cv2.threshold(blurred_hsv, 127, 255, cv2.THRESH_BINARY)
        kernel_hsv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        clean_hsv_mask = cv2.morphologyEx(clean_hsv_mask, cv2.MORPH_OPEN, kernel_hsv, iterations=2)
        clean_hsv_mask = cv2.morphologyEx(clean_hsv_mask, cv2.MORPH_CLOSE, kernel_hsv, iterations=2)

        if "hsv_mask" in video_writers:
            video_writers["hsv_mask"].write(clean_hsv_mask)

        # --- Process for Isolated Trapezoid Mask (final_masks from Jupyter) ---
        hsv_mask_trap = cv2.inRange(frame_hsv, lower_hsv_mask2, upper_hsv_mask2)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (5,5), 0)
        _, otsu_mask = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        combined_mask_trap = cv2.bitwise_and(hsv_mask_trap, otsu_mask)
        kernel_trap = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        clean_combined_mask_trap = cv2.morphologyEx(combined_mask_trap, cv2.MORPH_CLOSE, kernel_trap)
        clean_combined_mask_trap = cv2.morphologyEx(clean_combined_mask_trap, cv2.MORPH_OPEN, kernel_trap)

        contours_trap, _ = cv2.findContours(clean_combined_mask_trap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_trap_mask = np.zeros_like(clean_combined_mask_trap)

        filtered_contours_trap = []
        for cnt in contours_trap:
            area = cv2.contourArea(cnt)
            if min_area_trap < area < max_area_trap:
                x,y,w,h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if min_aspect_ratio_trap < aspect_ratio < max_aspect_ratio_trap:
                    filtered_contours_trap.append(cnt)

        if filtered_contours_trap:
            largest_contour_trap = max(filtered_contours_trap, key=cv2.contourArea)
            cv2.drawContours(final_trap_mask, [largest_contour_trap], -1, 255, thickness=-1)

        if "isolated_trapezoid" in video_writers:
            video_writers["isolated_trapezoid"].write(final_trap_mask)

        # --- Combine Masks (union_mask from Jupyter) ---
        union_mask = cv2.bitwise_or(clean_hsv_mask, final_trap_mask)

        # Create a visual representation of the combined mask
        union_masked_rgb = cv2.bitwise_and(frame_rgb, frame_rgb, mask=union_mask)
        union_masked_bgr = cv2.cvtColor(union_masked_rgb, cv2.COLOR_RGB2BGR) # Convert back to BGR for writer

        if "combined_mask" in video_writers:
            video_writers["combined_mask"].write(union_masked_bgr)

        # --- Final Outlined Shapes ---
        overlay_frame = frame_bgr.copy() # Start with the original frame for overlay

        contours_final, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shape_count = 0
        for cnt in contours_final:
            area = cv2.contourArea(cnt)
            if area > area_threshold_final_shapes:
                shape_count += 1
                perimeter = cv2.arcLength(cnt, True)
                epsilon = 0.01 * perimeter
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                cv2.drawContours(overlay_frame, [approx], -1, green_contour, 10) # BGR green

                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    cv2.circle(overlay_frame, (cX, cY), 5, red, -1) # BGR red

                    # Text labels with white outline
                    cv2.putText(overlay_frame, f"Shape {shape_count}", (cX + 10, cY + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 3)
                    cv2.putText(overlay_frame, f"Shape {shape_count}", (cX + 10, cY + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)

                    coord_text = f"({cX}, {cY})"
                    cv2.putText(overlay_frame, coord_text, (cX - 40, cY + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue_text, 3)
                    cv2.putText(overlay_frame, coord_text, (cX - 40, cY + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)

        if "outlined_shapes" in video_writers:
            video_writers["outlined_shapes"].write(overlay_frame)

    # --- 6. Release all resources ---
    cap.release()
    for step_name, writer in video_writers.items():
        if writer.isOpened():
            writer.release()
            print(f"Writer '{step_name}' released.")
        else:
            print(f"Writer '{step_name}' was not opened or already released.")

    total_time = time.time() - start_time
    print(f"\n✅ Processing complete. Output videos saved to '{output_folder}'")
    print(f"🕒 Total processing time: {total_time:.2f} seconds")
    print(f"🕒 Processing rate: {frame_count / total_time:.2f} fps")

# Example Usage
input_video = 'DynamicHard24.mp4'
output_base_folder = 'processed_video_output2'

process_video_for_shapes_and_steps(input_video, output_base_folder)