import cv2
import numpy as np
import os
import time
from tqdm import tqdm

def process_video_for_shapes_and_steps(input_video_path, output_folder):
    # --- 1. Ensure the output folder exists ---
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True) # exist_ok=True prevents error if folder already exists
        print(f"Output folder '{output_folder}' ensured.")
    except Exception as e:
        print(f"Error creating output folder '{output_folder}': {e}")
        return # Exit if folder cannot be created

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
        "canny": {"filename": "canny_edges.mp4", "is_color": False}, # Expects 1-channel
        "dilated_eroded_inverted": {"filename": "dilated_eroded_inverted.mp4", "is_color": False}, # Expects 1-channel
        "final_shapes": {"filename": "final_shapes_detected.mp4", "is_color": True}
    }

    # --- 4. Initialize all video writers ---
    print("Initializing video writers...")
    all_writers_opened = True
    for step_name, config in steps_config.items():
        output_path = os.path.join(output_folder, config["filename"])
        
        # Ensure dimensions match the input video
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=config["is_color"])
        
        if not writer.isOpened():
            print(f"CRITICAL ERROR: Could not open video writer for '{step_name}' at '{output_path}'.")
            print("This could be due to an unsupported codec, invalid path, or missing write permissions.")
            all_writers_opened = False
            break # Stop if any writer fails to open
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

    # Define colors for final shapes step (BGR format)
    red = (143, 45, 86) 
    white = (211, 213, 212) 
    green_contour = (0, 255, 0) 
    blue_center = (255, 0, 0)
    blue_text = (1, 22, 56) 

    # --- 5. Main Frame Processing Loop ---
    for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Reached end of video early at frame {frame_idx}.")
            break

        # Write original frame (is_color=True, so it's 3-channel BGR)
        if "original" in video_writers:
            video_writers["original"].write(frame)

        # 1. Canny Edge Detection
        edges = cv2.Canny(frame, 0, 50, apertureSize=3)
        if "canny" in video_writers:
            # Writer was initialized with isColor=False, so write the 1-channel grayscale image directly
            video_writers["canny"].write(edges) 

        # 2. Dilate, erode, and invert
        kernel = np.ones((15, 15), np.uint8)
        dilated_img = cv2.dilate(edges, kernel, iterations=3)
        eroded = cv2.erode(dilated_img, kernel, iterations=1)
        inverted_img = cv2.bitwise_not(eroded)
        if "dilated_eroded_inverted" in video_writers:
            # Writer was initialized with isColor=False, so write the 1-channel grayscale image directly
            video_writers["dilated_eroded_inverted"].write(inverted_img) 

        # 3. Contour detection and filtering (Final Shapes)
        contours, _ = cv2.findContours(inverted_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        output_frame = np.zeros_like(frame) # Start with a black canvas for the final shapes (3-channel BGR)

        area_threshold = 1000

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > area_threshold:
                cv2.drawContours(output_frame, [contour], -1, (255, 255, 255), thickness=cv2.FILLED) 

                perimeter = cv2.arcLength(contour, True)
                epsilon = 0.01 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)

                cv2.drawContours(output_frame, [approx], 0, green_contour, 2) 

                M = cv2.moments(approx)
                if M["m00"] != 0: 
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(output_frame, (cX, cY), 5, blue_center, -1) 
                    
                    cv2.putText(output_frame, f"Shape {i+1}", (cX + 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 3)
                    cv2.putText(output_frame, f"Shape {i+1}", (cX + 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)

                    cv2.putText(output_frame, f"({cX}, {cY})", (cX - 40, cY + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_text, 3)
                    cv2.putText(output_frame, f"({cX}, {cY})", (cX - 40, cY + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
        
        # Writer was initialized with isColor=True, so write the 3-channel BGR output_frame
        if "final_shapes" in video_writers:
            video_writers["final_shapes"].write(output_frame)

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

# Example Usage
input_video = 'DynamicHard24.mp4' 
output_base_folder = 'testruns_multistep'

process_video_for_shapes_and_steps(input_video, output_base_folder)