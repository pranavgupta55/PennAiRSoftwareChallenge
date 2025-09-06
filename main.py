import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def process_video_for_shapes(input_video_path, output_video_path):

    red = (86, 45, 143)
    white = (212, 213, 211)
    green = (64, 90, 58)
    blue = (56, 22, 1)

    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing '{input_video_path}' with {frame_count} frames at {fps:.2f} FPS...")
    start_time = time.time()

    for _ in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Canny Edge Detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 50, 150, apertureSize=3)

        edges_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 2. Dilate to get clean shapes
        kernel = np.ones((3, 3), np.uint8)
        dilated_img = cv2.dilate(edges_display, kernel, iterations=5)
        inverted_img = cv2.bitwise_not(dilated_img)

        # 3. Contour detection
        output_frame = inverted_img.copy()
        gray = cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

        for i, contour in enumerate(contours):
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.01 * perimeter  # roughly approximating the perimeter for cleaner edges
            approx = cv2.approxPolyDP(contour, epsilon, True)

            cv2.drawContours(output_frame, [approx], 0, (0, 255, 0), 2)

            # Calculate and draw the center for reference
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(output_frame, (cX, cY), 5, (0, 0, 255), -1)
                cv2.putText(output_frame, f"Shape {i+1}", (cX + 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 3)
                cv2.putText(output_frame, f"Shape {i+1}", (cX + 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)

            cv2.putText(output_frame, f"({cX}, {cY})", (cX - 40, cY + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 3)
            cv2.putText(output_frame, f"({cX}, {cY})", (cX - 40, cY + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)

        out.write(output_frame)

    cap.release()
    out.release()
    print(f"Processing complete. Output video saved to {output_video_path}")
    total_time = time.time() - start_time
    print(f"\n✅ Processing complete. Output video saved to '{output_video_path}'")
    print(f"🕒 Total processing time: {total_time:.2f} seconds")


input_video = 'Dynamic24.mp4'
output_video = 'testruns/dynamicOutput1.mp4'

process_video_for_shapes(input_video, output_video)