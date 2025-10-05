import cv2
import time
from collections import Counter


cap = cv2.VideoCapture(2)  

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the data format to MJPG
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
print("Format:", "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]))

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Camera width: {width}, height: {height}, FPS: {fps}")

prev_time = None
dropped_frames = 0
fps_list = []  # Store all FPS values

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            dropped_frames += 1
            print(f"Frame not captured, dropped frames {dropped_frames}")
            continue

        curr_time = time.time()
        if prev_time is not None:
            elapsed = curr_time - prev_time
            current_fps = 1.0 / elapsed
            fps_list.append(current_fps)
            print(f"Current FPS: {current_fps:.2f}")

            # Check for dropped frames
            expected_interval = 1.0 / fps
            if elapsed > expected_interval * 1.5:  # threshold
                missed = round(elapsed / expected_interval) - 1
                dropped_frames += missed
                print(f"Dropped {missed} frames, total dropped frames {dropped_frames}")

        prev_time = curr_time

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected. Exiting gracefully...")

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Print FPS stats
    if fps_list:
        average_fps = sum(fps_list) / len(fps_list)
        print(f"\nAverage FPS: {average_fps:.2f}")
        print("All unique recorded FPS values:")
        # Convert to int, drop decimals
        int_fps_list = [int(f) for f in fps_list]

        # Count occurrences
        fps_counts = Counter(int_fps_list)

        print("FPS values and their counts:")
        for fps_value, count in sorted(fps_counts.items()):
            print(f"{fps_value} FPS: {count} times")

    print(f"Total dropped frames: {dropped_frames}")
