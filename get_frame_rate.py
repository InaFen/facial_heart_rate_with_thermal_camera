import cv2
import time

cap = cv2.VideoCapture(2)  

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the data format to MJPG
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"BGR3"))
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
print("Format:", "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]))

# Retrieve the FPS from the camera
# Set first so fps setting works correctly
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

#cap.set(cv2.CAP_PROP_FPS, 9)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Camera width: {width}, height: {height}, FPS: {fps}")

prev_time = None
dropped_frames = 0

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
        print(f"Current FPS: {current_fps:.2f}")
        # If elapsed time is bigger than expected interval, frames were dropped
        expected_interval = 1.0 / fps
        if elapsed > expected_interval * 1.5:  # threshold
            missed = round(elapsed / expected_interval) - 1 # -1 to remove current frame
            dropped_frames += missed
            print(f"Dropped {missed} frames, total dropped frames {dropped_frames}")

    prev_time = curr_time

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Total dropped frames: {dropped_frames}")
cap.release()
cv2.destroyAllWindows()