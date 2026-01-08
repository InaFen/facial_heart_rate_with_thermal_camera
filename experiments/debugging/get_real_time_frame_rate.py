import cv2
import time

cap = cv2.VideoCapture(2)  # change the index if needed

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()


if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the data format to MJPG
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
print(f"Format: {cap.get(cv2.CAP_PROP_FOURCC)}")

# Retrieve the FPS from the camera
# Set first so fps setting works correctly
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

cap.set(cv2.CAP_PROP_FPS, 25)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Camera width: {width}, height: {height}, FPS: {fps}")



frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    cv2.imshow("Frame", frame)

    # Calculate FPS every second
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        actual_fps = frame_count / elapsed
        print(f"Actual FPS: {actual_fps:.2f}")
        frame_count = 0
        start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

