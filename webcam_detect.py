from ultralytics import YOLO
import cv2

# Load YOUR trained apple model instead of the default yolov8n
model = YOLO('best.pt')  # make sure best.pt is in the same folder

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🍎 Apple Quality Detector started - Press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run classification on the frame
    results = model.predict(source=frame, conf=0.5, verbose=False)
    probs   = results[0].probs
    top_idx = probs.top1
    label   = model.names[top_idx]          # "good_apple" or "bad_apple"
    conf    = float(probs.top1conf) * 100   # e.g. 94.2

    # Pick color — green for good, red for bad
    color = (0, 200, 0) if label == "good_apple" else (0, 0, 220)

    # Draw a filled banner at the top
    cv2.rectangle(frame, (0, 0), (640, 60), color, -1)

    # Label text
    text = f"{'GOOD APPLE' if label == 'good_apple' else 'BAD APPLE'}  {conf:.1f}%"
    cv2.putText(frame, text, (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    # Small confidence bar at bottom
    bar_width = int((conf / 100) * 640)
    cv2.rectangle(frame, (0, 470), (bar_width, 480), color, -1)

    cv2.imshow('Apple Quality Detector - Press Q to Exit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed")