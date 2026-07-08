from ultralytics import YOLO
import cv2

model = YOLO('best.pt')  # ← your model

# Path to your image
image_path = 'r0_223.jpg'

# Run classification
results = model.predict(source=image_path, conf=0.5)

for result in results:
    probs   = result.probs
    label   = model.names[probs.top1]
    conf    = float(probs.top1conf) * 100

    print(f"\n🍎 Apple Quality Result:")
    print(f"   • {label}: {conf:.1f}%")

    # Show image with banner
    frame = cv2.imread(image_path)
    color = (0, 200, 0) if label == "good_apple" else (0, 0, 220)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), color, -1)
    cv2.putText(frame, f"{label.upper().replace('_', ' ')}  {conf:.1f}%",
                (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    cv2.imshow('Apple Quality Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()