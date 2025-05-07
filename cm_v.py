import cv2
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device).eval()

cap = cv2.VideoCapture(0)



# fps
prev_time = 0
fps = 0

while True:
    ret, frame = cap.read()


    with torch.no_grad(), torch.amp.autocast(device_type=device):
        results = model(frame, size=320)

    predictions = results.xyxy[0].cpu().numpy()

    for *box, confidence, class_id in predictions:
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(class_id)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence * 100:.1f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Comp vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()