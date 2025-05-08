import cv2
import torch
import time
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Загрузка локальной модели
model = YOLO('yolo11n.pt').to(device).eval()

cap = cv2.VideoCapture(4)

# FPS
prev_time = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    with torch.no_grad():
        results = model(frame, verbose=False)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[cls_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{label} {conf * 100:.1f}%",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    cv2.imshow("Comp vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()