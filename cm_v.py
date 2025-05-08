import cv2
import torch
import time
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolo11n.pt').to(device).eval()
cap = cv2.VideoCapture(4)  # Используйте 0 для встроенной камеры

# Настройки для сохранения видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # Желаемое количество кадров в секунду
output_file = 'output_video.mp4'

# Инициализация VideoWriter (используем кодек MP4V)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

prev_time = 0
fps_counter = 0

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

    # Записываем кадр в видеофайл
    out.write(frame)

    # Вычисление и отображение FPS
    current_time = time.time()
    fps_counter = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame,
               f"FPS: {int(fps_counter)}",
               (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.7,
               (0, 255, 0),
               2)

    cv2.imshow("Comp vision", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()  # Важно: закрываем VideoWriter
cv2.destroyAllWindows()
print(f"Видео сохранено как: {output_file}")