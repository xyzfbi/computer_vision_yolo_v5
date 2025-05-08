import cv2
import numpy as np

# Пути к файлам модели
model_weights = "mobilenet_iter_73000.caffemodel"
model_config = "deploy.prototxt"
class_file = "coco.names"

# Загрузка модели
net = cv2.dnn.readNetFromCaffe(model_config, model_weights)

# Загрузка классов COCO
with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# Инициализация камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Подготовка кадра для нейросети
    blob = cv2.dnn.blobFromImage(
        frame,
        0.007843,  # Масштабирование
        (300, 300),  # Размер входа сети
        (127.5, 127.5, 127.5),  # Средние значения для нормализации
        swapRB=True
    )

    # Подаем данные в сеть
    net.setInput(blob)
    detections = net.forward()

    # Обработка результатов
    h, w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Порог уверенности
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Рисуем прямоугольник и подпись
            label = f"{class_names[class_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()