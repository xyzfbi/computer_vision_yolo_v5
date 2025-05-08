from ultralytics import YOLO
import cv2

# Загружаем модель
model = YOLO("yolo11n.pt")  # или ваш кастомный путь

# Открываем видеопоток с камеры (0 - индекс камеры по умолчанию)
cap = cv2.VideoCapture(4)

while cap.isOpened():
    # Читаем кадр с камеры
    success, frame = cap.read()

    if success:
        # Обрабатываем кадр с трекингом
        results = model.track(frame, persist=True)  # persist=True для сохранения ID между кадрами

        # Визуализируем результаты
        annotated_frame = results[0].plot()

        # Показываем результат
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()