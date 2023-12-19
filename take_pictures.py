import cv2

# Открываем видеофайл
cap = cv2.VideoCapture('water.mp4')

# Проверяем, удалось ли открыть видео
if not cap.isOpened():
    print("Ошибка при открытии видео")
    exit()

# Номер кадра
frame_num = 0
count = 0
while True:
    # Читаем кадр
    ret, frame = cap.read()

    # Если кадр не прочитан, значит, видео закончилось
    if not ret:
        break

    # Если это каждый 30-й кадр, сохраняем его в PNG
    if frame_num % 30 == 0:
        cv2.imwrite(f'air/frame{count}.jpg', frame)
        count += 1

    # Увеличиваем номер кадра
    frame_num += 1

# Закрываем видеофайл
cap.release()
