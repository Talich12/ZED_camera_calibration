import cv2
import numpy as np

# Загрузка изображения
img = cv2.imread('./air/frame17.png')

fx = 1888.5348589470525
fy = 1883.303996763965
cx = 2010.7878387901337
cy = 1084.6168608497705
k1 = -0.2852251294450805
k2 = 0.1707185111320959
p1 = -0.00033042104469238056
p2 = 0.005170719651189922
k3 = -0.12257870433584951

# Матрица камеры и коэффициенты дисторсии
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2, k3])

# Применение undistort
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)

# Сохранение результата
cv2.imwrite('undistorted_image.jpg', undistorted_img)
