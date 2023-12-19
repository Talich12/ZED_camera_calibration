import numpy as np

# Подготовка
w = 7.564  # ширина сенсора в мм
h = 5.476  # высота сенсора в мм
fx_pixel = 1855.4991679340574
fy_pixel = 1851.5306550481187
pixel_size = 1.55  # размер пикселя в мкм

# Преобразование фокусного расстояния из пикселей в миллиметры
fx = fx_pixel * pixel_size / 1000
fy = fy_pixel * pixel_size / 1000

# Расчет
fov_x = np.rad2deg(2 * np.arctan(w / (2 * fx)))
fov_y = np.rad2deg(2 * np.arctan(h / (2 * fy)))

print("Field of View (degrees):")
print(f"  Horizontal: {fov_x:.1f}")
print(f"  Vertical: {fov_y:.1f}")
