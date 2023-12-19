import numpy as np

# Подготовка
w = 4056 
h =  3040
fx = 1855.4991679340574
fy = 1851.5306550481187
# Расчет
fov_x = np.rad2deg(2 * np.arctan(w / (2 * fx)))
fov_y = np.rad2deg(2 * np.arctan(h / (2 * fy)))

print("Field of View (degrees):")
print(f"  Horizontal: {fov_x:.1f}")
print(f"  Vertical: {fov_y:.1f}")
