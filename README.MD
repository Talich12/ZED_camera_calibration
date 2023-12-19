# Дилема расчета углов
## Есть 3 разных способа рассчитывать углы обзора камеры
### 1. Расчет по пикселям кадра
```python
import  numpy  as  np
# Подготовка
w  =  3840
h  =  2160
fx  =  1855.4991679340574
fy  =  1851.5306550481187

# Расчет
fov_x  =  np.rad2deg(2  *  np.arctan(w  / (2  *  fx)))
fov_y  =  np.rad2deg(2  *  np.arctan(h  / (2  *  fy)))  

print("Field of View (degrees):")
print(f" Horizontal: {fov_x:.1f}")
print(f" Vertical: {fov_y:.1f}")
```
В данном случае высота и ширина брались из данных о размере кадра который выдавала камера.
В таком случае получались данные результаты:
```python 
Field of View (degrees):
  Horizontal: 92.0
  Vertical: 60.5
 ```
### 2. Расчет по разрешению камеры
```python
import  numpy  as  np
# Подготовка
w  =  4056
h  =  3040
fx  =  1855.4991679340574
fy  =  1851.5306550481187

# Расчет
fov_x  =  np.rad2deg(2  *  np.arctan(w  / (2  *  fx)))
fov_y  =  np.rad2deg(2  *  np.arctan(h  / (2  *  fy)))

print("Field of View (degrees):")
print(f" Horizontal: {fov_x:.1f}")
print(f" Vertical: {fov_y:.1f}")
```
В данном примере высота и ширина задавались данными из документации камеры (брались значения ее разрешения).
В таком случае получались данные результаты:
```python
Field of View (degrees):
  Horizontal: 95.1
  Vertical: 78.8
```
### 2. Расчет по размеру сенсора и размеру пикселя на камере
```python
import  numpy  as  np  
# Подготовка
w  =  7.564  # ширина сенсора в мм
h  =  5.476  # высота сенсора в мм
fx_pixel  =  1855.4991679340574
fy_pixel  =  1851.5306550481187
pixel_size  =  1.55  # размер пикселя в мкм 

# Преобразование фокусного расстояния из пикселей в миллиметры
fx  =  fx_pixel  *  pixel_size  /  1000
fy  =  fy_pixel  *  pixel_size  /  1000

# Расчет
fov_x  =  np.rad2deg(2  *  np.arctan(w  / (2  *  fx)))
fov_y  =  np.rad2deg(2  *  np.arctan(h  / (2  *  fy)))
  
print("Field of View (degrees):")
print(f" Horizontal: {fov_x:.1f}")
print(f" Vertical: {fov_y:.1f}")
```
В данном случае за высоту и ширину брались размеры сенсора камеры, а также размер пикселя в мкм. (согласно документации).
Получились такие результаты:
```python
Field of View (degrees):
  Horizontal: 105.5
  Vertical: 87.3
  ```
## Вопрос кому верить?
### Ни один из данных результатов не приблизился к углам обзора камеры указанных в документации, если учитывать что документация может врать, то нет полной уверенности в ни одном из методов нахождения углов обзора!
