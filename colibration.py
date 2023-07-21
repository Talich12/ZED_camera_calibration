import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import configparser

def write_to_config(R, t, filename, config):
    # Инициализируйте объект configparser и создайте новую секцию STEREO
    config['STEREO'] = {}

    # Запись матрицы вращения и вектора переноса
    config['STEREO']['Baseline'] = str(np.linalg.norm(t))
    config['STEREO']['TY'] = str(t[1][0])
    config['STEREO']['TZ'] = str(t[2][0])
    config['STEREO']['CV_HD'] = str()  # Неизвестное значение, необходимо заменить
    config['STEREO']['RX_HD'] = str(cv2.Rodrigues(R)[0][0][0])
    config['STEREO']['RZ_HD'] = str(cv2.Rodrigues(R)[0][2][0])

    # Запишите данные в файл
    with open(filename, 'w') as configfile:
        config.write(configfile)

def find_relative_position_and_orientation(img1, img2, K1, K2):
    # Определение детектора особенностей
    sift = cv2.SIFT_create()

    # Поиск особенностей и вычисление дескрипторов для двух изображений
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # Сопоставим характеристики на двух изображениях
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Применим отношение теста исключения по Лоу
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])

    # Извлеките сопоставленные точки
    pts1 = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
    pts2 = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

    # Найдем фундаментальную матрицу
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # Ищем эссенциальную матрицу 
    E = K1.T @ F @ K2

    # Переводим в единый формат
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K1)

    return R, t

def get_colibration(img_path, string, config = configparser.ConfigParser()):
    CHECKERBOARD = (10,6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    objpoints = []

    imgpoints = [] 

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    # Extracting path of individual image stored in a given directory
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
    

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    config.optionxform = str
    config[string] = {}
    # Параметры камеры
    config[string]['fx'] = str(mtx[0,0])
    config[string]['fy'] = str(mtx[1,1])
    config[string]['cx'] = str(mtx[0,2])
    config[string]['cy'] = str(mtx[1,2])

    # Коэффициенты искажения
    config[string]['k1'] = str(dist[0][0])
    config[string]['k2'] = str(dist[0][1])
    config[string]['p1'] = str(dist[0][2])
    config[string]['p2'] = str(dist[0][3])
    config[string]['k3'] = str(dist[0][4])
    return mtx, config

img1 = cv2.imread("images/chessboard.jpg")
k1, config1 = get_colibration("images/left.jpg", "LEFT_CAM_2K")

img2 = cv2.imread("images/IMG_20230721_150516.jpg")
k2, config2 = get_colibration("images/right.jpg", "RIGHT_CAM_2K", config1)



R, t = find_relative_position_and_orientation(img1, img2, k1, k2)
write_to_config(R, t, "config.conf", config2)