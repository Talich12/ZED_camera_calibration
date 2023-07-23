import configparser
import cv2
import numpy as np
import os


class CameraCalibration:
    """Класс для калибровки ZED камеры."""

    def __init__(self):
        """Инициализирует калибровку камеры."""
        self._config = configparser.ConfigParser()
        self._config.optionxform = str
        self._R = []
        self._t = []

    def write_to_config(self, filename):
        """Записывает параметры калибровки в файл конфигурации."""
        self._config['STEREO'] = {}
        rot_vector = cv2.Rodrigues(self._R)[0]
        self._config['STEREO']['Baseline'] = str(np.linalg.norm(self._t))
        self._config['STEREO']['TY'] = str(self._t[1][0])
        self._config['STEREO']['TZ'] = str(self._t[2][0])
        self._config['STEREO']['CV_HD'] = str(rot_vector[1][0])
        self._config['STEREO']['RX_HD'] = str(rot_vector[0][0])
        self._config['STEREO']['RZ_HD'] = str(rot_vector[2][0])

        with open(filename, 'w') as configfile:
            self._config.write(configfile)

    def find_relative_position_and_orientation(self, img1, img2, K1, K2):
        """
        Находит относительное положение и ориентацию двух изображений.

        Параметры:
        img1, img2: Изображения, для которых нужно найти относительное положение и ориентацию.
        K1, K2: Матрицы камер для соответствующих изображений.

        """
        # Создаем объект SIFT для извлечения ключевых точек и дескрипторов
        sift = cv2.SIFT_create()
        # Извлекаем ключевые точки и дескрипторы для каждого изображения
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Создаем объект BFMatcher для сопоставления дескрипторов
        bf = cv2.BFMatcher()
        # Находим сопоставления между дескрипторами двух изображений
        matches = bf.knnMatch(des1, des2, k=2)

        # Отбираем хорошие сопоставления, используя отношение расстояний
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append([m])

        # Получаем координаты ключевых точек для хороших сопоставлений
        pts1 = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Находим фундаментальную матрицу
        F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        # Вычисляем матрицу существенности
        E = K1.T @ F @ K2
        # Восстанавливаем позу камеры
        _, self._R, self._t, _ = cv2.recoverPose(E, pts1, pts2, K1)

    def get_calibration(self, folder, string):
        """
        Получает параметры калибровки из изображений в указанной папке.

        Параметры:
        folder: Папка где находятся изображения.
        string: Строка которая будет являться объектом конфига

        """
        CHECKERBOARD = (10, 6)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objpoints = []  # 3d точка в реальном мире
        imgpoints = []  # 2d точки в плоскости изображения.

        objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        objp *= 4
        for filename in os.listdir(folder):
            if filename.endswith(".jpeg"):  # вы можете указать другие форматы изображений
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

                # Если найдено, добавьте точки объекта, точки изображения (после их уточнения)
                if ret:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners)

                    # Рисуем и отображаем углы
                    cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
                    # cv2.imshow('img', img)
                    # cv2.waitKey(500)

        cv2.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                           imgpoints,
                                                           gray.shape[::-1],
                                                           None,
                                                           None)

        self._config[string] = {}
        self._config[string]['fx'] = str(mtx[0, 0])
        self._config[string]['fy'] = str(mtx[1, 1])
        self._config[string]['cx'] = str(mtx[0, 2])
        self._config[string]['cy'] = str(mtx[1, 2])
        self._config[string]['k1'] = str(dist[0][0])
        self._config[string]['k2'] = str(dist[0][1])
        self._config[string]['p1'] = str(dist[0][2])
        self._config[string]['p2'] = str(dist[0][3])
        self._config[string]['k3'] = str(dist[0][4])

        return mtx


def main():
    """Основная функция для выполнения калибровки камеры."""
    calibration = CameraCalibration()

    img1 = cv2.imread("images/left.jpg")
    k1 = calibration.get_calibration("left", "LEFT_CAM_2K")

    img2 = cv2.imread("images/right.jpg")
    k2 = calibration.get_calibration("left_camera", "RIGHT_CAM_2K")

    calibration.find_relative_position_and_orientation(img1, img2, k1, k2)
    calibration.write_to_config("config.conf")


if __name__ == "__main__":
    main()
