import configparser
import cv2
import numpy as np
import os
import pickle


class OneCameraCalibration:
    """Тестовый класс для калибровки камеры."""

    def __init__(self):
        """Инициализирует калибровку камеры."""
        self._config = configparser.ConfigParser()
        self._config.optionxform = str
        self._R = []
        self._t = []

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
            if filename.endswith(".png"):  # вы можете указать другие форматы изображений
                print("find...")
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
                # Если найдено, добавьте точки объекта, точки изображения (после их уточнения)
                if ret:
                    objpoints.append(objp)
                    #corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners)

                    # Рисуем и отображаем углы
                    #cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
                    #cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                    #cv2.resizeWindow("img", 1600, 1200)
                    #cv2.imshow('img', img)
                    #cv2.waitKey(500)
                    print(f"find!!! {filename}")
                else:
                    print(f"Corners not found {filename}")

        cv2.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                           imgpoints,
                                                           gray.shape[::-1],
                                                           None,
                                                           None)
        print(dist)
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


        with open('points.pkl', 'wb') as f:
            pickle.dump((objpoints, imgpoints), f)

        with open("air_camera_config.conf", 'w') as configfile:
            self._config.write(configfile)

        self.get_angles(mtx, dist, objpoints, imgpoints)

        return mtx
    
    def get_angles(self, camera_matrix, dist_coeffs, object_points, image_points):

        object_points = np.array(object_points)
        image_points = np.array(image_points)
        object_points = np.float32(object_points)
        image_points = np.float32(image_points)

        #print(object_points)
        #print(image_points)

        #print(type(object_points[0]))
        #print(type(image_points[0]))

        # Пример кода для удаления искажения
        image = cv2.imread("air/frame17.png")  # Замените на ваше тестовое изображение
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

        if len(object_points) == len(image_points):
            ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

            # Ваш код ниже ...
        else:
            print("Не соответствующие точки для cv2.solvePnP()")

        # Пример кода для преобразования вектора вращения в матрицу вращения
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Пример кода для извлечения углов
        vertical_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        horizontal_angle = np.arctan2(rotation_matrix[2, 0], rotation_matrix[0, 0])

        # Перевод углов в градусы
        vertical_angle = np.degrees(vertical_angle)
        horizontal_angle = np.degrees(horizontal_angle)

        print(f"Vertical angle: {vertical_angle}")
        print(f"Horizontal angle: {horizontal_angle}")


def main():
    """Основная функция для выполнения калибровки камеры."""
    calibration = OneCameraCalibration()

    k1 = calibration.get_calibration("air", "FRONT_CAMERA")
    print(k1)


if __name__ == "__main__":
    main()
