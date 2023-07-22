import configparser
import cv2
import numpy as np


class CameraCalibration:

    def __init__(self):
        self._config = configparser.ConfigParser()
        self._config.optionxform = str
        self._R = []
        self._t = []

    def write_to_config(self, filename):
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
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append([m])

        pts1 = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

        F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        E = K1.T @ F @ K2
        _, self._R, self._t, _ = cv2.recoverPose(E, pts1, pts2, K1)

    def get_calibration(self, img_path, string):
        CHECKERBOARD = (10, 6)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

        objpoints = []
        imgpoints = []

        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints,
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
    calibration = CameraCalibration()

    img1 = cv2.imread("images/left.jpg")
    k1 = calibration.get_calibration("images/left.jpg", "LEFT_CAM_2K")

    img2 = cv2.imread("images/right.jpg")
    k2 = calibration.get_calibration("images/right.jpg", "RIGHT_CAM_2K")

    calibration.find_relative_position_and_orientation(img1, img2, k1, k2)
    calibration.write_to_config("config.conf")


if __name__ == "__main__":
    main()
