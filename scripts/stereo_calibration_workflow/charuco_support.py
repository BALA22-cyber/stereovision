"""ChArUco detection and ID-based correspondence helpers."""
import cv2
import numpy as np


class CharucoTarget:
    def __init__(self, cols, rows, square_size, marker_size, dictionary, legacy=False, min_corners=6):
        if not hasattr(cv2, 'aruco') or not hasattr(cv2.aruco, 'CharucoDetector'):
            raise RuntimeError('ChArUco requires OpenCV contrib with aruco.CharucoDetector (OpenCV >= 4.8).')
        if not dictionary or not hasattr(cv2.aruco, dictionary) or not dictionary.startswith('DICT_'):
            raise ValueError('Provide a valid OpenCV ArUco dictionary, e.g. --dictionary DICT_4X4_1000.')
        if cols < 2 or rows < 2 or not (0 < marker_size < square_size) or min_corners < 4:
            raise ValueError('Board dimensions must be >= 2 inner corners, 0 < marker size < square size, and min corners >= 4.')
        marker_count = ((cols + 1) * (rows + 1)) // 2
        available = len(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)).bytesList)
        if marker_count > available:
            raise ValueError(f'This {cols + 1}x{rows + 1}-square OpenCV board needs {marker_count} distinct markers, but {dictionary} has only {available}. Check whether the board dimensions describe squares rather than inner corners, or whether a different dictionary/layout was used.')
        self.dictionary_name = dictionary
        self.min_corners = min_corners
        self.board = cv2.aruco.CharucoBoard((cols + 1, rows + 1), square_size, marker_size,
                                             cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)))
        self.board.setLegacyPattern(legacy)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.object_points = np.asarray(self.board.getChessboardCorners(), dtype=np.float32)

    def detect(self, image):
        corners, ids, _, _ = self.detector.detectBoard(image)
        if ids is None or corners is None or len(ids) < self.min_corners:
            return None
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(self.object_points)):
            return None
        if self.board.checkCharucoCornersCollinear(ids.reshape(-1, 1)):
            return None
        return corners, ids

    def objects(self, detection):
        return self.object_points[detection[1]].copy()

    def shared(self, left, right):
        if left is None or right is None:
            return None
        ids, li, ri = np.intersect1d(left[1], right[1], return_indices=True)
        if len(ids) < self.min_corners or self.board.checkCharucoCornersCollinear(ids.reshape(-1, 1)):
            return None
        return self.object_points[ids].copy(), left[0][li].copy(), right[0][ri].copy()
