import cv2
import numpy as np


class ContourInfo:

    weights_no_overlap = [
        0.1,  # length
        0.3,  # square
        0.6,  # center
    ]

    assert sum(weights_no_overlap) == 1

    weights_with_overlap = [
        0.0,  # length
        0.1,  # square
        0.1,  # center
        0.8,  # overlap
    ]

    assert sum(weights_with_overlap) == 1

    def __init__(self, contour, shape):
        self.contour = contour
        self.length = cv2.arcLength(contour, True)
        self.square = cv2.contourArea(contour)
        self.center = np.mean(contour, axis=0).reshape(-1)
        assert len(self.center) == 2
        self.shape = shape
        self.contour_mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(self.contour_mask, [contour], -1, 255, -1)
        self.non_zero_pixels = np.count_nonzero(self.contour_mask)
        self.mark = None
        self.number_of_mp_points = 0

    def resize_contour(self, new_shape):
        self.contour_mask = cv2.resize(self.contour_mask, new_shape[::-1])
        self.shape = new_shape

    def __calculate_params(self, cnt: "ContourInfo") -> list:
        params = [
            1 - min(self.length, cnt.length) / max(self.length, cnt.length),
            1 - min(self.square, cnt.square) / max(self.square, cnt.square),
            min(100, np.linalg.norm(self.center - cnt.center)) / 100
        ]
        return params

    def compare_without_overlap(self, cnt: "ContourInfo") -> float:
        params = self.__calculate_params(cnt)
        result = 0.0
        for w, p in zip(self.weights_no_overlap, params):
            result += w * p
        return result * 100

    def compare_with_overlap(self, cnt: "ContourInfo") -> float:
        params = self.__calculate_params(cnt)
        if self.shape == cnt.shape:
            overlap = self.contour_mask ^ cnt.contour_mask
        else:
            overlap = cnt.contour_mask ^ cv2.resize(self.contour_mask, cnt.shape[::-1])
        params.append(1 - max(1, np.count_nonzero(overlap) / min(self.non_zero_pixels, cnt.non_zero_pixels)))
        result = 0.0
        for w, p in zip(self.weights_with_overlap, params):
            result += w * p
        return result * 100
