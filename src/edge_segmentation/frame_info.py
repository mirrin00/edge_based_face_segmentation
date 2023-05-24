import numpy as np
from .contour_info import ContourInfo


class FrameInfo:
    def __init__(self, origin_frame: "np.ndarray"):
        self.origin_frame: "np.ndarray" = origin_frame
        self.cur_frame: "np.ndarray" = origin_frame.copy()
        self.contours = []
        self.contours_info: list[ContourInfo] = []
        self.main_contour = None
        self.main_contour_mask = None
        self.mediapipe_points = []
        self.bounded_box = np.array([])
        self.number_of_face_points = 0
        self.result_mask = np.zeros((1, 1), dtype=np.uint8)
        self.result_points = []
        self.skip = False
        self.r_coeff = None
