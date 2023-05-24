import cv2
import numpy as np
import mediapipe
import logging
from enum import Enum, auto
from phycv import PST, PST_GPU
import torch
from .frame_info import FrameInfo
from .contour_info import ContourInfo


class ProcessingObject:

    def process_frame_info(self, frame_info: FrameInfo):
        pass


class Constants(Enum):
    OBJECT = auto()
    FACE = auto()
    NO_OVERLAP = auto()
    WITH_OVERLAP = auto()


class RemoveNoiseProcessingObject(ProcessingObject):

    def __init__(self, mediapipe_points_args={}, remove_noise_args={}):
        super().__init__()
        self.remove_noise_args = remove_noise_args

    def process_frame_info(self, frame_info: FrameInfo):
        self.remove_noise(frame_info, **self.remove_noise_args)

    def remove_noise(self, frame_info: FrameInfo, kernel_size=(5, 5)):
        denoised_image = frame_info.cur_frame
        ##################
        kernel_close = np.ones(kernel_size, np.uint8)
        kernel_open = np.ones(kernel_size, np.uint8)
        denoised_image = cv2.morphologyEx(denoised_image, cv2.MORPH_OPEN, kernel_open)
        denoised_image = cv2.morphologyEx(denoised_image, cv2.MORPH_CLOSE, kernel_close)
        frame_info.cur_frame = frame_info.origin_frame = denoised_image


class MediapipeProcessingObject(ProcessingObject):

    def __init__(self, mediapipe_points_args={}):
        super().__init__()
        self.mediapipe_points_args = mediapipe_points_args
        self.face_detector = mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def process_frame_info(self, frame_info: FrameInfo):
        self.get_mediapipe_points(frame_info, **self.mediapipe_points_args)

    def get_mediapipe_points(self, frame_info: FrameInfo, offsets=np.zeros((4,)), coeff=1) -> list:
        image: "np.ndarray" = frame_info.cur_frame
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = self.face_detector.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True
        points = []
        h, w = image.shape[:2]
        if res.multi_face_landmarks:
            for face_landmarks in res.multi_face_landmarks:
                for ln in face_landmarks.landmark:
                    points.append((
                        round(ln.x * w),
                        round(ln.y * h),
                    ))
        else:
            frame_info.mediapipe_points, frame_info.bounded_box = None, None
            frame_info.skip = True
            logging.warning("No face on frame")
            return
        bounded_box = np.array([
            min(points, key=lambda x: x[1])[1],  # min_y
            max(points, key=lambda x: x[1])[1],  # max_y
            min(points, key=lambda x: x[0])[0],  # min_x
            max(points, key=lambda x: x[0])[0],  # max_x
        ])
        bounded_box += coeff * offsets
        bounded_box[0] = max(bounded_box[0], 0)
        bounded_box[2] = max(bounded_box[2], 0)
        bounded_box[1] = min(bounded_box[1], h - 1)
        bounded_box[3] = min(bounded_box[3], w - 1)
        frame_info.mediapipe_points, frame_info.bounded_box = points, bounded_box
        min_y, max_y, min_x, max_x = bounded_box
        frame_info.cur_frame = frame_info.cur_frame[min_y:max_y, min_x:max_x]
        frame_info.origin_frame = frame_info.origin_frame[min_y:max_y, min_x:max_x]


class PhyCVProcessingObject(ProcessingObject):

    def __init__(self, phycv_params: dict, gpu_device: str or None = None, args={}):
        super().__init__()
        self.pst = PST() if gpu_device is None else PST_GPU(torch.device(gpu_device))
        self.pst_init_params = {
            "S": phycv_params["S"],
            "W": phycv_params["W"],
        }
        self.pst_apply_params = {
            "sigma_LPF": phycv_params["sigma_LPF"],
            "thresh_min": phycv_params["thresh_min"],
            "thresh_max": phycv_params["thresh_max"],
            "morph_flag": phycv_params["morph_flag"],
        }
        self.other_args = args

    def process_frame_info(self, frame_info: FrameInfo):
        self.process_with_phycv(frame_info, self.pst_init_params, self.pst_apply_params, **self.other_args)

    def process_with_phycv(self, frame_info: FrameInfo, init_params={}, apply_params={}, resize_func=None):
        image: "np.ndarray" = frame_info.cur_frame
        origin_size = None
        if resize_func is not None:
            origin_size = image.shape[:2][::-1]
            image = resize_func(image)
        if type(self.pst) is PST_GPU:
            image = torch.from_numpy(image)
            image = torch.permute(image, (2, 0, 1))
            self.pst.h = self.pst.w = None
        else:
            self.pst.h = image.shape[0]
            self.pst.w = image.shape[1]
        self.pst.load_img(img_array=image)
        self.pst.init_kernel(**init_params)
        self.pst.apply_kernel(**apply_params)
        res = self.pst.pst_output
        if type(self.pst) is PST_GPU:
            res = res.cpu().numpy()
        res *= 255
        res = res.astype(np.uint8)
        if origin_size is not None:
            res = cv2.resize(res, origin_size)
        frame_info.cur_frame = res


class MorphProcessingObject(ProcessingObject):

    def __init__(self, morph_args={}):
        super().__init__()
        self.morph_args = morph_args

    def process_frame_info(self, frame_info: FrameInfo):
        self.morph(frame_info, **self.morph_args)

    def morph(self, frame_info: FrameInfo, kernel_size=(5, 5), structure_element=cv2.MORPH_ELLIPSE):
        new_img: "np.ndarray" = frame_info.cur_frame
        kernel = cv2.getStructuringElement(structure_element, kernel_size)
        new_img = cv2.morphologyEx(new_img, cv2.MORPH_DILATE, kernel)
        frame_info.cur_frame = ~new_img


class WatershedProcessingObject(ProcessingObject):

    def __init__(self, watershed_args={}):
        super().__init__()
        self.watershed_args = watershed_args

    def process_frame_info(self, frame_info: FrameInfo):
        self.watershed(frame_info, **self.watershed_args)

    def watershed(self, frame_info: FrameInfo, cnt_len_coeff: float = 0.05,
                  erode_kernel_size=(4, 8), iterations=2, use_distance=False):
        img_shape = frame_info.cur_frame.shape[:2]
        # In frame_info.cur_frame are edges with value 255, invert them and
        # erode to get contours -- markers for watershed
        if use_distance:
            dst = cv2.distanceTransform(frame_info.cur_frame, cv2.DIST_L2, 3).astype(np.uint16)
            _, res = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
            res = res.astype(np.uint8)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, erode_kernel_size)
            res = cv2.erode(frame_info.cur_frame, kernel, iterations=iterations)
        contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Remove small contours
        min_cnt_len = 2 * (img_shape[0] + img_shape[1]) * cnt_len_coeff
        contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_cnt_len]
        markers = np.zeros(img_shape, dtype=np.int32)
        for i in range(len(contours)):
            cv2.drawContours(markers, contours, i, (i + 1), -1)
        # Run watershed
        denoised_image = frame_info.origin_frame.copy()
        edge_img = frame_info.cur_frame
        denoised_image[edge_img == 255] = np.array([0, 0, 0], dtype=np.uint8)
        cv2.watershed(denoised_image, markers)
        # Create new edges and dilate them
        new_edges = np.zeros(img_shape, dtype=np.uint8)
        new_edges[markers == -1] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        new_edges = cv2.morphologyEx(new_edges, cv2.MORPH_DILATE, kernel)
        frame_info.cur_frame[new_edges == 255] = 0


class FindCountoursProcessingObject(ProcessingObject):

    def __init__(self, contours_args={}):
        super().__init__()
        self.contours_args = contours_args

    def process_frame_info(self, frame_info: FrameInfo):
        self.find_contours(frame_info, **self.contours_args)

    def find_contours(self, frame_info: FrameInfo, cnt_len_coeff=1.0):
        image: "np.ndarray" = frame_info.cur_frame
        contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # TODO: Is this nessecary?
        if hierarchy is None:
            hierarchy = [[]]
        contours = [contours[i] for i, info in enumerate(hierarchy[0]) if info[3] < 0]
        lengths = [(i, cv2.contourArea(c)) for i, c in enumerate(contours)]
        min_cnt_len = image.shape[0] * cnt_len_coeff
        lengths = [length for length in lengths if length[1] > min_cnt_len]
        frame_info.contours = [contours[i] for i, _ in sorted(lengths, key=lambda x: x[1], reverse=True)]


class FindMainContourProcessingObject(ProcessingObject):

    def __init__(self, args={}):
        super().__init__()
        self.args = args

    def process_frame_info(self, frame_info: FrameInfo):
        self.find_main_contour(frame_info, **self.args)

    def find_main_contour(self, frame_info: FrameInfo, silhouette_points: list = []):
        min_x, min_y = frame_info.bounded_box[2], frame_info.bounded_box[0]
        points = frame_info.mediapipe_points
        # Points are coordinates on origin uncutted frame, so shift them to cutted area
        oval = [(points[i][0] - min_x, points[i][1] - min_y) for i in silhouette_points]
        oval = np.array(oval)
        frame_info.main_contour = oval.reshape((-1, 1, 2))
        main_contour_mask = np.zeros(frame_info.cur_frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(main_contour_mask, [oval], -1, 255, -1)
        frame_info.main_contour_mask = main_contour_mask


class CreateContourInfoProcessingObject(ProcessingObject):

    def __init__(self, contours_args={}):
        super().__init__()
        self.contours_args = contours_args

    def process_frame_info(self, frame_info: FrameInfo):
        self.create_contours_info(frame_info, **self.contours_args)

    def create_contours_info(self, frame_info: FrameInfo):
        shape = frame_info.cur_frame.shape[:2]
        frame_info.contours_info = [ContourInfo(cnt, shape) for cnt in frame_info.contours]
        img = np.zeros(frame_info.cur_frame.shape[:2], dtype=np.int32)
        # Replace with masks from contours_info?
        cnts = frame_info.contours
        for i in range(len(cnts)):
            cv2.drawContours(img, cnts, i, i + 1, -1)
        min_x, min_y = frame_info.bounded_box[[2, 0]]
        h, w = img.shape
        for p in frame_info.mediapipe_points:
            x = p[0] - min_x
            y = p[1] - min_y
            if 0 <= x < w and 0 <= y < h:
                cnt_index = img[y, x]
                if cnt_index != 0:
                    frame_info.contours_info[cnt_index - 1].number_of_mp_points += 1


class AnalyzeContoursProcessingObject(ProcessingObject):

    def __init__(self, compare_method=Constants.NO_OVERLAP, compare_args={}, mask_args={}):
        super().__init__()
        self.compare_args = compare_args
        self.mask_args = mask_args
        self.compare_method = compare_method
        self.prev_contours_info = None
        self.prev_number_of_points = None
        self.prev_mask = None

    def process_frame_info(self, frame_info: FrameInfo):
        self.compare_contours_with_prev(frame_info, **self.compare_args)
        self.create_mask(frame_info, **self.mask_args)

    def compare_contours_with_prev(self, frame_info: FrameInfo, max_val=15,
                                   min_mp_points: int = 10, max_coeff: float = 5):
        # SavedState must be on one FunctionWrapper with get_mask_from_contours()!
        if self.prev_contours_info is None:
            self.prev_contours_info = frame_info.contours_info
            return
        last_contours_info = self.prev_contours_info
        new_shape = frame_info.cur_frame.shape[:2]
        for cnt in last_contours_info:
            cnt.resize_contour(new_shape)

        for i, contour_info in enumerate(frame_info.contours_info):
            min_index, result = min(
                enumerate(
                    [self.compare_contours(contour_info, cnt) for cnt in last_contours_info]
                ), key=lambda x: x[1]
            )
            last_cnt = last_contours_info[min_index]
            # FIXME: Is it needed?
            mp_points_result = contour_info.number_of_mp_points < min_mp_points or \
                last_cnt.number_of_mp_points < min_mp_points or \
                contour_info.number_of_mp_points / last_cnt.number_of_mp_points < max_coeff
            if max_val is None or (result < max_val and mp_points_result):
                contour_info.mark = last_cnt.mark

    def compare_contours(self, cnt1: ContourInfo, cnt2: ContourInfo) -> float:
        if self.compare_method == Constants.NO_OVERLAP:
            return cnt1.compare_without_overlap(cnt2)
        else:
            return cnt1.compare_with_overlap(cnt2)

    def create_mask(self, frame_info: FrameInfo, mask_type=Constants.OBJECT,
                    default_eps=0.06, max_diff: float = 2):
        eps = default_eps
        img_shape = frame_info.cur_frame.shape[:2]
        mp_area = cv2.countNonZero(frame_info.main_contour_mask)
        mp_mask = np.where(frame_info.main_contour_mask == 255, True, False)
        mp_square = cv2.contourArea(frame_info.main_contour)
        if mask_type == Constants.FACE:
            res_mask = np.zeros(img_shape, dtype=np.uint8)
        else:  # mask_type == Constants.OBJECT
            res_mask = frame_info.main_contour_mask.copy()
        contours = frame_info.contours_info
        for i, cnt in enumerate(contours):
            is_object = False
            # If segment is not matched with previous
            if cnt.mark is None:
                # if segment area takes more than 1/3 of main contour -- increase eps
                cnt_area_count = np.count_nonzero(cnt.contour_mask[mp_mask])
                if cnt_area_count > mp_area / 3 and cnt.square > mp_square * 2 / 3:
                    eps = 0.5
                else:
                    eps = default_eps
                # Check that segment match main contour with eps
                cnt_not_matched = np.count_nonzero(cnt.contour_mask[~mp_mask])
                # FIXME: Divide by contour area?
                if cnt_not_matched / mp_area > eps:
                    is_object = True
                cnt.mark = Constants.OBJECT if is_object else Constants.FACE
            else:  # use info from previous frame
                is_object = cnt.mark == Constants.OBJECT
            # Add points from segments to current frame counter
            if not is_object:
                frame_info.number_of_face_points += cnt.number_of_mp_points
            if mask_type == Constants.OBJECT and is_object:
                res_mask[cnt.contour_mask == 255] = 0
            elif mask_type == Constants.FACE and not is_object:
                res_mask |= cnt.contour_mask
        # Check that points are not change dramatically
        if self.prev_number_of_points is not None:
            points_cmp = max(self.prev_number_of_points, frame_info.number_of_face_points) / \
                max(1, min(self.prev_number_of_points, frame_info.number_of_face_points))
        else:
            points_cmp = -1
        if points_cmp > max_diff:
            logging.warning("Face is OBJECT, using prev mask and contours")
            res_mask = self.prev_mask.copy()
        else:
            self.prev_mask = res_mask.copy()
            self.prev_number_of_points = frame_info.number_of_face_points
        frame_info.result_mask = res_mask


class CreateResultPointsInfoProcessingObject(ProcessingObject):

    def __init__(self, bad_point_coords=(-10000, -10000)):
        super().__init__()
        self.bad_point_coords = bad_point_coords

    def process_frame_info(self, frame_info: FrameInfo):
        self.create_result_points(frame_info)

    def create_result_points(self, frame_info: FrameInfo):
        bb = frame_info.bounded_box
        r_coeff = frame_info.r_coeff
        h, w = frame_info.result_mask.shape[:2]
        for p in frame_info.mediapipe_points:
            x = p[0] - bb[2]
            y = p[1] - bb[0]
            if 0 <= x < w and 0 <= y < h and frame_info.result_mask[y, x] != 0:
                if r_coeff is not None:
                    p = (int(p[1] / r_coeff), int(p[0] / r_coeff))
                else:
                    p = (p[1], p[0])
                frame_info.result_points.append(p)
            else:
                frame_info.result_points.append(self.bad_point_coords)


class ResizeProcessingObject(ProcessingObject):

    def __init__(self, new_x_size=None):
        super().__init__()
        self.x_size = new_x_size

    def process_frame_info(self, frame_info: FrameInfo):
        self.resize_frame(frame_info)

    def resize_frame(self, frame_info: FrameInfo):
        if self.x_size is None:
            return
        h, w = frame_info.cur_frame.shape[:2]
        if w <= self.x_size:
            return
        r_coeff = self.x_size / w
        const_size = (int(self.x_size), int(r_coeff * h))
        frame_info.cur_frame = cv2.resize(frame_info.cur_frame, const_size)
        frame_info.origin_frame = cv2.resize(frame_info.origin_frame, const_size)
        frame_info.mediapipe_points = [(int(p[0] * r_coeff), int(p[1] * r_coeff)) for p in frame_info.mediapipe_points]
        frame_info.bounded_box = (frame_info.bounded_box * r_coeff).astype(np.int64)
        frame_info.r_coeff = r_coeff
