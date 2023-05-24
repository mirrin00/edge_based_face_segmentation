import cv2
import numpy as np
from phycv import PST, PST_GPU
import torch
from enum import Enum, auto
import logging
from .frame_info import FrameInfo
from .contour_info import ContourInfo


class SavedState:
    def __init__(self):
        self.prev_contours_info = None
        self.prev_number_of_points = None
        self.prev_mask = None


class Constants(Enum):
    OBJECT = auto()
    FACE = auto()
    NO_OVERLAP = auto()
    WITH_OVERLAP = auto()


def get_mediapipe_points(frame_info: FrameInfo, saved_state: SavedState, face_detector=None, offsets=np.zeros((4,)), coeff=1) -> list:
    if face_detector is None:
        frame_info.mediapipe_points, frame_info.bounded_box = None, None
        frame_info.skip = True
        return
    image: "np.ndarray" = frame_info.cur_frame
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = face_detector.process(image)
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


def preprocessing_remove_noise(frame_info: FrameInfo, saved_state: SavedState, kernel_size=(5, 5)):
    denoised_image = frame_info.cur_frame
    ##################
    kernel_close = np.ones(kernel_size, np.uint8)
    kernel_open = np.ones(kernel_size, np.uint8)
    denoised_image = cv2.morphologyEx(denoised_image, cv2.MORPH_OPEN, kernel_open)
    denoised_image = cv2.morphologyEx(denoised_image, cv2.MORPH_CLOSE, kernel_close)
    frame_info.cur_frame = frame_info.origin_frame = denoised_image


def process_with_phycv(frame_info: FrameInfo, saved_state: SavedState, pst: PST or PST_GPU, init_params={}, apply_params={}, resize_func=None):
    image: "np.ndarray" = frame_info.cur_frame
    origin_size = None
    if resize_func is not None:
        origin_size = image.shape[:2][::-1]
        image = resize_func(image)
    if type(pst) is PST_GPU:
        image = torch.from_numpy(image)
        image = torch.permute(image, (2, 0, 1))
        pst.h = pst.w = None
    else:
        pst.h = image.shape[0]
        pst.w = image.shape[1]
    pst.load_img(img_array=image)
    pst.init_kernel(**init_params)
    pst.apply_kernel(**apply_params)
    res = pst.pst_output
    if type(pst) is PST_GPU:
        res = res.cpu().numpy()
    res *= 255
    res = res.astype(np.uint8)
    if origin_size is not None:
        res = cv2.resize(res, origin_size)
    frame_info.cur_frame = res


def apply_morph(frame_info: FrameInfo, saved_state: SavedState, kernel_size=(5, 5), structure_element=cv2.MORPH_ELLIPSE):
    new_img: "np.ndarray" = frame_info.cur_frame
    kernel = cv2.getStructuringElement(structure_element, kernel_size)
    new_img = cv2.morphologyEx(new_img, cv2.MORPH_DILATE, kernel)
    frame_info.cur_frame = ~new_img


def watershed(frame_info: FrameInfo, saved_state: SavedState,
              cnt_len_coeff: float = 0.05, erode_kernel_size=(4, 8)):
    img_shape = frame_info.cur_frame.shape[:2]
    # In frame_info.cur_frame are edges with value 255, invert them and
    # erode to get contours -- markers for watershed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, erode_kernel_size)
    res = cv2.erode(~frame_info.cur_frame, kernel, iterations=2)
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
    edge_img ^= new_edges
    frame_info.cur_frame = new_edges


def draw_contours(frame_info: FrameInfo, saved_state: SavedState, cnt_len_coeff=1.0):
    image: "np.ndarray" = frame_info.cur_frame
    contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    lengths = [(i, cv2.contourArea(c)) for i, c in enumerate(contours)]
    min_cnt_len = image.shape[0] * cnt_len_coeff
    lengths = [length for length in lengths if length[1] > min_cnt_len]
    frame_info.contours = [contours[i] for i, _ in sorted(lengths, key=lambda x: x[1], reverse=True)]


def get_cont_from_mp(frame_info: FrameInfo, saved_state: SavedState, silhouette_points: list):
    min_x, min_y = frame_info.bounded_box[2], frame_info.bounded_box[0]
    points = frame_info.mediapipe_points
    # Points are coordinates on origin uncutted frame, so shift them to cutted area
    oval = [(points[i][0] - min_x, points[i][1] - min_y) for i in silhouette_points]
    oval = np.array(oval)
    frame_info.main_contour = oval.reshape((-1, 1, 2))
    main_contour_mask = np.zeros(frame_info.cur_frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(main_contour_mask, [oval], -1, 255, -1)
    frame_info.main_contour_mask = main_contour_mask


def create_contours_info(frame_info: FrameInfo, saved_state: SavedState):
    shape = frame_info.cur_frame.shape[:2]
    frame_info.contours_info = [ContourInfo(cnt, shape) for cnt in frame_info.contours]
    img = np.zeros(frame_info.cur_frame.shape[:2], dtype=np.int32)
    # Replace with masks from contours_info?
    cnts = frame_info.contours
    for i in range(len(cnts)):
        cv2.drawContours(img, cnts, i, i + 1, -1)
    min_x, min_y = frame_info.bounded_box[[2, 0]]
    for p in frame_info.mediapipe_points:
        x = p[0] - min_x
        y = p[1] - min_y
        cnt_index = img[y, x]
        if cnt_index != 0:
            frame_info.contours_info[cnt_index - 1].number_of_mp_points += 1


def compare_contours_with_prev(frame_info: FrameInfo, saved_state: SavedState, max_val=15,
                               compare_method=Constants.NO_OVERLAP, min_mp_points: int = 10,
                               max_coeff: float = 5):
    # SavedState must be on one FunctionWrapper with get_mask_from_contours()!
    if saved_state.prev_contours_info is None:
        saved_state.prev_contours_info = frame_info.contours_info
        return
    last_contours_info = saved_state.prev_contours_info
    new_shape = frame_info.cur_frame.shape[:2]
    for cnt in last_contours_info:
        cnt.resize_contour(new_shape)

    def compare_contours(cnt1: ContourInfo, cnt2: ContourInfo) -> float:
        if compare_method == Constants.NO_OVERLAP:
            return cnt1.compare_without_overlap(cnt2)
        else:
            return cnt1.compare_with_overlap(cnt2)

    for i, contour_info in enumerate(frame_info.contours_info):
        min_index, result = min(
            enumerate(
                [compare_contours(contour_info, cnt) for cnt in last_contours_info]
            ), key=lambda x: x[1]
        )
        last_cnt = last_contours_info[min_index]
        # FIXME: Is it needed?
        mp_points_result = contour_info.number_of_mp_points < min_mp_points or \
            last_cnt.number_of_mp_points < min_mp_points or \
            contour_info.number_of_mp_points / last_cnt.number_of_mp_points < max_coeff
        if max_val is None or (result < max_val and mp_points_result):
            contour_info.mark = last_cnt.mark


def get_mask_from_contours(frame_info: FrameInfo, saved_state: SavedState,
                           mask_type=Constants.OBJECT, default_eps=0.06,
                           max_diff: float = 2):
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
    if saved_state.prev_number_of_points is not None:
        points_cmp = max(saved_state.prev_number_of_points, frame_info.number_of_face_points) / \
            max(1, min(saved_state.prev_number_of_points, frame_info.number_of_face_points))
    else:
        points_cmp = -1
    if points_cmp > max_diff:
        logging.warning("Face is OBJECT, using prev mask and contours")
        res_mask = saved_state.prev_mask.copy()
    else:
        saved_state.prev_mask = res_mask.copy()
        saved_state.prev_number_of_points = frame_info.number_of_face_points
    frame_info.result_mask = res_mask


def create_result_points(frame_info: FrameInfo, saved_state: SavedState):
    bb = frame_info.bounded_box
    for p in frame_info.mediapipe_points:
        x = p[0] - bb[2]
        y = p[1] - bb[0]
        if frame_info.result_mask[y, x] != 0:
            frame_info.result_points.append(p)
