import numpy as np
from multiprocessing import Queue
from queue import Queue as ThreadQueue
from collections import deque
import logging
from .frame_info import FrameInfo
from .processing_object_wrapper import ProcessingObjectWrapper
from ..chunk import ReaderChunk
from .processing_objects import Constants, MediapipeProcessingObject, RemoveNoiseProcessingObject, \
    PhyCVProcessingObject, MorphProcessingObject, WatershedProcessingObject, FindCountoursProcessingObject, \
    FindMainContourProcessingObject, CreateContourInfoProcessingObject, CreateResultPointsInfoProcessingObject, \
    AnalyzeContoursProcessingObject, ResizeProcessingObject
# import classes.edge_segmentation.processing_functions as pf


class ContourSegmentor:

    silhouette = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]

    colors = [
        [184, 126, 55],   # 377eb8
        [0, 127, 255],    # ff7f00
        [74, 175, 77],    # 4daf4a
        [163, 78, 152],   # 984ea3
        [153, 153, 153],  # 999999
        [28, 26, 228],    # e41a1c
        [0, 222, 222],    # dede00
        [191, 129, 247],  # f781bf
        [40, 86, 166],    # a65628
    ]

    default_offset = np.array([-10, 10, -10, 10])
    default_offset_coeff = 4

    def __init__(self, chunk_queue: ThreadQueue or None, output_landmarks_queue: ThreadQueue or None,
                 use_processes: bool = False, gpu_device: str or None = None, maxsize=1, chunk_size=30):
        self.functions = []
        # self.pst = phycv.PST() if gpu_device is None else phycv.PST_GPU(torch.device(gpu_device))
        self.gpu_device = gpu_device
        self.is_process = use_processes
        self.pst_all_params = {
            "S": 0.4,
            "W": 20,
            "sigma_LPF": 0.1,
            "thresh_min": 0.0,
            "thresh_max": 0.8,
            "morph_flag": 1,
        }
        self.max_queue_size = maxsize
        self.chunk_size = chunk_size
        self.chunk_queue = chunk_queue
        self.output_landmarks = output_landmarks_queue
        self.start_queue = Queue(maxsize=maxsize) if self.is_process else ThreadQueue(maxsize=maxsize)
        self.output_queue = Queue(maxsize=maxsize) if self.is_process else ThreadQueue(maxsize=maxsize)
        self.pst_init_params = {
            "S": self.pst_all_params["S"],
            "W": self.pst_all_params["W"],
        }
        self.pst_apply_params = {
            "sigma_LPF": self.pst_all_params["sigma_LPF"],
            "thresh_min": self.pst_all_params["thresh_min"],
            "thresh_max": self.pst_all_params["thresh_max"],
            "morph_flag": self.pst_all_params["morph_flag"],
        }
        self.functions = []
        self.init_functions()

    def __put_to_queue(self, data):
        self.start_queue.put(data)

    def __get_from_queue(self):
        return self.output_queue.get()

    def process_queue(self):
        output_lms = deque(maxlen=self.chunk_size)
        self.run()
        if self.chunk_queue is None:
            return
        while True:
            chunk = self.chunk_queue.get()
            if chunk is None:
                self.chunk_queue.task_done()
                self.stop()
                break
            try:
                landmarks = self.process_chunk(chunk)
            except Exception as e:
                logging.error(f"Error during segmentation: {e}")
                self.chunk_queue.task_done()
                if self.output_landmarks is not None:
                    self.output_landmarks.put(None)
                continue
            output_lms.extend(landmarks)
            if self.output_landmarks is not None:
                self.output_landmarks.put(list(output_lms))
            self.chunk_queue.task_done()

    def process_chunk(self, chunk: ReaderChunk) -> list[tuple[int, int]]:
        chunk_len = len(chunk.frames_list)
        start_range = min(self.max_queue_size, chunk_len)
        result_points = []
        for i in range(start_range):
            frame_info = FrameInfo(chunk.frames_list[i])
            # frame_info.skip = True
            self.__put_to_queue(frame_info)
        for i in range(start_range, chunk_len):
            frame_info = FrameInfo(chunk.frames_list[i])
            # frame_info.skip = True
            self.__put_to_queue(frame_info)
            frame_info: FrameInfo = self.__get_from_queue()
            result_points.append(frame_info.result_points)
        for i in range(start_range):
            frame_info: FrameInfo = self.__get_from_queue()
            result_points.append(frame_info.result_points)
        return result_points

    def init_functions(self):
        self.functions = [
            ProcessingObjectWrapper("mediapipe", [
                (MediapipeProcessingObject, {
                    "mediapipe_points_args": {"offsets": self.default_offset, "coeff": self.default_offset_coeff}
                }),
                (RemoveNoiseProcessingObject, {}),
                (ResizeProcessingObject, {"new_x_size": 300})
            ], is_process=self.is_process, maxlen=self.max_queue_size),
            ProcessingObjectWrapper("phycv", [
                (PhyCVProcessingObject, {"phycv_params": self.pst_all_params, "gpu_device": self.gpu_device}),
            ], is_process=self.is_process, maxlen=self.max_queue_size),
            ProcessingObjectWrapper("watershed+contours", [
                (MorphProcessingObject, {}),
                (WatershedProcessingObject, {}),
                (FindCountoursProcessingObject, {}),
                (FindMainContourProcessingObject, {"args": {"silhouette_points": self.silhouette}}),
            ], is_process=self.is_process, maxlen=self.max_queue_size),
            ProcessingObjectWrapper("contour analysis", [
                (CreateContourInfoProcessingObject, {}),
                (AnalyzeContoursProcessingObject, {
                    "compare_method": Constants.WITH_OVERLAP,
                    "compare_args": {"max_val": 15},
                    "mask_args": {"mask_type": Constants.FACE}
                }),
                (CreateResultPointsInfoProcessingObject, {})
            ], is_process=self.is_process, maxlen=self.max_queue_size),
        ]
        # self.functions = [
        #     FunctionWrapper("mediapipe", funcs=[
        #         (pf.get_mediapipe_points, {"face_detector": self.face_detector, "offsets": self.default_offset, "coeff": self.default_offset_coeff}),
        #         (pf.preprocessing_remove_noise, {}),
        #     ], is_process=self.is_process, maxlen=self.max_queue_size),
        #     FunctionWrapper("phycv", funcs=[
        #         (pf.process_with_phycv, {"pst": self.pst, "init_params": self.pst_init_params, "apply_params": self.pst_apply_params})
        #     ], is_process=self.is_process, maxlen=self.max_queue_size),
        #     FunctionWrapper("watershed+contours", funcs=[
        #         (pf.watershed, {}),
        #         (pf.apply_morph, {}),
        #         (pf.draw_contours, {}),
        #         (pf.get_cont_from_mp, {"silhouette_points": self.silhouette}),
        #     ], is_process=self.is_process, maxlen=self.max_queue_size),
        #     FunctionWrapper("contour analysis", funcs=[
        #         (pf.create_contours_info, {}),
        #         (pf.compare_contours_with_prev, {"max_val": 15, "compare_method": pf.Constants.WITH_OVERLAP}),
        #         (pf.get_mask_from_contours, {"mask_type": pf.Constants.FACE}),
        #         (pf.create_result_points, {}),
        #     ], is_process=self.is_process, maxlen=self.max_queue_size),
        # ]
        self.functions[0].input_queue = self.start_queue
        for i in range(1, len(self.functions)):
            self.functions[i].input_queue = self.functions[i - 1].output_queue
        self.output_queue = self.functions[-1].output_queue

    def run(self):
        for func in self.functions:
            func.start()

    def stop(self):
        for func in self.functions:
            func.stop()

    def __del__(self):
        self.stop()
