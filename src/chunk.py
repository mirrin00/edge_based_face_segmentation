from typing import Tuple
import numpy as np
import logging


class ReaderChunk:
    def __init__(self, frames_list: list, frame_rate: float):
        if not isinstance(frames_list, list):
            raise TypeError(
                'Frames list is not a list. Frames list type: {}'.format(type(frames_list)))
        if len(frames_list) == 0:
            logging.warning('Creating empty reader chunk')
        if not isinstance(frame_rate, float):
            raise TypeError(
                'Frame rate type is not float. Frame rate type: {}'.format(type(frame_rate)))
        if frame_rate <= 0:
            raise ValueError(
                'Frame rate value is not positive. Frame rate value: {}'.format(frame_rate))

        self.frame_rate = frame_rate
        self.frames_list = [el[0] for el in frames_list]
        self.timestamps = [el[1] for el in frames_list]


class PreprocessedChunk:
    def __init__(self, reader_chunk: ReaderChunk, landmarks: 'list[Tuple[int, int]]',
                 width: int, height: int):
        if not isinstance(reader_chunk, ReaderChunk):
            raise TypeError('reader_chunk is not an instance of ReaderChunk. reader_chunk type: {}'
                            .format(type(reader_chunk)))
        if not isinstance(width, int):
            raise TypeError(
                'Width type is not int. Width type: {}'.format(type(width)))
        if not isinstance(height, int):
            raise TypeError(
                'Height type is not int. Height type: {}'.format(type(height)))
        self.landmarks = landmarks
        self.reader_chunk = reader_chunk
        self.width = width
        self.height = height


class RenderedChunk:
    def __init__(self, rendered_frames: 'np.ndarray[float]', rendered_timestamps: list[float], hrm_function: 'list[Tuple]',
                 landmarks: 'list[Tuple[int, int]]', points: dict, points_functions: dict, measured_pulse: dict,
                 real_pulse: int, peaks_timestamps: dict, clusters: dict):
        self.rendered_frames = rendered_frames
        self.timestamps = rendered_timestamps
        self.hrm_function = hrm_function
        self.landmarks = landmarks
        self.points = points
        self.points_functions = points_functions
        self.measured_pulse = measured_pulse
        self.real_pulse = real_pulse
        self.peaks_timestamps = peaks_timestamps
        self.clusters = clusters
