from multiprocessing import Process, Queue
from queue import Empty
from queue import Queue as ThreadQueue
from threading import Thread
from typing import Type
import logging
from .processing_objects import ProcessingObject
from .frame_info import FrameInfo


class ProcessingObjectWrapper:

    STOP_TIMEOUT = 1.0

    def __init__(self, name: str, processing_object_classes: list[tuple[Type[ProcessingObject], dict]],
                 is_process=False, maxlen=1, enabled=True):
        self.name = name
        self.is_process = is_process
        self.input_queue = None
        self.output_queue = Queue(maxsize=maxlen) if is_process else ThreadQueue(maxsize=maxlen)
        self.process_thread = None
        self.is_active = False
        self.enabled = enabled
        self.processing_object_classes = processing_object_classes

    def run_function(self):
        if not self.enabled or self.process_thread is None:
            return
        processing_objects = [processing_object_class(**args) for processing_object_class, args in self.processing_object_classes]
        while self.is_active:
            try:
                # Read next frame
                while True:
                    try:
                        frame_info: FrameInfo = self.input_queue.get(timeout=self.STOP_TIMEOUT)
                        break
                    except Empty:
                        if not self.is_active:
                            return
                # logging.info(f"{self.name}: Processign frame info")
                for po in processing_objects:
                    if frame_info.skip:
                        break
                    po.process_frame_info(frame_info)
                # Pass frame to output
                self.output_queue.put(frame_info)
            except Exception as e:
                logging.error(f"Processing object {self.name} failed to process frame with error: {e}")
                frame_info.skip = True
                self.output_queue.put(frame_info)

    def create_process_thread(self):
        if not self.enabled:
            return
        self.process_thread = Process(target=self.run_function) if self.is_process else Thread(target=self.run_function)

    def start(self):
        if not self.enabled:
            return
        self.create_process_thread()
        self.is_active = True
        self.process_thread.start()

    def stop(self):
        self.is_active = False
        if self.process_thread is None:
            return
        self.process_thread.join(self.STOP_TIMEOUT)
        if self.process_thread.is_alive() and self.is_process:
            self.process_thread.kill()

    def __del__(self):
        self.stop()
