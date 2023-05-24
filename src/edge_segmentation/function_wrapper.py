from multiprocessing import Process, Queue
from queue import Empty
from queue import Queue as ThreadQueue
from threading import Thread
import logging
import time
from .processing_functions import SavedState


class FunctionWrapper:

    STOP_TIMEOUT = 0.1

    def __init__(self, name: str, funcs: list[callable, dict], enabled=True, is_process=False, maxlen=1):
        self.funcs = funcs
        self.enabled = enabled
        self.name = name
        self.is_process = is_process
        self.input_queue = None
        self.output_queue = Queue(maxsize=maxlen) if is_process else ThreadQueue(maxsize=maxlen)
        self.process_thread = None
        self.is_active = False
        self.saved_state = SavedState()
        self.times = []

    def run_function(self):
        if not self.enabled or self.process_thread is None:
            return
        try:
            while self.is_active:
                # Read next frame
                while True:
                    try:
                        frame_info = self.input_queue.get(timeout=self.STOP_TIMEOUT)
                        break
                    except Empty:
                        if not self.is_active:
                            return
                tic = time.thread_time()
                for func, args in self.funcs:
                    if frame_info.skip:
                        continue
                    func(frame_info, self.saved_state, **args)
                toc = time.thread_time()
                self.times.append(toc - tic)
                # Pass frame to output
                self.output_queue.put(frame_info)
        except Exception as e:
            logging.error(f"FunctionWrapper {self.name} failed with error: {e}")
            raise e

    def create_process_thread(self):
        if not self.enabled:
            return
        self.process_thread = Process(target=self.run_function) if self.is_process else Thread(target=self.run_function)

    def start(self):
        if not self.enabled:
            return
        if self.process_thread is None:
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
