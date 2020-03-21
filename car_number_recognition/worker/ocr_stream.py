import logging
from threading import Thread
import numpy as np

from ocr.predictor import Predictor
from worker.state import State
from worker.video_reader import VideoReader


class OcrStream:
    def __init__(self, name, state: State, video_reader: VideoReader, predictor_info: dict):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = state
        self.video_reader = video_reader
        self.ocr_thread = None
        self.stopped = False
        self.predictor = Predictor(predictor_info['model'], (32, 80), predictor_info['device'])

        self.logger.info("Create OcrStream")

    def _ocr_loop(self):
        try:
            while True:
                if self.stopped:
                    return
                frame = self.video_reader.read()

                pred = self.predictor.predict(frame[np.newaxis, :])
                self.state.text = pred
                self.state.frame = frame

        except Exception as e:
            # self.logger.exception(e)
            self.state.exit_event.set()

    def _start_ocr(self):
        self.ocr_thread = Thread(target=self._ocr_loop)
        self.ocr_thread.start()

    def start(self):
        self._start_ocr()
        self.logger.info("Start OcrStream")

    def stop(self):
        self.stopped = True
        if self.ocr_thread is not None:
            self.ocr_thread.join()
        self.logger.info("Stop OcrStream")
