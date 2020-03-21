import logging

import cv2


class VideoReader:
    def __init__(self, name, video_path=0):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.video_path = video_path
        self.logger.info("Create VideoReader")

    def start(self):
        self.logger.info("Start VideoReader")
        self.video = cv2.VideoCapture(self.video_path)

    def read(self):
        if self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                return frame
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                raise Exception("No frame found")

        else:
            self.logger.error("Video is not open!")
            raise Exception("No video found")

    def stop(self):
        self.logger.info("Stop VideoReader")
        self.video.release()
        cv2.destroyAllWindows()
