import logging
import cv2


class VideoWriter:
    def __init__(self, name, video_path, fps, frame_size):
        self.name = name
        self.frame_size = frame_size
        self.logger = logging.getLogger(self.name)
        self.video_path = video_path
        self.video = cv2.VideoWriter(video_path,
                                     cv2.VideoWriter_fourcc('M','J','P','G'),
                                     fps, self.frame_size)
        self.logger.info("Create VideoWriter")

    def write(self, frame):
        self.video.write(frame)

    def start(self, ):
        self.video.open(self.video_path)

    def stop(self):
        self.video.release()
        cv2.destroyAllWindows()
        self.logger.info(f"Stop VideoWriter")
