import cv2

cv2.setNumThreads(0)
import sys
import logging
from logging.handlers import RotatingFileHandler

from worker.state import State
from worker.video_reader import VideoReader
from worker.ocr_stream import OcrStream
from worker.visualize_stream import VisualizeStream


# from worker.config import CONFIG
# TODO: INITIALIZE YOUR args parser and remove sys.argv[] calls


def setup_logging(path, level='INFO'):
    handlers = [logging.StreamHandler()]
    file_path = path
    if file_path:
        file_handler = RotatingFileHandler(filename=file_path,
                                           maxBytes=10 * 10 * 1024 * 1024,
                                           backupCount=5)
        handlers.append(file_handler)
    logging.basicConfig(
        format='[{asctime}][{levelname}] - {name}: {message}',
        style='{',
        level=logging.getLevelName(level),
        handlers=handlers,
    )


class CNDProject:
    def __init__(self, name, video_path, save_path, model_path, use_cuda, true_text, fps=10, frame_size=(1600, 800), coord=(500, 500)):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = State()
        self.video_reader = VideoReader("VideoReader", video_path)
        if use_cuda:
            predictor_info = {'device': 'cuda',
                              'model': model_path}
        else:
            predictor_info = {'device': 'cpu',
                              'model': model_path}
        self.ocr_stream = OcrStream(self.name, self.state, self.video_reader, predictor_info)

        self.visualize_stream = VisualizeStream("VisualizeStream", self.video_reader,
                                                self.state, save_path, fps, frame_size, coord, true_text)
        self.logger.info("Start Project")

    def start(self):
        self.logger.info("Start project act start")
        try:
            self.video_reader.start()
            self.ocr_stream.start()
            self.visualize_stream.start()
            self.state.exit_event.wait()
        except Exception as e:
            self.logger.exception(e)
        finally:
            self.stop()

    def stop(self):
        self.logger.info("Stop Project")

        self.video_reader.stop()
        self.ocr_stream.stop()
        self.visualize_stream.stop()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", help="Path to videofile", required=True)
    parser.add_argument("--model", help="Path to model", required=True)
    parser.add_argument("--level", help="Level for logging", default='INFO')
    parser.add_argument("--save", help="Save path for video", required=True)
    parser.add_argument("--log", help="Logging file", required=True)
    parser.add_argument("--use-cuda", help="Use cuda for prediction", action='store_true')
    parser.add_argument("--text", help="True text ", required=True)
    args = parser.parse_args()
    setup_logging(args.log, args.level)
    logger = logging.getLogger(__name__)
    project = None
    try:
        project = CNDProject("CNDProject", args.video_path, args.save, args.model, args.use_cuda, args.text)
        project.start()
    except Exception as e:
        logger.exception(e)
    finally:
        if project is not None:
            project.stop()
