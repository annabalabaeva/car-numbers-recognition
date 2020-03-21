import threading
from datetime import datetime
from copy import copy


class State:
    def __init__(self):
        self.exit_event = threading.Event()
        self.text = ""
        self.frame = None

    @property
    def data(self):
        data = {'text': copy(self.text), 'ts': datetime.utcnow().timestamp(), 'frame': copy(self.frame)}

        return data
