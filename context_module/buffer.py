from collections import deque
import numpy as np

class SequenceBuffer:
    def __init__(self, max_len=5, num_classes=7):
        self.max_len = max_len
        self.num_classes = num_classes
        self.buffer = deque(maxlen=max_len)

    def add(self, class_id):
        one_hot = np.zeros(self.num_classes)
        one_hot[class_id] = 1.0
        self.buffer.append(one_hot)

    def get_sequence(self):
        if len(self.buffer) < self.max_len:
            return None
        return np.stack(self.buffer)[None, :, :]
