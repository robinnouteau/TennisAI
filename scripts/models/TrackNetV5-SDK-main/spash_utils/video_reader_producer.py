import threading


class VideoReaderProducer(threading.Thread):
    def __init__(self, queue, start_idx, end_idx, fn):
        super().__init__()
        self.queue = queue
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.fn = fn
        self.done = False

    def run(self):
        for frame_idx in range(self.start_idx, self.end_idx):
            self.queue.put(self.fn(frame_idx))
        self.done = True


class BatchVideoReaderProducer(threading.Thread):
    def __init__(self, queue, start_idx, end_idx, fn, batch_size):
        super().__init__()
        self.queue = queue
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.fn = fn
        self.done = False
        self.batch_size = batch_size

    def run(self):
        for batch_start_idx in range(self.start_idx, self.end_idx, self.batch_size):
            batch_end_idx = min(batch_start_idx + self.batch_size, self.end_idx)
            batch_frames = self.fn(batch_start_idx, batch_end_idx)
            for f in batch_frames:
                self.queue.put(f, block=True)
        self.done = True
