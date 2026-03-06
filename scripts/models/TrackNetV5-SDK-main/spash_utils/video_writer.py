import cv2
import os
import platform
import tempfile
from loguru import logger

HAS_FFMPEG = False
try:
    import ffmpeg
    HAS_FFMPEG = True
except:
    pass

HAS_PYAV = False
try:
    import av
    HAS_PYAV = True
except:
    pass


def is_fourcc_available(codec):
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        temp_video = cv2.VideoWriter('temp.mkv', fourcc, 30, (640, 480), isColor=True)
        return temp_video.isOpened()
    except:
        return False


DEFAULT_CODEC = 'mp4v'
if platform.system() == 'Darwin':
    DEFAULT_CODEC = 'H264'
elif is_fourcc_available('H264'):
    DEFAULT_CODEC = 'H264'
print(f"Default codec set to {DEFAULT_CODEC}")


class VideoWriter():
    def __init__(self, backend='opencv') -> None:
        self.backend = backend
        if backend == 'pyav':
            self._writer = PyAV_VideoWriter()
        elif backend == 'ffmpeg' and HAS_FFMPEG:
            self._writer = FFMpeg_VideoWriter()
        else:
            self._writer = OpenCV_VideoWriter()

    def open(self, filename, fps, w, h, codec=DEFAULT_CODEC):
        self._writer.open(filename, fps, w, h, codec)

    def add_frame(self, frame):
        self._writer.add_frame(frame)

    def close(self):
        self._writer.close()


class OpenCV_VideoWriter():
    def __init__(self):
        pass

    def open(self, filename, fps, w, h, codec=DEFAULT_CODEC):
        self.out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*codec), fps, (int(w), int(h)))

    def add_frame(self, frame):
        self.out.write(frame)

    def close(self):
        self.out.release()


class PyAV_VideoWriter():
    def __init__(self):
        pass

    def open(self, filename, fps, w, h, codec=DEFAULT_CODEC):
        self.vid = av.open(filename, "w")
        self.vs = self.vid.add_stream(codec, fps)
        self.vs.width = w
        self.vs.height = h

    def add_frame(self, frame):
        frame = frame.astype(frame.dtype, order='C', copy=False)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        new_frame.pts = None
        # self.vid.mux(self.vs.encode(new_frame))
        for packet in self.vs.encode(new_frame):
            self.vid.mux(packet)

        del new_frame
        del frame

    def close(self):
        self.vid.close()


class FFMpeg_VideoWriter():
    def __init__(self):
        pass

    def open(self, filename, fps, w, h, codec=DEFAULT_CODEC):
        self.dir = tempfile.TemporaryDirectory()
        self.filename = filename
        self.fps = fps
        self.codec = codec  # Not yet used
        self.i = 0

    def add_frame(self, frame):
        if frame.shape[0] != 0 and frame.shape[1] != 0:
            img_filename = os.path.join(self.dir.name, f"img_{self.i:05d}.jpg")
            cv2.imwrite(img_filename, frame)
            self.i += 1

    def close(self):
        err = ''
        out = ''
        try:
            out, err = (ffmpeg.input(os.path.join(self.dir.name, "*.jpg"), pattern_type='glob', framerate=self.fps)
                        .output(os.path.join(self.filename), **{'c:v': 'libx264'})
                        .run(overwrite_output=True, capture_stdout=True, capture_stderr=True))
            logger.debug(out)
            logger.debug(err)
        except:
            logger.error(f'Error generating video {self.filename} {err} {out}')
        finally:
            self.dir.cleanup()
