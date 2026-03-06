import json
import subprocess
from typing import OrderedDict
import tempfile
import shutil
import os
import torch
import numpy as np
from loguru import logger
import torchvision
import math
import cv2

from spash_utils.m3u8_utils import SegmentInfo, get_segments

VIDEO_LIBRARY = {'opencv': True, 'decord': False, 'torchcodec': False, 'av': False}

try:
    import av
    VIDEO_LIBRARY['av'] = True
except:
    pass
DEFAULT_BACKEND = 'opencv'
try:
    from decord import VideoReader as VR, cpu
    import decord
    VIDEO_LIBRARY['decord'] = True
    DEFAULT_BACKEND = 'decord'
except:
    pass

try:
    from torchcodec.decoders import VideoDecoder
    VIDEO_LIBRARY['torchcodec'] = True
    # For MAcOS : export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
except (ImportError, RuntimeError):
    pass

print('Video Library', VIDEO_LIBRARY)


class VideoReader():
    def __init__(self, filename: str, backend=DEFAULT_BACKEND) -> None:
        self.backend = backend
        if backend == 'm3u8' or filename.endswith('.m3u8'):
            self._reader = M3u8VideoReader()
        elif backend == 'decord' and VIDEO_LIBRARY['decord']:
            self._reader = Decord_VideoReader()
        elif backend == 'opencv':
            self._reader = OpenCV_VideoReader() 
        elif backend == 'tv' or backend == 'torchvision':
            self._reader = TorchVision_VideoReader()
        elif backend == 'av' and VIDEO_LIBRARY['av']:
            self._reader = AV_VideoReader()
        elif backend == 'torchcodec' and VIDEO_LIBRARY['torchcodec']:
            self._reader = TorchCodec_Reader()
        elif backend == 'mock':
            self._reader = Mock_VideoReader()
        else:
            self._reader = OpenCV_VideoReader()
        self._reader.open(filename)

    def get_batch(self, indices):
        return self._reader.get_batch(indices)

    def __len__(self):
        return len(self._reader)

    def __getitem__(self, idx):
        return self._reader[idx]

    def get_avg_fps(self):
        return self._reader.get_avg_fps()

    def get_width(self):
        return self._reader.get_width()   # float `width`

    def get_height(self):
        return self._reader.get_height() # float `height`

    def seek(self, i):
        self._reader.seek(i)

    def set_bridge(self, bridge):
        self._reader.set_bridge(bridge)

    def __iter__(self):
        return self._reader.__iter__()

    def __next__(self):
        return self._reader.__next__()

    # def __del__(self):
    #    self._reader.__del__()
    #    del self._reader



class Frame():
    def __init__(self, vr: VideoReader, arr: np.ndarray, to_rgb=True):
        if to_rgb:
            try:
                # print ('Arr', arr.shape, arr.dtype, vr.get_width(), vr.get_height())
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            except:
                print('Error reading frame .... Replace by empty one')
                arr = np.zeros((int(vr.get_height()), int(vr.get_width()), 3), dtype=np.uint8)
        self.arr = arr

    def asnumpy(self):
        return self.arr

    @property
    def shape(self):
        return self.arr.shape



class ImageVideoReader():
    def __init__(self) -> None:
        pass

    def open(self, filename):
        self.img = cv2.imread(filename)

    def get_avg_fps(self):
        return 0

    def get_width(self):
        return self.img.shape[1]  

    def get_height(self):
        return self.img.shape[0]

    def __getitem__(self, idx):
        return Frame(self, self.img)


class VideoReaderList():

    def __init__(self, vr: list[VideoReader]) -> None:
        self.vr = vr

    def __getitem__(self, idx):
        return self.vr[idx]


class TorchVision_VideoReader():
    def __init__(self) -> None:
        torchvision.set_video_backend('video_reader')
        pass

    def open(self, filename):
        self.reader = torchvision.io.VideoReader(filename, "video")
        self.meta = self.reader.get_metadata()
        self.fps = 25

    def seek(self, i):
        self.reader.seek(i / self.fps)

    def __del__(self):
        self.reader = None


class Decord_VideoReader():
    def __init__(self):
        self.vr = None
        pass

    def open(self, filename):
        self.vr = VR(filename, ctx=cpu(0))
        h, w, _ = self.vr[0].shape
        self.w = w
        self.h = h

    def get_batch(self, indices):
        return self.vr.get_batch(indices)

    def __len__(self):
        return len(self.vr)

    def __getitem__(self, idx):
        return self.vr[idx]

    def __next__(self):
        return self.vr.next()

    def get_avg_fps(self):
        return self.vr.get_avg_fps()

    def get_width(self):
        return self.w   # float `width`

    def get_height(self):
        return self.h  # float `height`

    def seek(self, i):
        self.vr.seek(i)

    def set_bridge(self, bridge):
        decord.bridge.set_bridge(bridge)

    def __iter__(self):
        return self

    def __del__(self):
        if self.vr is not None:
            del self.vr
            self.vr = None


def get_nb_frames_from_ffprobe(videoname) -> int:
    try:
        cmd = f'ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -print_format json {videoname}'
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        d = json.load(p.stdout)
        return int(d['streams'][0]['nb_read_frames'])
    except Exception as ex:
        logger.error('Error when runing ffprobe', ex)
        if p is not None:
            logger.error('ffprobe stderr {}', p.stderr.read().decode('UTF-8'))
    return 0


class VideoMetadata:
    fps = 0
    width = 0
    height = 0
    nb_frames = 0

    def __repr__(self):
        return f'[fps:{self.fps}, width:{self.width}, height:{self.height}, nb_frames: {self.nb_frames}]'


def get_val(division_str):
    try:
        numerator, denominator = division_str.split('/')
        return float(numerator) / float(denominator)
    except:
        return 0


def get_metadata_from_ffprobe(videoname) -> VideoMetadata:
    metadata = VideoMetadata()
    try:
        cmd = f"ffprobe -v error -select_streams v:0 -count_frames -show_entries stream -print_format json '{videoname}'"
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        d = json.load(p.stdout)
        metadata.nb_frames = int(d['streams'][0]['nb_read_frames'])
        metadata.width = int(d['streams'][0]['width'])
        metadata.height = int(d['streams'][0]['height'])
        metadata.fps = get_val(d['streams'][0]['avg_frame_rate'])
        if metadata.fps == 0:
            metadata.fps = get_val(d['streams'][0]['r_frame_rate'])
        # check that nb_read_frames / duration almost equal to fps, otherwise put zero to ignore fps
        frame_rate_from_duration = metadata.nb_frames / float(d['streams'][0]['duration'])
        if abs(metadata.fps - frame_rate_from_duration) >= 1:
            metadata.fps = 0  # disable fps if too big difference
    except Exception as ex:
        logger.error('Error when runing ffprobe', ex)
        if p is not None:
            logger.error('ffprobe stderr {}', p.stderr.read().decode('UTF-8'))
    logger.info(f'Metatada for {videoname} : {metadata}')
    return metadata


class OpenCV_VideoReader():
    def __init__(self):
        self.bridge = ''

    def open(self, filename):
        self.filename = filename
        self._vr = cv2.VideoCapture()
        self._vr.open(self.filename)
        ok, frame = self.read()  # read frame to get number of channels
        if ok:
            self.frame_channels = int(frame.shape[-1])
        else:
            raise IOError(f'cannot read frame from {self.filename}.')
        self.number_of_frames = int(self._vr.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.get_avg_fps() == 100:
            self.number_of_frames = get_nb_frames_from_ffprobe(self.filename)
        self.seek(0)  # reset to first frame

    def seek(self, frame_number):
        """Go to frame."""
        self._vr.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def get_batch(self, index, device=None):
        _l = list(index)
        frames = []
        self.seek(_l[0])
        for i in range(len(_l)):
            ret, frame = self._vr.read()  # read
            frames.append(Frame(self, frame).asnumpy())
        if self.bridge == 'torch':
            return torch.tensor(np.array(frames), device=device)
        return frames

    def __iter__(self):
        return self

    def get_avg_fps(self):
        return self._vr.get(cv2.CAP_PROP_FPS)

    def get_width(self):
        return self._vr.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`

    def get_height(self):
        return self._vr.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    def __len__(self):
        """Length is number of frames."""
        return self.number_of_frames  # TODO: Check if it's correct ????

    def __getitem__(self, index):
        """Now we can get frame via self[index] and self[start:stop:step]."""
        if isinstance(index, slice):
            return (self[ii] for ii in range(*index.indices(len(self))))
        elif isinstance(index, (list, tuple, range)):
            return (self[ii] for ii in index)
        else:
            return Frame(self, self.read(index)[1])

    def read(self, frame_number=None):
        """Read next frame or frame specified by `frame_number`."""
        is_current_frame = frame_number == self.current_frame_pos
        # no need to seek if we are at the right position - greatly speeds up reading sunbsequent frames
        if frame_number is not None and not is_current_frame:
            self.seek(frame_number)
        ret, frame = self._vr.read()  # read
        return ret, frame

    def __next__(self):
        ret, frame = self._vr.read() 
        if ret != -1:
            return Frame(self, frame)

    def __exit__(self):
        """Release video file."""
        del (self)

    def __del__(self):
        try:
            self._vr.release()
        except AttributeError:  # if file does not exist this will be raised since _vr does not exist
            pass

    def set_bridge(self, bridge):
        self.bridge = bridge  # TODO

    @property
    def current_frame_pos(self):
        return self._vr.get(cv2.CAP_PROP_POS_FRAMES)


class TorchCodec_Reader():
    def __init__(self):
        pass

    def open(self, filename):
        self.decoder = VideoDecoder(filename, device='cpu')

    def get_avg_fps(self):
        return self.decoder.metadata.average_fps

    def __len__(self):
        return self.decoder.metadata.num_frames

    def get_width(self):
        return self.decoder.metadata.width

    def get_height(self):
        return self.decoder.metadata.height  # float `height`

    def __getitem__(self, idx):
        frame = self.decoder[idx]
        return Frame(self, frame.cpu().numpy())

    def set_bridge(self, bridge):
        self.bridge = bridge  # TODO

    def get_batch(self, index, device=None):
        _l = list(index)
        frames = []
        _frames = self.decoder.get_frames_at(_l)
        for i in range(len(_l)):
            frames.append(_frames.data[i].cpu().numpy())
        return _frames


class AV_VideoReader():
    def __init__(self):
        pass

    def open(self, filename):
        self.filename = filename
        self.container = av.open(filename, "r")
        # self.container.fast_seek = True

    def get_batch(self, indices):
        pass

    def __len__(self):  # not working ???
        return int(self.container.streams.video[0].frames)

    def get_avg_fps(self):
        return self.container.streams.video[0].average_rate

    def get_width(self):
        return self.container.streams.video[0].width   # float `width`

    def get_height(self):
        return self.container.streams.video[0].height  # float `height`

    def __getitem__(self, idx):
        frame_num = idx  # the frame I want
        framerate = self.container.streams.video[0].average_rate  # get the frame rate
        # time_base = self.container.streams.video[0].time_base # get the time base
        sec = int(frame_num/framerate)  # timestamp for that frame_num
        # self.container.seek(sec*1000000, whence='time', backward=True)  # seek to that nearest timestamp
        self.container.seek(sec*1000000, any_frame=True, backward=True)
        frame = next(self.container.decode(video=0))  # get the next available frame
        # sec_frame = int(frame.pts * time_base * framerate) # get the proper key frame number of that timestamp

        # for _ in range(sec_frame, frame_num):
        #    frame = next(container.decode(video=0))
        return Frame(self, frame.to_ndarray(format='bgr24'))


class VideoReader_VideoReader():

    def __init__(self):
        pass

    def open(self, filename):
        self.filename = filename

    def get_batch(self, indices):
        import spash_utils.video_reader as video_reader

        return video_reader.get_batch(self.filename, indices)

    def __len__(self):
        import spash_utils.video_reader as video_reader

        (n, _, _) = video_reader.get_shape(self.filename)
        return n

    def __getitem__(self, idx):
        # import video_reader
        pass


class Mock_VideoReader():
    def __init__(self, len):
        self.len = len
        self.fps = 25

    def __len__(self):
        return self.len

    def get_avg_fps(self):
        return self.fps


class M3u8VideoReader():
    def __init__(self):
        self.bridge = ''
        self.segments: list[SegmentInfo] = []
        self._cur_uri = None
        self._cur_vr = None
        self._cur_index = None

    def open(self, filename):
        self.base = os.path.dirname(filename)
        self.segments = get_segments(filename)

        self._load()

    def _get_meta(self):
        _idx = min(10, len(self.segments))
        for i in range(_idx):
            meta = self.get_metadata(self.segments[i].uri)
            if meta.fps != 0:
                return meta
        return None

    def _load(self):
        # Compute sats using first items
        self.duration = [0]
        if len(self.segments) >= 1:
            meta = self._get_meta()
            if meta is not None:
                self.fps = meta.fps  # _vr.get(cv2.CAP_PROP_FPS)
                self.width = meta.width  # _vr.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
                self.height = meta.height  # _vr.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
            else:
                _vr = self.get_VideoCapture(self.segments[0].uri)
                self.fps = _vr.get(cv2.CAP_PROP_FPS)
                self.width = _vr.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
                self.height = _vr.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

            if self.fps > 1000:
                raise Exception('Too big FPS')

            d = 0
            for seg in self.segments:
                d += round(seg.duration * self.fps)
                self.duration.append(d)
            self.len = d
        else:
            self.fps = 0
            self.width = 0
            self.height = 0
            self.len = 0
        self.duration = np.array(self.duration)

    def __len__(self):
        return self.len

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_avg_fps(self):
        return self.fps

    def get_uri_new_index(self, index):
        seg_idx = self.duration.searchsorted(index, side='right') - 1
        new_index = index - self.duration[seg_idx]
        return self.segments[seg_idx].uri, new_index

    def __getitem__(self, index):
        next_uri, new_index = self.get_uri_new_index(index)
        if self._cur_uri is None or self._cur_uri != next_uri:
            self._cur_vr = self.get_VideoCapture(next_uri)
            self._cur_uri = next_uri
            self._cur_index = None  # Reset index to force seek on new video
        if new_index != self._cur_index:
            self._cur_vr.set(cv2.CAP_PROP_POS_FRAMES, new_index)
        ret, frame = self._cur_vr.read()  # read
        self._cur_index = new_index + 1
        return Frame(self, frame)

    def get_batch(self, index, device=None):
        frames = []

        for idx in index:
            next_uri, new_index = self.get_uri_new_index(idx)

            if self._cur_uri is not None and next_uri == self._cur_uri:
                vr = self._cur_vr
                if self._cur_index is not None and new_index != self._cur_index:
                    vr.set(cv2.CAP_PROP_POS_FRAMES, new_index)
            else:
                vr = self.get_VideoCapture(next_uri)
                if new_index != 0:
                    vr.set(cv2.CAP_PROP_POS_FRAMES, new_index)

            # _frame_idx = vr.get(cv2.CAP_PROP_POS_FRAMES)
            # assert _frame_idx == new_index, f"Frame index mismatch: {_frame_idx} != {new_index}"
            ret, frame = vr.read()  # read
            self._cur_vr = vr
            self._cur_uri = next_uri
            self._cur_index = new_index + 1
            frames.append(Frame(self, frame).asnumpy())

        if self.bridge == 'torch':
            return torch.tensor(np.array(frames), device=device)
        return frames

    def get_metadata(self, video):
        return get_metadata_from_ffprobe(os.path.join(self.base, video))

    def get_VideoCapture(self, video):
        _vr = cv2.VideoCapture()
        _vr.open(os.path.join(self.base, video))
        return _vr

    def set_bridge(self, bridge):
        self.bridge = bridge


class TemporaryFileCache:
    def __init__(self, max_files=10):
        self.cache = OrderedDict()
        self.max_files = max_files
        self.temp_dir = tempfile.mkdtemp()

    def _clean_cache(self):
        while len(self.cache) > self.max_files:
            self.cache.popitem(last=False)

    def create_file(self, key):
        if key in self.cache:
            raise ValueError(f"File with key '{key}' already exists.")

        file_path = os.path.join(self.temp_dir, key)
        open(file_path, 'a').close()  # Crée un fichier vide

        self.cache[key] = file_path
        self._clean_cache()
        return file_path

    def get_file_path(self, key):
        if key not in self.cache:
            raise ValueError(f"File with key '{key}' does not exist.")

        return self.cache[key]

    def has_file_path(self, key):
        return key in self.cache

    def clear_cache(self):
        for file_path in self.cache.values():
            os.remove(file_path)
        self.cache.clear()

    def __del__(self):
        shutil.rmtree(self.temp_dir)


class S3M3u8VideoReader(M3u8VideoReader):
    def __init__(self, s3, bck: str):
        super().__init__()
        self.s3 = s3
        self.bck = bck  # Bucket
        self.cache = TemporaryFileCache(10)

    def load(self, segments: list[SegmentInfo]):  # Load from m3u8 segments
        self.segments = segments
        self._load()

    def _get_file_path(self, video):
        if not self.cache.has_file_path(self.bck + video):
            file_path = self.cache.create_file(self.bck + video)
            self.s3.download_file(self.bck, video, file_path)
        else:
            file_path = self.cache.get_file_path(self.bck + video)
        return file_path

    def get_VideoCapture(self, video):
        file_path = self._get_file_path(video)

        _vr = cv2.VideoCapture()
        _vr.open(file_path)
        return _vr

    def get_metadata(self, video):
        file_path = self._get_file_path(video)
        return get_metadata_from_ffprobe(file_path)
