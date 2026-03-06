import m3u8
from loguru import logger


class SegmentInfo():
    def __init__(self, uri: str, duration: float):
        self.uri = uri
        self.duration = duration

    def __repr__(self):
        return f'[{self.uri} : {self.duration}]'


def get_playlist(filename: str):
    try:
        playlist = m3u8.load(filename)
        return playlist.files
    except IOError:
        logger.error(f'm3u8 file {filename} do not exist.')
    return []


def get_segments(filename: str) -> list[SegmentInfo]:
    try:
        playlist = m3u8.load(filename)
        return [SegmentInfo(seg.uri, seg.duration) for seg in playlist.segments]
    except IOError:
        logger.error(f'm3u8 file {filename} do not exist.')
    return []