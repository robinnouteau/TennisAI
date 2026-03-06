from typing import Optional

from tqdm import tqdm
from spash_utils.video_reader import VideoReader, Frame
from loguru import logger
import numpy as np


def _is_frame_ok(frame: Frame) -> bool:
    return not np.all(frame.arr == 0)


def is_frame_ok(vr: VideoReader, frame_id: int, frame_tolerance: int = 2) -> bool:
    for i in range(frame_tolerance):
        frame = read_frame(vr, frame_id + i)
        if _is_frame_ok(frame):
            return True
    return False


def read_frame(vr: VideoReader, frame_id: int) -> Frame:
    import os

    exc = None
    fd = os.dup(1)  # Save stdout
    fd2 = os.dup(2)  # Save stderr
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)  # Redirect stdout to /dev/null
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)  # Redirect stderr to /dev/null
    try:
        frame = vr[frame_id]
    except Exception as e:
        exc = e
    finally:
        os.dup2(fd, 1)  # Restore stdout
        os.dup2(fd2, 2)  # Restore stderr
    if exc is not None:
        raise exc
    return frame


def dichotomic_search(
    vr: VideoReader, frame_id: int, min_val: int, max_val: int, frame_tolerance: int = 5
) -> int:
    if frame_id in [min_val, max_val]:
        return frame_id

    is_ok = is_frame_ok(vr, frame_id, frame_tolerance=frame_tolerance)
    logger.info(f"Frame {frame_id} in [{min_val}:{max_val}] {is_ok}")
    if is_ok:
        # Search right
        next_id = (frame_id + max_val) // 2
        return dichotomic_search(
            vr, next_id, frame_id, max_val, frame_tolerance=frame_tolerance
        )
    else:
        # Search left
        next_id = (min_val + frame_id) // 2

        return dichotomic_search(
            vr, next_id, min_val, frame_id, frame_tolerance=frame_tolerance
        )


def get_max_idx(path: str, video_reader: Optional[VideoReader] = None, frame_tolerance: int = 2, use_linear: bool = False) -> int:
    if video_reader is None:
        video_reader = VideoReader(path, 'opencv')
    n = len(video_reader)
    last_ok = n

    if use_linear:
        logger.info("Looking for max idx with linear search")
        for i in tqdm(range(0, n, frame_tolerance)):
            frame = read_frame(video_reader, i)
            if _is_frame_ok(frame) is False:
                last_ok = i - frame_tolerance
                break
            else:
                last_ok = i

    else:
        logger.info("Looking for max idx with dichotomic search")
        last_frame_id = n - 1
        is_last_frame_ok = is_frame_ok(video_reader, last_frame_id, frame_tolerance=frame_tolerance)

        if is_last_frame_ok:
            last_ok = last_frame_id
        else:
            last_ok = dichotomic_search(
                video_reader, last_frame_id // 2, 0, last_frame_id, frame_tolerance=frame_tolerance)

    ratio = last_ok / n
    print(f"{path} len={n} last_ok={last_ok} ratio={ratio:.2%}")

    return last_ok
