import datetime
import time

from typing import Callable
from loguru import logger


def monday_of_the_week(week):
    jan_first = datetime.datetime(datetime.date.today().year, 1, 1)

    days_until_monday = (8 - jan_first.isoweekday()) % 7
    first_monday = jan_first + datetime.timedelta(days=days_until_monday)

    monday = first_monday + datetime.timedelta(weeks=int(week) - 1)

    return monday.date()


def format_date(date: datetime.date) -> str:
    return date.strftime("%Y-%m-%d")

def format_date_hour(date: datetime.datetime) -> str:
    return date.strftime("%Y-%m-%d %H:%M:%S")


def seconds_to_hms(n_seconds: float) -> str:
    """
    Convert a number of seconds to a string in the format 'HHhMMmSS.SSs'.

    Args:
        n_seconds (float): The number of seconds to convert.

    Returns:
        str: A string representation of the time in the format 'HHhMMmSS.SSs'.
    """
    hours, remainder = divmod(n_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    hours = int(hours)
    minutes = int(minutes)

    if hours > 0:
        return f"{hours:02d}h{minutes:02d}m{seconds:05.2f}s"
    elif minutes > 0:
        return f"{minutes:02d}m{seconds:05.2f}s"
    else:
        return f"{seconds:.2f}s"


def timed_execution(func: Callable, *args, **kwargs):
    start_time = time.time()

    o = func(*args, **kwargs)
    logger.info(f"Function {func.__name__} took {time.time() - start_time:.2f}s")

    duration = time.time() - start_time
    return duration, o


def now_utc():
    """Get current datetime in UTC timezone"""
    return datetime.datetime.now(datetime.timezone.utc)


def timestamp_delta(d1: datetime.datetime, d2: datetime.datetime) -> float:
    return abs(d1.timestamp() - d2.timestamp())


def eq_datetime(d1: datetime.datetime, d2: datetime.datetime, tol_seconds: float = 1e-3) -> bool:
    return timestamp_delta(d1, d2) <= tol_seconds


def ge_datetime(d1: datetime.datetime, d2: datetime.datetime, tol_seconds: float = 1e-3) -> bool:
    return d1.timestamp() - d2.timestamp() >= -tol_seconds


def gt_datetime(d1: datetime.datetime, d2: datetime.datetime) -> bool:
    return d1.timestamp() - d2.timestamp() > 0
