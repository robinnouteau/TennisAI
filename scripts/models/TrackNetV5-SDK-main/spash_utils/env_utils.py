import os
from typing import Union
from loguru import logger

SPORT_PADEL = 'padel'
SPORT_FOOT = 'foot'
SPORT_FOOT_LONG = 'football'


def env_get_sport() -> str:
    return os.getenv('SPORT', SPORT_PADEL).lower()


def env_is_padel() -> bool:
    return env_get_sport() == SPORT_PADEL or env_get_sport() == ''


def env_is_foot() -> bool:
    return env_get_sport() == SPORT_FOOT or env_get_sport() == SPORT_FOOT_LONG

TRUTHY_VALUES = ('true', '1', 't')  # Add more entries if you want, like: `y`, `yes`, `on`, ...
FALSY_VALUES = ('false', '0', 'f')  # Add more entries if you want, like: `n`, `no`, `off`, ...
VALID_VALUES = TRUTHY_VALUES + FALSY_VALUES


def get_bool_env_variable(name: str, default_value: Union[bool, None] = None) -> bool:
    value = os.getenv(name) or default_value
    if value is None:
        raise ValueError(f'Environment variable "{name}" is not set!')
    value = str(value).lower()
    if value not in VALID_VALUES:
        raise ValueError(f'Invalid value "{value}" for environment variable "{name}"!')
    return value in TRUTHY_VALUES


def get_int_env_variable(name: str, default_value: int = 0):
    value = os.getenv(name) or default_value
    try:
        return int(value)
    except:
        logger.error(f'Environment variable error {name}, use dault one {default_value}')
    return default_value


def get_tuple_env_variable(name: str, default_value=()) -> tuple:
    value = os.getenv(name)
    if not value:
        return default_value
    try:
        return tuple(value.split(','))
    except Exception:
        logger.error(f'Environment variable error {name}, use default value {default_value}')
    return default_value


def has_env_variable(name: str) -> bool:
    # Check if the environment variable is set and not empty
    return os.getenv(name, '') != ''
