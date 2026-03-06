from typing import Any


def safe_dict_add(dict: dict, key: Any, value: Any, default_value: Any=0) -> None:
    if key not in dict:
        dict[key] = default_value
    dict[key] += value


def safe_nested_dict_set(dict: dict, keys: list[Any], value: Any) -> None:
    for key in keys[:-1]:
        if key not in dict:
            dict[key] = {}
        dict = dict[key]
    dict[keys[-1]] = value


def safe_dict_append(dict: dict, key: Any, value: Any) -> None:
    if key not in dict:
        dict[key] = []
    dict[key].append(value)


def dict_get_set(dict: dict, key: Any, default_value: Any) -> Any:
    o = dict.get(key, default_value)
    dict[key] = o
    return o


def get_int(dict: dict, key: Any, default_value: int = 0) -> int:
    try:
        return int(dict.get(key, default_value))
    except:  # noqa: E722
        return default_value


def get_bool(dict: dict, key: Any, default_value: bool = False) -> bool:
    try:
        return str(dict.get(key, str(default_value))).lower() == 'true'
    except:  # noqa: E722
        return default_value


def sum_dicts(d1, d2):
    merged = {}
    keys = set(d1.keys()) | set(d2.keys())
    for key in keys:
        merged[key] = {}

        v1 = d1.get(key, None)
        v2 = d2.get(key, None)

        if v1 is not None and v2 is not None:
            if isinstance(v1, dict) and isinstance(v2, dict):
                merged[key] = sum_dicts(v1, v2)
            else:
                if type(v1) != type(v2):
                    raise TypeError(f"Type mismatch: {type(v1)} != {type(v2)}")
                merged[key] = v1 + v2
        else:
            merged[key] = v1 if v1 is not None else v2

    return merged
