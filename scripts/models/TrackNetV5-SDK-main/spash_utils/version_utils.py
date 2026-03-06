import re
from typing import Union
from packaging.version import parse as parse_version
from loguru import logger


def parse_version(version_str: str) -> list[int]:
    """
    Parse version string into list of integers.

    Args:
        version_str: Version string like '0.0.8.3' or 'v0.0.8.3'

    Returns:
        List of integers representing version components
    """
    try:
        # Handle case where version might be stored as integer
        if isinstance(version_str, (int, float)):
            return [int(version_str)]

        version_str = str(version_str).strip()

        # Remove 'v' prefix if present (case insensitive)
        if version_str.lower().startswith("v"):
            version_str = version_str[1:]

        # Split by dots and convert to integers
        parts = version_str.split(".")
        version = []
        for part in parts:
            if part.isdigit():
                version.append(int(part))
            else:
                raise ValueError(f"Invalid version part: {part} in {version_str}")
        return version
    except (ValueError, AttributeError) as e:
        logger.error(f"Invalid version string: {version_str}: {type(e)}: {e}")
        raise


def compare_versions(version1: Union[str, int], version2: Union[str, int]) -> int:
    """
    Compare two version strings

    Returns:
        -1 if version1 < version2
         0 if version1 == version2
         1 if version1 > version2
    """
    v1_parts = parse_version(version1)
    v2_parts = parse_version(version2)

    # Pad shorter version with zeros
    max_len = max(len(v1_parts), len(v2_parts))
    v1_parts.extend([0] * (max_len - len(v1_parts)))
    v2_parts.extend([0] * (max_len - len(v2_parts)))

    # Compare component by component
    for v1_part, v2_part in zip(v1_parts, v2_parts):
        if v1_part < v2_part:
            return -1
        elif v1_part > v2_part:
            return 1

    return 0


def evaluate_version_filter(version: Union[str, int], filter_expr: str) -> bool:
    """
    Evaluate version against filter expression.

    Args:
        version: Version to check
        filter_expr: Filter expression like '0.1', '>0.1', '<=0.0.7.3', 'v0.1', '>v0.1'

    Returns:
        True if version matches filter, False otherwise
    """
    # Parse operator and version from filter expression
    match = re.match(r"^(>=|<=|>|<|=)?(.+)$", filter_expr.strip())
    if not match:
        raise ValueError(f"Invalid version filter expression: {filter_expr}")

    operator = match.group(1) or "="
    filter_version = match.group(2)

    comparison = compare_versions(version, filter_version)

    if operator == "=":
        return comparison == 0
    elif operator == ">":
        return comparison > 0
    elif operator == "<":
        return comparison < 0
    elif operator == ">=":
        return comparison >= 0
    elif operator == "<=":
        return comparison <= 0
    else:
        raise ValueError(f"Unsupported operator: {operator}")
