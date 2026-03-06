from enum import Enum
from typing import Union


class StrEnum(str, Enum):
    """
    Enum that behaves like a str.
    ie:
      - str(ExampleStrEnum.A) return its value not "ExampleStrEnum.A"
      - can be compared with str
      - can be compared with other StrEnum class

    NOTE: Use `is` to strictly compare that the object is the same instance.
    i.e:
        `ExampleStrEnum.A == ExampleStrEnum.A` might return True (if same .value)
        but `ExampleStrEnum.A is ExampleStrEnum.A` will return False
    """

    def __str__(self):
        return self.value
    
    def __eq__(self, other: Union[str, "StrEnum"]):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))