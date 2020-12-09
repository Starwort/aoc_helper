import re
import typing


def extract_ints(raw: str) -> typing.List[int]:
    """Utility function to extract all integers from some string.

    Many inputs can be directly parsed with this function.
    """
    return list(map(int, re.findall(r"((?:-|\+)?\d+)", raw)))
