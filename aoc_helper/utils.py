import re


def extract_ints(raw: str) -> list[int]:
    """Utility function to extract all integers from some string.

    Many inputs can be directly parsed with this function.
    """
    return list(map(int, re.findall(r"(\d+)", raw)))
