import datetime
import pathlib
import re
import typing

try:
    import importlib_metadata as metadata  # type: ignore
except ImportError:
    from importlib import metadata  # type: ignore

DATA_DIR = pathlib.Path.home() / ".config" / "aoc_helper"
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True)
PRACTICE_DATA_DIR = DATA_DIR / "practice"
if not PRACTICE_DATA_DIR.exists():
    PRACTICE_DATA_DIR.mkdir(parents=True)

DEFAULT_YEAR = datetime.datetime.today().year
TODAY = datetime.datetime.today().day
LEADERBOARD_URL = "https://adventofcode.com/{year}/leaderboard/day/{day}"
URL = "https://adventofcode.com/{year}/day/{day}"
WAIT_TIME = re.compile(r"You have (?:(\d+)m )?(\d+)s left to wait.")
RANK = re.compile(r"You (?:got|achieved) rank (\d+) on this star's leaderboard.")

HEADERS = {
    "User-Agent": (
        f"github.com/starwort/aoc_helper v{metadata.version('aoc_helper')} contact:"
        " Discord @starwort Github https://github.com/Starwort/aoc_helper/issues"
    )
}


@typing.overload
def get_cookie(missing_ok: typing.Literal[False]) -> dict[str, str]: ...
@typing.overload
def get_cookie() -> dict[str, str]: ...
@typing.overload
def get_cookie(
    missing_ok: typing.Literal[True],
) -> typing.Optional[typing.Dict[str, str]]: ...
@typing.overload
def get_cookie(
    missing_ok: bool,
) -> typing.Optional[typing.Dict[str, str]]: ...


def get_cookie(missing_ok=False):
    token_file = DATA_DIR / "token.txt"
    if token_file.exists():
        return {"session": token_file.read_text().strip("\n")}
    if missing_ok:
        return None
    token = input("Could not find configuration file. Please enter your token\n>>> ")
    token_file.write_text(token)
    return {"session": token}
