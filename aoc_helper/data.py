import datetime
import pathlib
import re

try:
    import importlib_metadata as metadata  # type: ignore
except ImportError:
    from importlib import metadata  # type: ignore

DATA_DIR = pathlib.Path.home() / ".config" / "aoc_helper"
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True)

DEFAULT_YEAR = datetime.datetime.today().year
TODAY = datetime.datetime.today().day
URL = "https://adventofcode.com/{year}/day/{day}"
WAIT_TIME = re.compile(r"You have (?:(\d+)m )?(\d+)s left to wait.")
RANK = re.compile(r"You (?:got|achieved) rank (\d+) on this star's leaderboard.")

HEADERS = {
    "User-Agent": (
        f"github.com/starwort/aoc_helper v{metadata.version('aoc_helper')}"
        " contact: Reddit u/starwort Discord @Starwort#6129"
    )
}


def get_cookie():
    token_file = DATA_DIR / "token.txt"
    if token_file.exists():
        return {"session": token_file.read_text().strip("\n")}
    token = input("Could not find configuration file. Please enter your token\n>>> ")
    token_file.write_text(token)
    return {"session": token}
