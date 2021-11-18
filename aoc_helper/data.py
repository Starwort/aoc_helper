import datetime
import pathlib
import re

DATA_DIR = pathlib.Path.home() / ".config" / "aoc_helper"
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True)

DEFAULT_YEAR = datetime.datetime.today().year
URL = "https://adventofcode.com/{year}/day/{day}"
WAIT_TIME = re.compile(r"You have (?:(\d+)m )?(\d+)s left to wait.")


def get_cookie():
    token_file = DATA_DIR / "token.txt"
    if token_file.exists():
        return {"session": token_file.read_text().strip('\n')}
    token = input("Could not find configuration file. Please enter your token\n>>> ")
    token_file.write_text(token)
    return {"session": token}
