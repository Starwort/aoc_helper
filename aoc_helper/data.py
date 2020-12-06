import datetime
import os
import pathlib
import re

DATA_DIR = pathlib.Path(os.path.expanduser(os.path.join("~", ".config", "aoc_helper")))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

DEFAULT_YEAR = datetime.datetime.today().year
URL = "https://adventofcode.com/{year}/day/{day}"
WAIT_TIME = re.compile(r"You have (?:(\d+)m )?(\d+)s left to wait.")


def get_cookie():
    try:
        with open(DATA_DIR / "token.txt") as file:
            return {"session": file.read()}
    except FileNotFoundError:
        token = input(
            "Could not find configuration file. Please enter your token\n>>> "
        )
        with open(DATA_DIR / "token.txt", "w") as file:
            file.write(token)
        return {"session": token}
