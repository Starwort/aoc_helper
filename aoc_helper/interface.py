import datetime
import time
import typing
import webbrowser
from os import makedirs
from os.path import exists

import requests
from bs4 import BeautifulSoup as Soup

try:
    import colorama as colour

    colour.init()
    RED = colour.Fore.LIGHTRED_EX
    YELLOW = colour.Fore.LIGHTYELLOW_EX
    GREEN = colour.Fore.LIGHTGREEN_EX
    BLUE = colour.Fore.LIGHTBLUE_EX
    RESET = colour.Fore.RESET
except ImportError:
    RED = ""
    YELLOW = ""
    GREEN = ""
    BLUE = ""
    RESET = ""

from .data import DATA_DIR, DEFAULT_YEAR, URL, WAIT_TIME, get_cookie


def make_if_not_exists(folder: str):
    if not exists(folder):
        makedirs(folder)


def analyse_and_print(message: str) -> None:
    if message.startswith("That's the"):
        print(GREEN + message + RESET)
    elif message.startswith("You don't"):
        print(YELLOW + message + RESET)
    elif message.startswith("That's not"):
        print(RED + message + RESET)
    else:
        raise ValueError("Failed to parse response.")


def fetch_data(day: int, year: int = DEFAULT_YEAR) -> str:
    """Fetch and return the input for `day` of `year`.

    All inputs are cached in `aoc_helper.DATA_DIR`."""
    day_ = str(day)
    year_ = str(year)

    make_if_not_exists(DATA_DIR / year_)
    input_path = DATA_DIR / year_ / (day_ + ".in")

    if exists(input_path):
        with open(input_path) as file:
            return file.read()
    else:
        unlock = datetime.datetime(year, 12, day, 5)
        now = datetime.datetime.utcnow()
        if now < unlock:
            print(YELLOW + "Waiting for puzzle unlock..." + RESET)
            time.sleep((unlock - now).total_seconds())
            print(GREEN + "Fetching input!" + RESET)
        resp = requests.get(
            URL.format(day=day, year=year) + "/input", cookies=get_cookie()
        )
        if not resp.ok:
            raise ValueError("Received bad response")
        data = resp.text.strip()
        with open(input_path, "w") as file:
            file.write(data)
        return data


day = fetch_data  # alias for old interface


def submit_answer(
    day: int, part: int, answer: typing.Any, year: int = DEFAULT_YEAR
) -> None:
    """Submit an answer.

    Submissions are cached; submitting the previous answer will return the
    previous response.
    """
    day_ = str(day)
    year_ = str(year)
    part_ = str(part)
    answer_ = str(answer)

    make_if_not_exists(DATA_DIR / year_)
    path_base = str(DATA_DIR / year_ / part_)

    if exists(path_base + ".solution"):  # empty flag file
        with open(path_base + ".out") as file:
            print(
                "Day "
                + BLUE
                + day_
                + RESET
                + " part "
                + BLUE
                + part_
                + RESET
                + " has already been solved.\nThe solution was: "
                + BLUE
                + file.read()
            )
            return

    with open(path_base + ".out") as file:
        if answer_ == file.read():
            with open(path_base + ".resp") as response:
                print(
                    YELLOW
                    + "Solution "
                    + BLUE
                    + answer_
                    + YELLOW
                    + " to part "
                    + BLUE
                    + part_
                    + YELLOW
                    + " was your last submission."
                    + RESET
                )
                analyse_and_print(response.read())
                return

    with open(path_base + ".out", "w") as file:
        file.write(answer_)

    submitted = False
    msg = resp = None
    while not submitted:
        print(
            "Submitting "
            + BLUE
            + answer_
            + RESET
            + " as the solution to part "
            + BLUE
            + part_
            + RESET
            + "..."
        )
        resp = requests.post(
            url=URL.format(day=day, year=year) + "/answer",
            cookies=get_cookie(),
            data={"level": part_, "answer": answer_},
        )
        if not resp.ok:
            raise ValueError("Received bad response")
        msg: str = Soup(resp.text, "html.parser").article.text

        if msg.startswith("You gave"):
            print(RED + msg + RESET)
            wait_match = WAIT_TIME.search(msg)

            pause = 60 * int(wait_match[1] or 0) + int(wait_match[2])
            print(
                YELLOW
                + "Waiting "
                + BLUE
                + str(pause)
                + YELLOW
                + " seconds to retry..."
            )
            time.sleep(pause)
        else:
            submitted = True
    analyse_and_print(msg)
    if msg.startswith("That's the"):
        open(path_base + ".solution", "w").close()  # create the flag file
        if part == 1:
            webbrowser.open(resp.url)  # open part 2 in the user's browser
    with open(path_base + ".resp", "w") as file:
        file.write(msg)


def run_and_submit(
    day: int, solution: typing.Callable[[], typing.Any], year: int = DEFAULT_YEAR
) -> None:
    """Run a solution and submit its answer.

    solution is expected to be named 'part_one' or 'part_two'
    """
    part = 1 if solution.__name__ == "part_one" else 2
    if exists(DATA_DIR / str(year) / (str(part) + ".solution")):
        # don't run the solution, because it could take a while to run
        # `answer` is ignored anyway when the solution flag exists, so
        # just pass a 0
        answer = 0
    else:
        answer = solution()
    submit_answer(day, part, answer, year)


submit = run_and_submit  # alias for old interface
