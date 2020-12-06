import datetime
import json
import pathlib
import time
import typing
import webbrowser

import requests
from bs4 import BeautifulSoup as Soup

try:
    import colorama as colour
except ImportError:
    RED = ""
    YELLOW = ""
    GREEN = ""
    BLUE = ""
    RESET = ""
else:
    colour.init()
    RED = colour.Fore.LIGHTRED_EX
    YELLOW = colour.Fore.LIGHTYELLOW_EX
    GREEN = colour.Fore.LIGHTGREEN_EX
    BLUE = colour.Fore.LIGHTBLUE_EX
    RESET = colour.Fore.RESET

from .data import DATA_DIR, DEFAULT_YEAR, URL, WAIT_TIME, get_cookie


def _make(folder: pathlib.Path) -> None:
    """Create folder if it doesn't exist."""
    if not folder.exists():
        folder.mkdir(parents=True)


def _pretty_print(message: str) -> None:
    # The first index where each server response has a different character is index 7.
    # We differentiate messages with this index.
    #  *  "green" indicates a correct answer
    #  *  "yellow" indicates a submission to an already solved problem
    #  *  "red" indicates a timeout or wrong answer
    try:
        color = {"t": GREEN, "'": YELLOW, "n": RED, "e": RED, " ": YELLOW, "g": GREEN}[message[7]]
    except KeyError as e:
        raise ValueError("Failed to parse response.") from e
    print(f"{color}{message}{RESET}")


def fetch(day: int, year: int = DEFAULT_YEAR) -> str:  # Might consider a default TODAY for day
    """Fetch and return the input for `day` of `year`.

    All inputs are cached in `aoc_helper.DATA_DIR`."""
    day_ = str(day)
    year_ = str(year)

    _make(DATA_DIR / year_)
    input_path = DATA_DIR / year_ / (day_ + ".in")

    if input_path.exists():
        return input_path.read_text()
    else:
        unlock = datetime.datetime(year, 12, day, 5)
        now = datetime.datetime.utcnow()
        if now < unlock:
            _pretty_print("Waiting for puzzle unlock...")  # Rather esoteric checks in _pretty_print to
            time.sleep((unlock - now).total_seconds())
            _pretty_print("Fetching input!")               # distinguish between these messages.
        resp = requests.get(
            URL.format(day=day, year=year) + "/input", cookies=get_cookie()
        )
        if not resp.ok:
            raise ValueError("Received bad response")
        # Note to star: May consider rstrip instead -- I don't know if AoC will ever
        # publish input that has significant whitespace at the beginning though.
        data = resp.text.strip()
        input_path.write_text(data)
        return data


def submit(
    day: int, part: int, answer: typing.Any, year: int = DEFAULT_YEAR
) -> None:
    """Submit a solution.

    Submissions are cached; submitting an already submitted solution will return the
    previous response.
    """
    day_ = str(day)
    year_ = str(year)
    part_ = str(part)
    answer_ = str(answer)

    submission_dir = DATA_DIR / year_ / day_
    _make(submission_dir)

    # Load cached solutions
    submissions = submission_dir / "submissions.json"
    if submissions.exists():
        with submissions.open() as f:
            solutions = json.load(f)
    else:
        solutions = {"1": {}, "2": {}}

    # Check if solved
    solution_file = submission_dir / f"{part}.solution"
    if solution_file.exists():
        solution = solutions_file.read_text()
        print(
            f"Day {BLUE}{day}{RESET} part {BLUE}{part}{RESET} "
            f"has already been solved.\nThe solution was: "
            f"{BLUE}{solution}{RESET}\nResponse was:\n"
        )
        return _pretty_print(solutions[part_][solution])

    # Check if answer has already been submitted
    if answer_ in solutions[part_]:
        print(
            f"{YELLOW}Solution: {BLUE}{answer}{YELLOW} to part "
            f"{BLUE}{part}{YELLOW} has already been submitted.\n"
            "Response was:\n{RESET}"
        )
        return _pretty_print(solutions[part_][answer_])

    while True:
        print(
            f"Submitting {BLUE}{answer}{RESET} as the solution to part "
            f"{BLUE}{part}{RESET}..."
        )
        resp = requests.post(
            url=URL.format(day=day, year=year) + "/answer",
            cookies=get_cookie(),
            data={"level": part_, "answer": answer_},
        )
        if not resp.ok:
            raise ValueError("Received bad response")

        msg: str = Soup(resp.text, "html.parser").article.text
        _pretty_print(msg)

        if msg[4] == "g":  # A quick check to see if the message starts with "You gave ..."
            wait_match = WAIT_TIME.search(msg)
            pause = 60 * int(wait_match[1] or 0) + int(wait_match[2])
            print(
                f"{YELLOW}Waiting {BLUE}{pause}{YELLOW} seconds to retry...{RESET}""
            )
            time.sleep(pause)
        else:
            break

    if msg[7] == "t":  # "That's the right answer!", note index 7 differentiates between all other responses.
        solution_file.write_text(answer_)
        if part == 1:
            webbrowser.open(resp.url)  # open part 2 in the user's browser

    # Cache submission
    solutions[part_][answer_] = msg
    with submissions.open() as f:
        json.dump(solutions, f)


def run_and_submit(
    day: int, solution: typing.Callable[[], typing.Any], year: int = DEFAULT_YEAR
) -> None:
    """Run a solution and submit its answer.

    solution is expected to be named 'part_one' or 'part_two'
    """
    part = 1 if solution.__name__ == "part_one" else 2
    if (DATA_DIR / str(year) / f"{part}.solution").exists():
        # don't run the solution, because it could take a while to run
        # `answer` is ignored anyway when the solution flag exists, so
        # just pass a 0
        answer = 0
    else:
        answer = solution()
    submit_answer(day, part, answer, year)


submit = run_and_submit  # alias for old interface
