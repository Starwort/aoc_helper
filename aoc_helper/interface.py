import builtins
import datetime
import json
import pathlib
import time
import typing
import webbrowser

import requests
from bs4 import BeautifulSoup as Soup

T = typing.TypeVar("T")

try:
    from rich import print, progress
except ImportError:
    RED = ""
    YELLOW = ""
    GREEN = ""
    BLUE = ""
    GOLD = ""
    RESET = ""

    def wait(msg: str, secs: float) -> None:
        print(msg)
        time.sleep(secs)

    def work(msg: str, worker: typing.Callable[[], T]) -> T:
        print(msg)
        return worker()

else:
    RED = "[red]"
    YELLOW = "[yellow]"
    GREEN = "[green]"
    BLUE = "[blue]"
    GOLD = "[gold1]"
    RESET = "[/]"

    def wait(msg: str, secs: float) -> None:
        for _ in progress.track(
            builtins.range(int(100 * secs)),
            description=msg,
            show_speed=False,
            transient=True,
        ):
            time.sleep(0.01)

    def _rich_work(msg: str, worker: typing.Callable[[], T]) -> T:
        with progress.Progress(
            progress.TextColumn("{task.description}"),
            progress.SpinnerColumn(),
            progress.TimeElapsedColumn(),
            transient=True,
        ) as bar:
            task = bar.add_task(msg)
            val = worker()
            bar.advance(task)
            return val

    work = _rich_work


from .data import DATA_DIR, DEFAULT_YEAR, RANK, TODAY, URL, WAIT_TIME, get_cookie


def _open_page(page: str) -> None:
    """Open the page if the user hasn't opted out"""
    if not (DATA_DIR / ".nobrowser").exists():
        webbrowser.open(page)


def _make(folder: pathlib.Path) -> None:
    """Create folder if it doesn't exist."""
    if not folder.exists():
        folder.mkdir(parents=True)


def _pretty_print(message: str) -> None:
    """Analyse and print message"""
    if message.startswith("That's the"):
        print(GREEN + message + RESET)
    elif message.startswith("You don't"):
        print(YELLOW + message + RESET)
    elif message.startswith("That's not"):
        print(RED + message + RESET)
    elif message.startswith("You got rank"):
        print(GOLD + message + RESET)
    else:
        raise ValueError("Failed to parse response.")


def fetch(day: int = TODAY, year: int = DEFAULT_YEAR) -> str:
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
            # On the first day, run a stray request to validate the user's token
            if day == 1:
                resp = requests.get(
                    URL.format(day=1, year=2015) + "/input", cookies=get_cookie()
                )
                if resp.status_code == 400:
                    token_file = DATA_DIR / "token.txt"
                    print(
                        f"{RED}Your token has expired. Please enter your new"
                        f" token.{RESET}"
                    )
                    token = input(">>> ")
                    token_file.write_text(token)
                    return fetch(day, year)
                now = datetime.datetime.utcnow()
            wait(
                f"{YELLOW}Waiting for puzzle unlock...{RESET}",
                (unlock - now).total_seconds(),
            )
            print(GREEN + "Fetching input!" + RESET)
            _open_page(URL.format(day=day, year=year))
        resp = requests.get(
            URL.format(day=day, year=year) + "/input", cookies=get_cookie()
        )
        if not resp.ok:
            if resp.status_code == 400:
                token_file = DATA_DIR / "token.txt"
                print(
                    f"{RED}Your token has expired. Please enter your new token.{RESET}"
                )
                token = input(">>> ")
                token_file.write_text(token)
                return fetch(day, year)
            raise ValueError("Received bad response")
        data = resp.text.strip("\n")
        input_path.write_text(data)
        return data


def submit(day: int, part: int, answer: typing.Any, year: int = DEFAULT_YEAR) -> None:
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
        solution = solution_file.read_text()
        print(
            f"Day {BLUE}{day}{RESET} part {BLUE}{part}{RESET} "
            "has already been solved.\nThe solution was: "
            f"{BLUE}{solution}{RESET}"
        )
        if match := RANK.search(solutions[part_][solution]):
            _pretty_print(match.group(0))
        return

    # Check if answer has already been submitted
    if answer_ in solutions[part_]:
        print(
            f"{YELLOW}Solution: {BLUE}{answer}{RESET} to part "
            f"{BLUE}{part}{RESET} has already been submitted.\n"
            f"Response was:{RESET}"
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
            if resp.status_code == 400:
                token_file = DATA_DIR / "token.txt"
                token = input(
                    "Your token has expired. Please enter your new token\n>>> "
                )
                token_file.write_text(token)
                return submit(day, part, answer, year)
            raise ValueError("Received bad response")

        article = Soup(resp.text, "html.parser").article
        assert article is not None
        msg = article.text

        if msg.startswith("You gave"):
            print(RED + msg + RESET)
            wait_match = WAIT_TIME.search(msg)
            assert wait_match is not None
            pause = 60 * int(wait_match[1] or 0) + int(wait_match[2])
            wait(
                f"{YELLOW}Waiting {BLUE}{pause}{RESET} seconds to retry...{RESET}",
                pause,
            )
        else:
            break
    _pretty_print(msg)

    if msg.startswith("That's the"):
        solution_file.write_text(answer_)
        if part == 1:
            if not resp.url.endswith("#part2"):
                resp.url += "#part2"  # scroll to part 2
            _open_page(resp.url)  # open part 2 in the user's browser

    # Cache submission
    solutions[part_][answer_] = msg
    with submissions.open("w") as f:
        json.dump(solutions, f)


def submit_25(year: str):
    print(f"{GREEN}Finishing Advent of Code {BLUE}{year}{RESET}!{RESET}")
    resp = requests.post(
        url=URL.format(day="25", year=year) + "/answer",
        cookies=get_cookie(),
        data={"level": "2", "answer": "0"},
    )
    if not resp.ok:
        if resp.status_code == 400:
            token_file = DATA_DIR / "token.txt"
            token = input("Your token has expired. Please enter your new token\n>>> ")
            token_file.write_text(token)
            return submit_25(year)
        raise ValueError("Received bad response")

    print("Response from the server:")
    print(resp.text.strip())


def lazy_submit(
    day: int, solution: typing.Callable[[], typing.Any], year: int = DEFAULT_YEAR
) -> None:
    """Run the function only if we haven't seen a solution.

    solution is expected to be named 'part_one' or 'part_two'
    """
    part = 1 if solution.__name__ == "part_one" else 2
    submission_dir = DATA_DIR / str(year) / str(day)
    if day == 25 and part == 2:
        # Don't try to submit part 2 if part 1 isn't solved
        if (submission_dir / "1.solution").exists():
            submit_25(str(year))
    solution_file = submission_dir / f"{part}.solution"
    # Check if solved
    if solution_file.exists():
        # Load cached solutions
        submissions = submission_dir / "submissions.json"
        with submissions.open() as f:
            solutions = json.load(f)

        solution_ = solution_file.read_text()
        print(
            f"Day {BLUE}{day}{RESET} part {BLUE}{part}{RESET} "
            "has already been solved.\nThe solution was: "
            f"{BLUE}{solution_}{RESET}"
        )
        if match := RANK.search(solutions[str(part)][solution_]):
            _pretty_print(match.group(0))
    elif (
        answer := work(
            f"{YELLOW}Running part"
            f" {RESET}{BLUE}{part}{RESET}{YELLOW} solution...{RESET}",
            solution,
        )
    ) is not None:
        submit(day, part, answer, year)
