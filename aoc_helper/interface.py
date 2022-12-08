import builtins
import datetime
import json
import pathlib
import time
import typing
import webbrowser
from warnings import warn

import requests
from bs4 import BeautifulSoup as Soup

from .data import HEADERS

T = typing.TypeVar("T")
U = typing.TypeVar("U")

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

    def work(msg: str, worker: typing.Callable[[U], T], data: U) -> T:
        print(msg)
        return worker(data)

else:
    RED = "[red]"
    YELLOW = "[yellow]"
    GREEN = "[green]"
    BLUE = "[blue]"
    GOLD = "[gold1]"
    RESET = "[/]"

    def wait(msg: str, secs: float) -> None:
        for _ in progress.track(
            builtins.range(int(10 * secs)),
            description=msg,
            show_speed=False,
            transient=True,
        ):
            time.sleep(0.1)

    def _rich_work(msg: str, worker: typing.Callable[[U], T], data: U) -> T:
        with progress.Progress(
            progress.TextColumn("{task.description}"),
            progress.SpinnerColumn(),
            progress.TimeElapsedColumn(),
            transient=True,
        ) as bar:
            task = bar.add_task(msg)
            val = worker(data)
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


def fetch(day: int = TODAY, year: int = DEFAULT_YEAR, never_print: bool = False) -> str:
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
                    URL.format(day=1, year=2015) + "/input",
                    cookies=get_cookie(),
                    headers=HEADERS,
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
            URL.format(day=day, year=year) + "/input",
            cookies=get_cookie(),
            headers=HEADERS,
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
        if not never_print:
            print(data)
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
        match = RANK.search(solutions[part_][solution])
        if match:
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
            headers=HEADERS,
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
        headers=HEADERS,
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
    day: int,
    solution: typing.Callable[[U], typing.Any],
    data: U,
    year: int = DEFAULT_YEAR,
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
        match = RANK.search(solutions[str(part)][solution_])
        if match:
            _pretty_print(match.group(0))
    else:
        answer = work(
            f"{YELLOW}Running part"
            f" {RESET}{BLUE}{part}{RESET}{YELLOW} solution...{RESET}",
            solution,
            data,
        )
        if answer is not None:
            submit(day, part, answer, year)


def get_sample_input(
    day: int, year: int = DEFAULT_YEAR
) -> typing.Optional[typing.Tuple[str, str]]:
    """Retrieves the example input and answer for the corresponding AOC challenge."""
    resp = requests.post(
        url=URL.format(day=day, year=year), cookies=get_cookie(), headers=HEADERS
    )
    soup = Soup(resp.text, "html.parser")

    example_test_inputs = []
    # Find the example test input for that day.
    for possible_test_input in soup.find_all("pre"):
        preceding_text = (
            possible_test_input.previous_element.previous_element.text.lower()
        )
        if (
            "for example" in preceding_text or "consider" in preceding_text or "given" in preceding_text
        ) and ":" in preceding_text:
            example_test_inputs.append(possible_test_input.text.strip())

    if not example_test_inputs:
        return None

    test_input = example_test_inputs[-1]

    # Attempt to retrieve answer to said example data.
    current_part = soup.find_all("article")[-1]
    last_sentence = current_part.find_all("p")[-2]

    try:
        answer = last_sentence.find_all("code")[-1]
    except IndexError:
        return None
    if not answer.em:
        try:
            answer = last_sentence.find_all("em")[-1]
        except IndexError:
            pass

    answer = answer.text.strip().split()[-1]

    return test_input, answer


def test(
    day: int, part: int, answer: str, expected_answer: str, year: int = DEFAULT_YEAR
) -> None:
    day_ = str(day)
    year_ = str(year)
    part_ = str(part)

    testing_dir = DATA_DIR / year_ / day_
    _make(testing_dir)

    # Load cached tests
    tests = testing_dir / "tests.json"
    if tests.exists() and tests.read_text():
        with tests.open() as f:
            test_data = json.load(f)
    else:
        test_data = {"1": [], "2": []}

    test_data[part_].append({"answer": answer, "expected_answer": expected_answer})

    with tests.open("w") as f:
        json.dump(test_data, f)

    assert answer == expected_answer, (
        f"The expected answer for the example test input was {expected_answer} but"
        f" your answer was {answer}."
    )


def lazy_test(
    day: int,
    parse: typing.Callable[[str], T],
    solution: typing.Callable[[T], typing.Any],
    year: int = DEFAULT_YEAR,
    test_data: typing.Optional[typing.Tuple[str, typing.Any]] = None,
) -> None:
    """Test the function with AOC's example data only if we haven't tested it already.

    Solution is expected to be named 'part_one' or 'part_two'
    """
    part = 1 if solution.__name__ == "part_one" else 2
    testing_dir = DATA_DIR / str(year) / str(day)
    testing_file = testing_dir / "tests.json"

    # Check if the tests have ran for the specific part yet
    if not testing_file.exists() or (
        (testing_dir / "1.solution").exists() and part != 1
    ):
        if test_data is None:  # No test data passed (most common)
            test_data = get_sample_input(day, year)
            if test_data is None:  # No test data scraped (uncommon)
                warn(
                    f"An issue occurred while fetching test data for {year} day"
                    f" {day} part {part}. You may either ignore this message, or pass"
                    " custom test data to lazy_test.",
                    RuntimeWarning,
                )
                return
        test_input, test_answer = test_data

        answer = work(
            f"{YELLOW}Running the test for part"
            f" {RESET}{BLUE}{part}{RESET}{YELLOW} solution...{RESET}",
            solution,
            parse(test_input),
        )
        if answer is not None:
            test(day, part, str(answer).strip(), str(test_answer).strip(), year)
            print(
                f"{GREEN}Test for part {RESET}{YELLOW}{part}{YELLOW}{RESET} succeeded!"
                f" {RESET}Your answer for part 2 with the test data was:"
                f" {GREEN}{answer}{GREEN}{RESET} and the expected answer with the test"
                f" data was also: {GREEN}{test_answer}{GREEN}{RESET}"
            )
