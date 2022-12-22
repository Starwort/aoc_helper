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
        _print_rank(solutions[part_][solution])
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

    if msg.startswith("That's the"):
        _print_rank(msg)
        solution_file.write_text(answer_)
        if part == 1:
            if not resp.url.endswith("#part2"):
                resp.url += "#part2"  # scroll to part 2
            _open_page(resp.url)  # open part 2 in the user's browser
    else:
        _pretty_print(msg)

    # Cache submission
    solutions[part_][answer_] = msg
    with submissions.open("w") as f:
        json.dump(solutions, f)


def _print_rank(msg: str) -> None:
    match = RANK.search(msg)
    if match:
        _pretty_print(f"You got rank {match.group(1)} for this puzzle")


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
    article = Soup(resp.text, "html.parser").article
    assert article is not None
    print(article.text.strip())


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
        _print_rank(solutions[str(part)][solution_])
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
    day: int, part: int, year: int = DEFAULT_YEAR
) -> typing.Optional[typing.Tuple[str, str]]:
    """Retrieves the example input and answer for the corresponding AOC challenge."""
    testing_dir = DATA_DIR / str(year) / str(day)
    _make(testing_dir)
    testing_file = testing_dir / "tests.json"

    if testing_file.exists():
        test_info: typing.Dict[
            str, typing.Optional[typing.Tuple[str, str]]
        ] = json.loads(testing_file.read_text())
    else:
        test_info: typing.Dict[str, typing.Optional[typing.Tuple[str, str]]] = {}

    if str(part) in test_info:
        return test_info[str(part)]

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
            "for example" in preceding_text
            or "consider" in preceding_text
            or "given" in preceding_text
        ) and ":" in preceding_text:
            example_test_inputs.append(possible_test_input.text.strip())

    if not example_test_inputs:
        test_info[str(part)] = None
        testing_file.write_text(json.dumps(test_info))
        warn(
            f"An issue occurred while fetching test data for {year} day"
            f" {day} part {part}. You may either ignore this message, or pass"
            " custom test data to lazy_test.",
            RuntimeWarning,
        )
        return

    try:
        test_input = example_test_inputs[-1]

        # Attempt to retrieve answer to said example data from puzzle part.
        current_part = soup.find_all("article")[part - 1]
        last_sentence = current_part.find_all("p")[-2]
        answer = last_sentence.find_all("code")[-1]
    except IndexError:
        test_info[str(part)] = None
        testing_file.write_text(json.dumps(test_info))
        warn(
            f"An issue occurred while fetching test data for {year} day"
            f" {day} part {part}. You may either ignore this message, or pass"
            " custom test data to lazy_test.",
            RuntimeWarning,
        )
        return
    if not answer.em:
        try:
            answer = last_sentence.find_all("em")[-1]
        except IndexError:
            pass

    try:
        answer = answer.text.strip("\n").split()[-1]
    except IndexError:
        test_info[str(part)] = None
        testing_file.write_text(json.dumps(test_info))
        warn(
            f"An issue occurred while fetching test data for {year} day"
            f" {day} part {part}. You may either ignore this message, or pass"
            " custom test data to lazy_test.",
            RuntimeWarning,
        )
        return

    test_data = test_input, answer
    test_info[str(part)] = test_data
    testing_file.write_text(json.dumps(test_info))
    return test_data


def _test(part: int, answer: str, expected_answer: str) -> None:
    assert answer == expected_answer, (
        f"The expected answer for the example test input was {expected_answer} but"
        f" your answer was {answer}."
    )
    print(
        f"{GREEN}Test for part {BLUE}{part}{RESET} succeeded!"
        f" The answer for part {BLUE}{part}{RESET} with the test data was:"
        f" {BLUE}{answer}{RESET}{RESET}"
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
    _make(testing_dir)

    # If this is part 2, skip fetching/running tests if part 1 hasn't been submitted
    if part == 2 and not (testing_dir / "1.solution").exists():
        return

    # If this part has been submitted, skip running tests
    if not (testing_dir / f"{part}.solution").exists():
        if test_data is None:  # No test data passed (most common)
            test_data = get_sample_input(day, part, year)
            if test_data is None:  # No test data scraped (uncommon)
                return
        test_input, test_answer = test_data

        answer = work(
            f"{YELLOW}Running the test for part {BLUE}{part}{RESET} solution...{RESET}",
            solution,
            parse(test_input),
        )
        if answer is not None:
            _test(part, str(answer).strip(), str(test_answer).strip())
