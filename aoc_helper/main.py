import json
import os
import pathlib
import re
import typing
from datetime import date, timedelta

try:
    import click
    from click_aliases import ClickAliasedGroup
except ImportError:
    print("Missing dependencies for the CLI. Please `pip install aoc_helper[cli]`")
    exit(1)

from .data import DATA_DIR, DEFAULT_YEAR, PRACTICE_DATA_DIR, RANK
from .interface import _estimate_practice_rank, _format_timedelta
from .interface import fetch as fetch_input
from .interface import submit as submit_answer

TEMPLATE = (pathlib.Path(__file__).parent / "day_template.py").read_text()

RANGE_REGEX = re.compile(r"(2[0-5]|1[0-9]|0?[1-9])-(2[0-5]|1[0-9]|0?[2-9])")


def parse_range(_, __, value: str) -> typing.List[int]:
    ranges = value.split(",")
    days: set[int] = set()
    for range_ in ranges:
        match = RANGE_REGEX.match(range_)
        if match:
            lb = int(match[1])
            ub = int(match[2]) + 1
            days |= set(range(lb, ub))
        elif range_.isnumeric() and 1 <= int(range_) <= 25:
            days.add(int(range_))
        elif range_ == "all":
            days = set(range(1, 26))
        else:
            raise click.BadParameter(
                "every part must be a single day, a range of days in the form "
                "a-b, or the word 'all'"
            )
    return sorted(days)


@click.group(cls=ClickAliasedGroup)
def cli():
    pass


@cli.command(aliases=["f", "get", "get-input", "download"])
@click.argument("day", type=int)
@click.option("--year", type=int, default=DEFAULT_YEAR)
def fetch(day: int, year: int):
    """Fetch and print the input for day DAY of --year"""
    print(fetch_input(day, year))


@cli.command(aliases=["s", "send"])
@click.argument("day", type=int)
@click.argument("part", type=click.Choice(["1", "2"]))
@click.argument("answer")
@click.option("--year", type=int, default=DEFAULT_YEAR)
@click.option("--practice", is_flag=True)
def submit(
    day: int, part: typing.Literal["1", "2"], answer: str, year: int, practice: bool
):
    """Submit the answer for day DAY part PART of --year"""
    _ = practice  # used via sys.argv
    submit_answer(day, int(part), answer, year)


@cli.command(aliases=["t", "create"])
@click.argument("days", callback=parse_range)
@click.option("--year", type=int, default=DEFAULT_YEAR)
def template(days: typing.List[int], year: int):
    """Generate an answer stub for every day of DAYS in --year"""
    for day in days:
        print(f"Generating day_{day:0>2}.py")
        pathlib.Path(f"day_{day:0>2}.py").write_text(
            TEMPLATE.format(day=day, year=year)
        )


@cli.command(aliases=["get-browser", "set-browser"])
@click.argument("state", type=bool, default=None, required=False)
def browser(state: typing.Optional[bool]):
    """Enable, disable, or check browser automation"""
    file = DATA_DIR / ".nobrowser"
    if state is None:
        print(f"Web browser automation is {'dis' if file.exists() else 'en'}abled.")
    elif state:
        file.unlink(True)
        print("Enabled web browser automation")
    else:
        file.touch()
        print("Disabled web browser automation")


@cli.command(
    aliases=[
        "clear",
        "purge",
        "delete",
        "clear-cache",
        "clean-cache",
        "purge-cache",
        "delete-cache",
    ]
)
@click.argument("days", callback=parse_range)
@click.argument("year", type=int, default=DEFAULT_YEAR)
@click.option(
    "--type",
    type=click.Choice(
        ["input", "submissions", "solutions", "1", "2", "tests", "practice", "all"]
    ),
    help="What to delete",
    default="input",
)
def clean(days: typing.List[int], year: int, type: str):
    """Clean the cached --type data for DAYS of YEAR"""
    for day in days:
        if type in ("input", "all"):
            (DATA_DIR / f"{year}" / f"{day}.in").unlink(True)
        if type in ("submissions", "all"):
            file = DATA_DIR / f"{year}" / f"{day}" / "submissions.json"
            if (
                not file.exists()
                or not RANK.search(file.read_text())
                or click.confirm(
                    f"Are you sure you want to delete your submissions for {year} day"
                    f" {day}? Your cached rank will be forgotten"
                )
            ):
                file.unlink(True)
        if type in ("practice", "all"):
            folder = PRACTICE_DATA_DIR / f"{year}" / f"{day}"
            if not folder.exists() or click.confirm(
                f"Are you sure you want to delete your practice data for {year} day"
                f" {day}? {len(os.listdir(folder))} entries will be forgotten"
            ):
                for file in os.listdir(folder):
                    (folder / file).unlink(True)
                folder.rmdir()
        if type in ("solutions", "all", "1"):
            (DATA_DIR / f"{year}" / f"{day}" / "1.solution").unlink(True)
        if type in ("solutions", "all", "2"):
            (DATA_DIR / f"{year}" / f"{day}" / "2.solution").unlink(True)
        if type in ("tests", "all"):
            (DATA_DIR / f"{year}" / f"{day}" / "tests.json").unlink(True)


@cli.command(name="practice-results")
@click.argument("day", type=int)
@click.option("--year", type=int, default=DEFAULT_YEAR)
def practice_results(day: int, year: int):
    """Show all practice results for day DAY of --year"""
    import locale

    locale.setlocale(
        locale.LC_TIME, ""
    )  # https://github.com/python/cpython/issues/73643
    folder = PRACTICE_DATA_DIR / f"{year}" / f"{day}"
    if not folder.exists():
        print("No practice results found")
        return

    def format_result(result: typing.Optional[typing.Tuple[int, int, int]]):
        if not result:
            return "no rank"
        estimated, best, worst = result
        if best == worst:
            return f"rank {best}"
        if worst > 100:
            worst = "100+"
        return f"approximately rank {estimated} - {best} to {worst}"

    for file in sorted(folder.iterdir()):
        attempt_year, attempt_month, attempt_day = map(int, file.stem.split("-"))
        attempt_date = date(attempt_year, attempt_month, attempt_day).strftime("%x")
        results: typing.List[float] = json.loads(file.read_text())
        if len(results) == 1:
            solve_time = timedelta(seconds=results[0])
            result = format_result(_estimate_practice_rank(day, 1, year, solve_time))
            print(
                f"{attempt_date} - Part 1: {_format_timedelta(solve_time)} ({result}),"
                " Part 2: (unsolved)"
            )
        elif len(results) == 2:
            solve_time_1 = timedelta(seconds=results[0])
            solve_time_2 = timedelta(seconds=results[1])
            result_1 = format_result(
                _estimate_practice_rank(day, 1, year, solve_time_1)
            )
            result_2 = format_result(
                _estimate_practice_rank(day, 2, year, solve_time_2)
            )
            print(
                f"{attempt_date} - "
                f"Part 1: {_format_timedelta(solve_time_1)} ({result_1}), "
                f"Part 2: {_format_timedelta(solve_time_2)} ({result_2})"
            )
