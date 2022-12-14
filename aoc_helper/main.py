import pathlib
import re
import typing

try:
    import click
except ImportError:
    print("Missing dependencies for the CLI. Please `pip install aoc_helper[cli]`")
    exit(1)

from .data import DATA_DIR, DEFAULT_YEAR, RANK
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


@click.group()
def cli():
    pass


@cli.command()
@click.argument("day", type=int)
@click.option("--year", type=int, default=DEFAULT_YEAR)
def fetch(day: int, year: int):
    """Fetch and print the input for DAY of --year"""
    print(fetch_input(day, year))


@cli.command()
@click.argument("day", type=int)
@click.argument("part", type=int)
@click.argument("answer")
@click.option("--year", type=int, default=DEFAULT_YEAR)
def submit(day: int, part: int, answer: str, year: int):
    """Submit the answer for DAY part PART of --year"""
    submit_answer(day, part, answer, year)


@cli.command()
@click.argument("days", callback=parse_range)
@click.option("--year", type=int, default=DEFAULT_YEAR)
def template(days: typing.List[int], year: int):
    """Generate an answer stub for every day of DAYS in --year"""
    for day in days:
        print(f"Generating day_{day:0>2}.py")
        pathlib.Path(f"day_{day:0>2}.py").write_text(
            TEMPLATE.format(day=day, year=year)
        )


@cli.command()
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


@cli.command()
@click.argument("days", callback=parse_range)
@click.argument("year", type=int, default=DEFAULT_YEAR)
@click.option(
    "--type",
    type=click.Choice(["input", "submissions", "solutions", "1", "2", "tests", "all"]),
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
        if type in ("solutions", "all", "1"):
            (DATA_DIR / f"{year}" / f"{day}" / "1.solution").unlink(True)
        if type in ("solutions", "all", "2"):
            (DATA_DIR / f"{year}" / f"{day}" / "2.solution").unlink(True)
        if type in ("tests", "all"):
            (DATA_DIR / f"{year}" / f"{day}" / "tests.json").unlink(True)
