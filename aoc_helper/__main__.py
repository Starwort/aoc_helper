import pathlib
import re

import click

from .data import DEFAULT_YEAR
from .interface import fetch_data, submit_answer

TEMPLATE = (pathlib.Path(__file__).parent / "day_template.py").read_text()

RANGE_REGEX = re.compile(r"(0?[1-9]|1[0-9]|2[0-5])-(0?[2-9]|1[0-9]|2[0-5])")


def parse_range(_, __, value: str) -> list[int]:
    ranges = value.split(",")
    days: set[int] = set()
    for range_ in ranges:
        if match := RANGE_REGEX.match(range_):
            lb = int(match[1])
            ub = int(match[2]) + 1
            days |= set(range(lb, ub))
        elif range_.isnumeric() and 1 <= (day := int(range_)) <= 25:
            days.add(day)
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
@click.argument("year", type=int, default=DEFAULT_YEAR)
def fetch(day: int, year: int):
    print(fetch_data(day, year))


@cli.command()
@click.argument("day", type=int)
@click.argument("year", type=int, default=DEFAULT_YEAR)
@click.argument("part", type=int)
@click.argument("answer")
def submit(day: int, year: int, part: int, answer: str):
    submit_answer(day, part, answer, year)


@cli.command()
@click.argument("days", callback=parse_range)
@click.argument("year", type=int, default=DEFAULT_YEAR)
def template(days: list[int], year: int):
    for day in days:
        print(f"Generating day_{day:0>2}.py")
        pathlib.Path(f"day_{day:0>2}.py").write_text(
            TEMPLATE.format(day=day, year=year)
        )


cli()
