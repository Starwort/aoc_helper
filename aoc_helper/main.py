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

from .data import (
    DATA_DIR,
    DEFAULT_YEAR,
    HEADERS,
    PRACTICE_DATA_DIR,
    RANK,
    URL,
    get_cookie,
)
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


@cli.command(aliases=["read-puzzle", "puzzle"])
@click.argument("day", type=int)
@click.option("--year", type=int, default=DEFAULT_YEAR)
@click.option(
    "-c",
    "--colour",
    "--color",
    type=click.Choice(["auto", "always", "never"]),
    default="auto",
)
def read(day: int, year: int, colour: typing.Literal["auto", "always", "never"]):
    """Read the puzzle for day DAY or --year in your terminal"""
    import sys
    from os import getenv

    import requests
    from bs4 import BeautifulSoup, Tag

    try:
        from rich.console import Console, ConsoleOptions, RenderResult
        from rich.markdown import TextElement
        from rich.panel import Panel
        from rich.segment import Segment
    except ImportError:
        print(
            "Missing dependency rich. Please `pip install rich`,"
            " `pip install aoc_helper[full]` or `pip install aoc_helper[fancy]`",
            file=sys.stderr,
        )
        return
    puzzle_info = requests.get(
        URL.format(year=year, day=day), headers=HEADERS, cookies=get_cookie()
    )
    soup = BeautifulSoup(puzzle_info.text, "html.parser")
    puzzle: Tag = soup.find("main")  # type: ignore
    for emphasis in puzzle.find_all("em"):
        emphasis.string.replace_with(f"[bold gold1]{emphasis.text}[/]")
    terminal = Console()
    if colour == "auto":
        pager = getenv("PAGER") or ""
        colour = (
            "always"
            if terminal.is_terminal
            and any(
                flag in pager
                for flag in ("-r", "-R", "--raw-control-chars", "--RAW-CONTROL-CHARS")
            )
            else "never"
        )

    class CodeBlock(TextElement):
        def __init__(self, text: str):
            self.text = text

        def __rich_console__(
            self, console: Console, options: ConsoleOptions
        ) -> RenderResult:
            lines = self.text.split("\n")
            line_count = len(lines)
            width = len(str(line_count))
            render_options = options.update(width=console.width - 7 - width)
            for line_no, line in enumerate(lines, 1):
                inner_lines = console.render_lines(line, render_options)
                for i, line in enumerate(inner_lines):
                    yield Segment("    ")
                    yield Segment(
                        f"{line_no:>{width}} │ " if i == 0 else " " * (width) + " │ "
                    )
                    yield from line
                    yield Segment("\n")

    class BulletItem(TextElement):
        def __init__(self, text: str):
            self.text = text

        def __rich_console__(
            self, console: Console, options: ConsoleOptions
        ) -> RenderResult:
            render_options = options.update(width=console.width - 2)
            lines = console.render_lines(self.text, render_options)
            for i, line in enumerate(lines):
                yield Segment("- " if i == 0 else "  ")
                yield from line
                yield Segment("\n")

    class NumberedItem(TextElement):
        def __init__(self, text: str, number: int, width: int):
            self.text = text
            self.number = number
            self.width = width

        def __rich_console__(
            self, console: Console, options: ConsoleOptions
        ) -> RenderResult:
            render_options = options.update(width=console.width - self.width - 2)
            lines = console.render_lines(self.text, render_options)
            for i, line in enumerate(lines):
                yield f"{self.number:>{self.width}}. " if i == 0 else (
                    " " * self.width + "  "
                )
                yield from line
                yield "\n"

    with terminal.pager(styles=colour == "always") as pager:
        first = True
        for el in puzzle.children:
            if not isinstance(el, Tag):
                continue
            if el.name == "article":
                for part_el in el.children:
                    if not isinstance(part_el, Tag):
                        continue
                    if not first:
                        terminal.print()
                    first = False
                    if part_el.name == "h2":
                        terminal.rule(
                            "[bold gold1 underline]" + part_el.text.strip("- "),
                            style="bold gold1",
                        )
                    elif part_el.name == "p":
                        terminal.print(part_el.text)
                    elif part_el.name == "pre":
                        terminal.print(CodeBlock(part_el.text.strip("\n")))
                    elif part_el.name == "ul":
                        for li in part_el.find_all("li"):
                            terminal.print(BulletItem(li.text))
                    elif part_el.name == "ol":
                        lis = list(part_el.find_all("li"))
                        width = len(str(len(lis)))
                        for i, li in enumerate(lis, 1):
                            terminal.print(NumberedItem(li.text, i, width))

            elif el.name == "p":
                if el.text.startswith("Your puzzle answer was"):
                    terminal.print()
                    terminal.print()
                    terminal.print(el.text)
                    terminal.print()
                elif el.text.startswith("Both parts of this puzzle are complete!"):
                    terminal.print(f"[bold gold1]{el.text}[/]")
                    return


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
