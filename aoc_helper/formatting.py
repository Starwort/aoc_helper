import builtins
import time
import typing
from builtins import print as print_raw

T = typing.TypeVar("T")
U = typing.TypeVar("U")

try:
    from rich import print, progress
except ImportError:
    print = print_raw  # suppress a Pylance warning

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

__all__ = (
    "RED",
    "YELLOW",
    "GREEN",
    "BLUE",
    "GOLD",
    "RESET",
    "wait",
    "work",
    "print",
    "print_raw",
)
