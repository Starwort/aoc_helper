import builtins
import functools
import itertools
import re
import typing
from collections import deque

T = typing.TypeVar("T")
U = typing.TypeVar("U")


def extract_ints(raw: str) -> typing.List[int]:
    """Utility function to extract all integers from some string.

    Many inputs can be directly parsed with this function.
    """
    return list(map(int, re.findall(r"((?:-|\+)?\d+)", raw)))


def chunk(
    iterable: typing.Iterable[T], chunk_size: int
) -> typing.Iterable[tuple[T, ...]]:
    """Utility function to chunk an iterable into chunks of a given size.

    Assumes your iterable produces a number of items that is a multiple of the
    chunk size.
    """
    return zip(*[builtins.iter(iterable)] * chunk_size)


def chunk_default(
    iterable: typing.Iterable[T], chunk_size: int, default: T
) -> typing.Iterable[tuple[T, ...]]:
    """Utility function to chunk an iterable into chunks of a given size.

    Assumes your iterable produces a number of items that is a multiple of the
    chunk size.
    """
    return itertools.zip_longest(
        *[builtins.iter(iterable)] * chunk_size, fillvalue=default
    )


class SmartIterator(typing.Generic[T]):
    _SENTINEL = object()

    def __init__(self, it: typing.Iterable[T]) -> None:
        self.it = builtins.iter(it)

    def __iter__(self) -> typing.Iterable[T]:
        return self.it.__iter__()

    def map(self, func: typing.Callable[[T], U]) -> "SmartIterator[U]":
        return type(self)(map(func, self))

    def filter(
        self, pred: typing.Union[typing.Callable[[T], bool], None]
    ) -> "SmartIterator[T]":
        return type(self)(filter(pred, self))

    @typing.overload
    def reduce(self, func: typing.Callable[[T, T], T]) -> T:
        ...

    @typing.overload
    def reduce(self, func: typing.Callable[[T, T], T], initial: T) -> T:
        ...

    @typing.overload
    def reduce(self, func: typing.Callable[[U, T], U], initial: U) -> U:
        ...

    def reduce(self, func, initial=_SENTINEL):
        if initial is self._SENTINEL:
            return functools.reduce(func, self)
        else:
            return functools.reduce(func, self, initial)

    def foreach(self, func: typing.Callable[[T], typing.Any]) -> None:
        for el in self:
            func(el)

    def chunk(self, n: int) -> "SmartIterator[typing.Tuple[T, ...]]":
        return type(self)(chunk(self, n))

    def chunk_default(
        self, n: int, default: T
    ) -> "SmartIterator[typing.Tuple[T, ...]]":
        return type(self)(chunk_default(self, n, default))

    def _window(
        self, window_size: int
    ) -> typing.Generator[typing.Tuple[T, ...], None, None]:
        elements = deque()
        for _ in range(window_size):
            try:
                elements.append(self.next())
            except StopIteration:
                return

        yield tuple(elements)

        for el in self:
            elements.popleft()
            elements.append(el)
            yield tuple(elements)

    def window(self, window_size: int) -> "SmartIterator[typing.Tuple[T, ...]]":
        return type(self)(self._window(window_size))

    def next(self) -> T:
        return next(self.it)

    def skip(self, n: int = 1) -> "SmartIterator[T]":
        for _ in builtins.range(n):
            self.next()
        return self

    def nth(self, n: int) -> T:
        self.skip(n)
        return self.next()

    def take(self, n: int) -> typing.Tuple[T, ...]:
        return tuple(self.next() for _ in builtins.range(n))

    def collect(self) -> "typing.List[T]":
        return list(self)

    def chain(self, other: typing.Iterable[T]) -> "SmartIterator[T]":
        return type(self)(itertools.chain(self, other))


@functools.wraps(range)
def range(*args, **kw):
    return SmartIterator(builtins.range(*args, **kw))


iter = SmartIterator
