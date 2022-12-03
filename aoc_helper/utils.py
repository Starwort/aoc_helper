import builtins
import copy
import functools
import itertools
import math
import operator
import re
import sys
import typing
from collections import Counter, UserList, deque
from heapq import heapify, heappop, heappush, nlargest, nsmallest

from typing_extensions import ParamSpec

from aoc_helper.types import (
    AddableT,
    AddableU,
    MultipliableT,
    MultipliableU,
    SupportsMean,
    SupportsProdNoDefaultT,
    SupportsRichComparison,
    SupportsRichComparisonT,
    SupportsSumNoDefaultT,
)

T = typing.TypeVar("T")
U = typing.TypeVar("U")
GenericU = typing.Generic[T]
P = ParamSpec("P")


MaybeIterator = typing.Union[T, typing.Iterable["MaybeIterator[T]"]]


def extract_ints(raw: str) -> "list[int]":
    """Utility function to extract all integers from some string.

    Many inputs can be directly parsed with this function.
    """
    return list(map(int, re.findall(r"((?:-|\+)?\d+)", raw)))


def chunk(
    iterable: typing.Iterable[T], chunk_size: int
) -> typing.Iterable[tuple[T, ...]]:
    """Utility function to chunk an iterable into chunks of a given size.

    If there are not enough elements in the iterable to fill the last chunk,
    the last chunk will be dropped.
    """
    return zip(*[builtins.iter(iterable)] * chunk_size)


def chunk_default(
    iterable: typing.Iterable[T], chunk_size: int, default: T
) -> typing.Iterable[tuple[T, ...]]:
    """Utility function to chunk an iterable into chunks of a given size.

    If there are not enough elements in the iterable to fill the last chunk,
    the missing elements will be replaced with the default value.
    """
    return itertools.zip_longest(
        *[builtins.iter(iterable)] * chunk_size, fillvalue=default
    )


class list(UserList, typing.Generic[T]):
    """Smart/fluent list class"""

    _SENTINEL = object()

    def iter(self) -> "iter[T]":
        """Return an iterator over the list."""
        return iter(self)

    def mapped(self, func: typing.Callable[[T], U]) -> "list[U]":
        """Return a list containing the result of calling func on each
        element in the list. The function is called on each element immediately.
        """
        return list(map(func, self))

    def mapped_each(
        self: "list[typing.Iterable[T]]", func: typing.Callable[[T], U]
    ) -> "list[list[U]]":
        """Return a list containing the results of mapping each element of self
        with func. The function is called on each element immediately.
        """
        return self.mapped(lambda i: list(i).mapped(func))

    def filtered(
        self, pred: typing.Union[typing.Callable[[T], bool], T, None] = None
    ) -> "list[T]":
        """Return a list containing only the elements for which pred
        returns True.

        If pred is None, return a list containing only elements that are
        truthy.

        If pred is a T (and T is not a callable or None), return a list
        containing only elements that compare equal to pred.
        """
        if not callable(pred) and pred is not None:
            pred = (lambda j: lambda i: i == j)(pred)
        return list(filter(pred, self))

    def find(
        self, pred: typing.Union[typing.Callable[[T], bool], T, None] = None
    ) -> typing.Optional[T]:
        """Return the first element of self for which pred returns True.

        If pred is None, return the first element which is truthy.

        If pred is a T (and T is not a callable or None), return the first element
        that compares equal to pred.

        If no such element exists, return None.
        """
        if pred is None:
            pred = lambda i: bool(i)
        elif not callable(pred):
            pred = (lambda j: lambda i: i == j)(pred)
        for i in self:
            if pred(i):
                return i

    def windowed(self, window_size: int) -> "list[typing.Tuple[T, ...]]":
        """Return an list containing the elements of this list in
        a sliding window of size window_size. If there are not enough elements
        to create a full window, the list will be empty.
        """
        return list(self.iter().window(window_size))

    def shifted_zip(self, shift: int = 1) -> "iter[typing.Tuple[T, ...]]":
        """Return an iterator containing pairs of elements separated by shift.

        If there are fewer than shift elements, the iterator will be empty.
        """
        return iter(zip(self, self[shift:]))

    @typing.overload
    def reduce(self, func: typing.Callable[[T, T], T]) -> T:
        ...

    @typing.overload
    def reduce(self, func: typing.Callable[[U, T], U], initial: U) -> U:
        ...

    def reduce(self, func, initial=_SENTINEL):
        """Reduce the list to a single value, using the reduction
        function provided.
        """
        if initial is self._SENTINEL:
            return functools.reduce(func, self)
        return functools.reduce(func, self, initial)

    @typing.overload
    def accumulated(self) -> "list[T]":
        ...

    @typing.overload
    def accumulated(self, func: typing.Callable[[T, T], T]) -> "list[T]":
        ...

    @typing.overload
    def accumulated(self, func: typing.Callable[[T, T], T], initial: T) -> "list[T]":
        ...

    @typing.overload
    def accumulated(self, func: typing.Callable[[U, T], U], initial: U) -> "list[U]":
        ...

    def accumulated(self, func=operator.add, initial=_SENTINEL):
        """Return the accumulated results of calling func on the elements in
        this list.

        initial is only usable on versions of Python equal to or greater than 3.8.
        """
        if initial is self._SENTINEL:
            return list(itertools.accumulate(self, func))
        return list(itertools.accumulate(self, func, initial))  # type: ignore

    def chunked(self, n: int) -> "list[typing.Tuple[T, ...]]":
        """Return a list containing the elements of this list in chunks
        of size n. If there are not enough elements to fill the last chunk, it
        will be dropped.
        """
        return list(chunk(self, n))

    def chunked_default(self, n: int, default: T) -> "list[typing.Tuple[T, ...]]":
        """Return a list containing the elements of this list in chunks
        of size n. If there are not enough elements to fill the last chunk, the
        missing elements will be replaced with the default value.
        """
        return list(chunk_default(self, n, default))

    @typing.overload
    def sum(
        self: "list[SupportsSumNoDefaultT]",
    ) -> typing.Union[SupportsSumNoDefaultT, typing.Literal[0]]:
        ...

    @typing.overload
    def sum(
        self: "list[AddableT]", initial: AddableU
    ) -> typing.Union[AddableT, AddableU]:
        ...

    def sum(self, initial=_SENTINEL):
        """Return the sum of all elements in this list.

        If initial is provided, it is used as the initial value.
        """
        if initial is self._SENTINEL:
            return sum(self)
        return sum(self, typing.cast(AddableU, initial))

    @typing.overload
    def prod(
        self: "list[SupportsProdNoDefaultT]",
    ) -> typing.Union[T, typing.Literal[1]]:
        ...

    @typing.overload
    def prod(
        self: "list[MultipliableT]", initial: MultipliableU
    ) -> typing.Union[MultipliableT, MultipliableU]:
        ...

    def prod(self, initial=_SENTINEL):
        """Return the product of all elements in this list.

        If initial is provided, it is used as the initial value.
        """
        if initial is self._SENTINEL:
            return math.prod(self)
        # math.prod isn't actually guaranteed to run for non-numerics, so we
        # have to ignore the type error here.
        return math.prod(self, start=initial)  # type: ignore

    @typing.overload
    def sorted(
        self: "list[SupportsRichComparisonT]",
        *,
        reverse: bool = False,
    ) -> "list[SupportsRichComparisonT]":
        ...

    @typing.overload
    def sorted(
        self,
        key: typing.Callable[[T], SupportsRichComparison],
        reverse: bool = False,
    ) -> "list[T]":
        ...

    def sorted(self, key=None, reverse=False):
        """Return a list containing the elements of this list sorted
        according to the given key and reverse parameters.
        """
        return list(sorted(self, key=key, reverse=reverse))

    def reversed(self) -> "list[T]":
        """Return a list containing the elements of this list in
        reverse order.
        """
        return list(reversed(self))

    @typing.overload
    def min(
        self: "iter[SupportsRichComparisonT]",
    ) -> T:
        ...

    @typing.overload
    def min(
        self,
        key: typing.Callable[[T], SupportsRichComparisonT],
    ) -> T:
        ...

    def min(self, key=None) -> T:
        """Return the minimum element of this list, according to the given
        key.
        """
        return min(self, key=key)

    @typing.overload
    def max(
        self: "iter[SupportsRichComparisonT]",
    ) -> T:
        ...

    @typing.overload
    def max(
        self,
        key: typing.Callable[[T], SupportsRichComparisonT],
    ) -> T:
        ...

    def max(self, key=None) -> T:
        """Return the maximum element of this list, according to the given
        key.
        """
        return max(self, key=key)

    def len(self) -> int:
        """Return the length of this list."""
        return len(self)

    def mean(self: "list[SupportsMean]") -> SupportsMean:
        """Statistical mean of this list.

        T must be summable and divisible by an integer,
        and there must be at least one element in this list.
        """
        if self.len() == 0:
            raise ValueError("Called mean() on an empty list")
        return self.sum() / self.len()  # type: ignore

    @typing.overload
    def median(self: "list[SupportsRichComparisonT]") -> T:
        ...

    @typing.overload
    def median(self, key: typing.Callable[[T], SupportsRichComparisonT]) -> T:
        ...

    def median(self, key=None) -> T:
        """Statistical median of this list.

        T must be orderable and there must be at least one
        element in this list.
        Further more, if this list contains an odd number
        of elements, T must also be summable and divisible
        by an integer.
        """
        if self.len() == 0:
            raise ValueError("Called median() on an empty list")
        if self.len() % 2:
            return self.sorted(key=key)[self.len() // 2]  # type: ignore
        else:
            sorted_self = self.sorted(key=key)  # type: ignore
            return (sorted_self[self.len() // 2] + sorted_self[self.len() // 2 - 1]) / 2

    def mode(self) -> "list[T]":
        """Statistical mode of this list.

        T must be hashable and there must be at least one
        element in this list.
        """
        if self.len() == 0:
            raise ValueError("Called mode() on an empty list")
        counted = Counter(self).most_common()
        n_ties = max(i[1] for i in counted)
        return list(i[0] for i in counted if i[1] == n_ties)

    @typing.overload
    def flat(self: "list[typing.Iterable[T]]") -> "list[T]":
        ...

    @typing.overload
    def flat(
        self: "list[typing.Iterable[T]]", recursive: typing.Literal[False] = False
    ) -> "list[T]":
        ...

    @typing.overload
    def flat(
        self: "list[typing.Iterable[MaybeIterator[T]]]",
        recursive: typing.Literal[True] = True,
    ) -> "list[T]":
        ...

    def flat(self, recursive=False):
        """Flattened version of this list.

        If recursive is specified, flattens recursively instead
        of by one layer.
        """
        if not recursive:
            return list(item for list in self for item in list)
        return list(
            subitem
            for item in self
            for subitem in (
                item.tee(1)[0].flatten(True)
                if isinstance(item, iter)
                else list(item).flat(True)
                if isinstance(item, (builtins.list, list))
                else item
            )
        )

    def enumerated(self, start: int = 0) -> "list[typing.Tuple[int, T]]":
        return list(enumerate(self, start))

    def deepcopy(self) -> "list[T]":
        return copy.deepcopy(self)

    def nlargest(self, n: int) -> "list[T]":
        """Return the n largest elements of self."""
        return list(nlargest(n, self))

    def nsmallest(self, n: int) -> "list[T]":
        """Return the n smallest elements of self."""
        return list(nsmallest(n, self))

    def __repr__(self) -> str:
        return f"list({super().__repr__()})"


class iter(typing.Generic[T], typing.Iterator[T], typing.Iterable[T]):
    """Smart/fluent iterator class"""

    _SENTINEL = object()

    def __init__(self, it: typing.Iterable[T]) -> None:
        self.it = builtins.iter(it)

    def __iter__(self) -> typing.Iterator[T]:
        return self.it.__iter__()

    def __next__(self) -> T:
        return next(self)

    def map(self, func: typing.Callable[[T], U]) -> "iter[U]":
        """Return an iterator containing the result of calling func on each
        element in this iterator.
        """
        return iter(map(func, self))

    def map_each(
        self: "iter[typing.Iterable[T]]", func: typing.Callable[[T], U]
    ) -> "iter[iter[U]]":
        """Return an iterator containing the result of calling func on each
        element in each element in this iterator.
        """
        return iter(self.map(lambda i: iter(i).map(func)))

    def filter(
        self, pred: typing.Union[typing.Callable[[T], bool], T, None] = None
    ) -> "iter[T]":
        """Return an iterator containing only the elements for which pred
        returns True.

        If pred is None, return an iterator containing only elements that are
        truthy.

        If pred is a T (and T is not a callable or None), return an iterator
        containing only elements that compare equal to pred.
        """
        if not callable(pred) and pred is not None:
            pred = (lambda j: lambda i: i == j)(pred)
        return iter(filter(pred, self))

    def find(
        self, pred: typing.Union[typing.Callable[[T], bool], T, None] = None
    ) -> typing.Optional[T]:
        """Return the first element of self for which pred returns True.

        If pred is None, return the first element which is truthy.

        If pred is a T (and T is not a callable or None), return the first element
        that compares equal to pred.

        If no such element exists, return None.
        """
        if pred is None:
            pred = lambda i: bool(i)
        elif not callable(pred):
            pred = (lambda j: lambda i: i == j)(pred)
        for i in self:
            if pred(i):
                return i

    @typing.overload
    def reduce(self, func: typing.Callable[[T, T], T]) -> T:
        ...

    @typing.overload
    def reduce(self, func: typing.Callable[[U, T], U], initial: U) -> U:
        ...

    def reduce(self, func, initial=_SENTINEL):
        """Reduce the iterator to a single value, using the reduction
        function provided.
        """
        if initial is self._SENTINEL:
            return functools.reduce(func, self)
        return functools.reduce(func, self, initial)

    @typing.overload
    def accumulate(self) -> "iter[T]":
        ...

    @typing.overload
    def accumulate(self, func: typing.Callable[[T, T], T]) -> "iter[T]":
        ...

    @typing.overload
    def accumulate(self, func: typing.Callable[[T, T], T], initial: T) -> "iter[T]":
        ...

    @typing.overload
    def accumulate(self, func: typing.Callable[[U, T], U], initial: U) -> "iter[U]":
        ...

    def accumulate(self, func=operator.add, initial=_SENTINEL):
        """Return the accumulated results of calling func on the elements in
        this iterator.

        initial is only usable on versions of Python equal to or greater than 3.8.
        """
        if initial is self._SENTINEL:
            return iter(itertools.accumulate(self, func))
        return iter(itertools.accumulate(self, func, initial))  # type: ignore

    def foreach(self, func: typing.Callable[[T], typing.Any]) -> None:
        """Run func on every value in this iterator, immediately."""
        for el in self:
            func(el)

    def chunk(self, n: int) -> "iter[typing.Tuple[T, ...]]":
        """Return an iterator containing the elements of this iterator in chunks
        of size n. If there are not enough elements to fill the last chunk, it
        will be dropped.
        """
        return iter(chunk(self, n))

    def chunk_default(self, n: int, default: T) -> "iter[typing.Tuple[T, ...]]":
        """Return an iterator containing the elements of this iterator in chunks
        of size n. If there are not enough elements to fill the last chunk, the
        missing elements will be replaced with the default value.
        """
        return iter(chunk_default(self, n, default))

    def _window(
        self, window_size: int
    ) -> typing.Generator[typing.Tuple[T, ...], None, None]:
        elements: typing.Deque[T] = deque()
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

    def window(self, window_size: int) -> "iter[typing.Tuple[T, ...]]":
        """Return an iterator containing the elements of this iterator in
        a sliding window of size window_size. If there are not enough elements
        to create a full window, the iterator will be empty.
        """
        return iter(self._window(window_size))

    def shifted_zip(self, shift: int = 1) -> "iter[typing.Tuple[T, ...]]":
        """Return an iterator containing pairs of elements separated by shift.

        If there are fewer than shift elements, the iterator will be empty.
        """
        return self.window(shift + 1).map(lambda x: (x[0], x[-1]))

    def next(self) -> T:
        """Return the next element in the iterator, or raise StopIteration."""
        return next(self.it)

    @typing.overload
    def next_or(self, default: T) -> T:
        ...

    @typing.overload
    def next_or(self, default: U) -> typing.Union[T, U]:
        ...

    def next_or(self, default):
        """Return the next element in the iterator, or default."""
        try:
            return next(self.it, default)
        except StopIteration:
            return default

    def skip(self, n: int = 1) -> "iter[T]":
        """Skip and discard n elements from this iterator.

        Raises StopIteration if there are not enough elements.
        """
        for _ in builtins.range(n):
            self.next()
        return self

    def nth(self, n: int) -> T:
        """Return the nth element of this iterator.

        Discards all elements up to the nth element, and raises StopIteration
        if there are not enough elements.
        """
        self.skip(n)
        return self.next()

    def take(self, n: int) -> typing.Tuple[T, ...]:
        """Return the next n elements of this iterator.

        Raises StopIteration if there are not enough elements.
        """
        return tuple(self.next() for _ in builtins.range(n))

    @typing.overload
    def collect(self) -> list[T]:
        ...

    @typing.overload  # TODO: why doesn't this work?
    def collect(self, collection_type: typing.Type[GenericU]) -> "GenericU[T]":
        ...

    def collect(self, collection_type=None):
        """Return a list containing all remaining elements of this iterator."""
        if collection_type is None:
            collection_type = list
        return collection_type(self)

    def chain(self, other: typing.Iterable[T]) -> "iter[T]":
        """Return an iterator containing the elements of this iterator followed
        by the elements of other.
        """
        return iter(itertools.chain(self, other))

    @typing.overload
    def sum(
        self: "iter[SupportsSumNoDefaultT]",
    ) -> typing.Union[SupportsSumNoDefaultT, typing.Literal[0]]:
        ...

    @typing.overload
    def sum(
        self: "iter[AddableT]", initial: AddableU
    ) -> typing.Union[AddableT, AddableU]:
        ...

    def sum(self, initial=_SENTINEL):
        """Return the sum of all elements in this iterator.

        If initial is provided, it is used as the initial value.
        """
        if typing.TYPE_CHECKING:
            # HACK to make mypy happy with iterating this iter
            self = list(self)
        if initial is self._SENTINEL:
            return sum(self)
        return sum(self, typing.cast(AddableU, initial))

    @typing.overload
    def prod(
        self: "iter[SupportsProdNoDefaultT]",
    ) -> typing.Union[T, typing.Literal[1]]:
        ...

    @typing.overload
    def prod(
        self: "iter[MultipliableT]", initial: MultipliableU
    ) -> typing.Union[MultipliableT, MultipliableU]:
        ...

    def prod(self, initial=_SENTINEL):
        """Return the product of all elements in this iterator.

        If initial is provided, it is used as the initial value.
        """
        if typing.TYPE_CHECKING:
            # HACK to make mypy happy with iterating this iter
            self = list(self)
        if initial is self._SENTINEL:
            return math.prod(self)
        # math.prod isn't actually guaranteed to run for non-numerics, so we
        # have to ignore the type error here.
        return math.prod(self, start=initial)  # type: ignore

    @typing.overload
    def sorted(
        self: "iter[SupportsRichComparisonT]",
        *,
        reverse: bool = False,
    ) -> "list[SupportsRichComparisonT]":
        ...

    @typing.overload
    def sorted(
        self,
        key: typing.Callable[[T], SupportsRichComparison],
        reverse: bool = False,
    ) -> "list[T]":
        ...

    def sorted(self, key=None, reverse=False):
        """Return a list containing the elements of this iterator sorted
        according to the given key and reverse parameters.
        """
        # HACK to make mypy happy with iterating this iter
        if typing.TYPE_CHECKING:
            self = list(self)
        return list(sorted(self, key=key, reverse=reverse))

    def reversed(self) -> "iter[T]":
        """Return an iterator containing the elements of this iterator in
        reverse order.
        """
        return iter(reversed(list(self)))

    @typing.overload
    def min(
        self: "iter[SupportsRichComparisonT]",
    ) -> T:
        ...

    @typing.overload
    def min(
        self,
        key: typing.Callable[[T], SupportsRichComparisonT],
    ) -> T:
        ...

    def min(self, key=None) -> T:
        """Return the minimum element of this iterator, according to the given
        key.
        """
        return min(self, key=key)  # type: ignore

    @typing.overload
    def max(
        self: "iter[SupportsRichComparisonT]",
    ) -> T:
        ...

    @typing.overload
    def max(
        self,
        key: typing.Callable[[T], SupportsRichComparisonT],
    ) -> T:
        ...

    def max(self, key=None) -> T:
        """Return the maximum element of this iterator, according to the given
        key.
        """
        return max(self, key=key)  # type: ignore

    def tee(self, n: int = 2) -> typing.Tuple["iter[T]", ...]:
        """Return a tuple of n iterators containing the elements of this
        iterator.
        """
        self.it, *iterators = itertools.tee(self, n + 1)
        return tuple(iter(iterator) for iterator in iterators)

    def permutations(
        self, r: typing.Union[int, None] = None
    ) -> "iter[typing.Tuple[T, ...]]":
        """Return an iterator over the permutations of the elements of this
        iterator.

        If r is provided, the returned iterator will only contain permutations
        of size r.
        """
        return iter(itertools.permutations(self, r))

    def combinations(self, r: int) -> "iter[typing.Tuple[T, ...]]":
        """Return an iterator over the combinations, without replacement, of
        length r of the elements of this iterator.
        """
        return iter(itertools.combinations(self, r))

    def combinations_with_replacement(self, r: int) -> "iter[typing.Tuple[T, ...]]":
        """Return an iterator over the combinations, with replacement, of
        length r of the elements of this iterator.
        """
        return iter(itertools.combinations_with_replacement(self, r))

    @typing.overload
    def flatten(self: "iter[typing.Iterable[T]]") -> "iter[T]":
        ...

    @typing.overload
    def flatten(
        self: "iter[typing.Iterable[T]]", recursive: typing.Literal[False] = False
    ) -> "iter[T]":
        ...

    @typing.overload
    def flatten(
        self: "iter[typing.Iterable[MaybeIterator[T]]]",
        recursive: typing.Literal[True] = True,
    ) -> "iter[T]":
        ...

    def flatten(self, recursive=False):
        """Flatten this iterator.

        If recursive is specified, flattens recursively instead
        of by one layer.
        """
        if not recursive:
            return iter(item for iterator in self for item in iterator)  # type: ignore
        return iter(
            item
            for iterator in self
            for item in (
                iterator.flatten(True)
                if isinstance(iterator, iter)
                else list(iterator).flat(True)
                if isinstance(iterator, (builtins.list, list))
                else iterator
            )  # type: ignore
        )

    def enumerate(self, start: int = 0) -> "iter[typing.Tuple[int, T]]":
        """Return an iterator over the elements of this iterator, paired with
        their index, starting at start.
        """
        return iter(enumerate(self, start))

    def count(self) -> int:
        """Consume this iterator and return the number of elements it contained."""
        return self.map(lambda _: 1).sum()

    def nlargest(self, n: int) -> list[T]:
        """Consume this iterator and return the n largest elements."""
        return list(nlargest(n, self))

    def nsmallest(self, n: int) -> list[T]:
        """Consume this iterator and return the n smallest elements."""
        return list(nsmallest(n, self))

    def __repr__(self) -> str:
        return f"iter({self.it!r})"


@functools.wraps(builtins.range)
def range(*args, **kw):
    return iter(builtins.range(*args, **kw))


@functools.wraps(builtins.map)
def map(*args, **kw):
    return iter(builtins.map(*args, **kw))


def irange(start: int, stop: int) -> iter[int]:
    """Inclusive range. Returns an iterator that
    yields values from start to stop, including both
    endpoints, stepping by one. Works even when
    stop < start (the iterator will step backwards).
    """
    if start <= stop:
        return range(start, stop + 1)
    else:
        return range(start, stop - 1, -1)


def _frange(
    start: float, stop: float, step: float
) -> typing.Generator[float, None, None]:
    if step == 0.0:
        raise ValueError("frange() arg 3 must not be zero")
    if step > 0:
        while start < stop:
            yield start
            start += step
    else:
        while start > stop:
            yield start
            start -= step


def frange(start: float, stop: float, step: float = 0.1) -> iter[float]:
    """Float range. Returns an iterator that yields values
    from start (inclusive) to stop (exclusive), changing by step.
    """
    return iter(_frange(start, stop, step))


class TailRecursionDetected(Exception):
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


def tail_call(func: typing.Callable[P, U]) -> typing.Callable[P, U]:
    """Add tail recursion optimisation to func.

    Useful for avoiding RecursionErrors.

    This is done by throwing an exception
    if the wrapper is its own grandparent (i.e. the wrapped
    function would be its own parent), and catching such
    exceptions to fake the tail call optimisation.

    func will behave strangely if the decorated
    function recurses in a non-tail context.
    """

    @functools.wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs):
        f = sys._getframe()
        if f.f_back and f.f_back.f_back and f.f_back.f_back.f_code == f.f_code:
            raise TailRecursionDetected(args, kwargs)
        else:
            while 1:
                try:
                    return func(*args, **kwargs)
                except TailRecursionDetected as e:
                    args = e.args  # type: ignore
                    kwargs = e.kwargs  # type: ignore
        raise Exception("unreachable")

    return wrapped


LetterRow = typing.Tuple[
    bool,
    bool,
    bool,
    bool,
    bool,
]
Letter = typing.Tuple[
    LetterRow,
    LetterRow,
    LetterRow,
    LetterRow,
    LetterRow,
    LetterRow,
]


def encode_letter(dots: Letter) -> int:
    """Encode a matrix of dots to an integer for efficient
    storage and lookup. Not expected to be used outside of
    this module and contributions to the lookup table.

    The matrix of dots should be 6 tall and 5 wide.
    """
    # Letters are 4 dots wide; the 5th column should always be empty.
    # This function assumes that input is not malformed; any dots in the
    # 5th column are treated as if they are in the 1st column of the next
    # row.
    # If something includes the 5th column it is malformed, but this
    # function will not check.
    out = 0
    for y, row in enumerate(dots):
        for x, dot in enumerate(row):
            if dot:
                out |= 1 << (x + 4 * y)
    return out


LETTERS: typing.Dict[int, str] = {
    # todo: fill in this lookup table
    0: " ",
    10090902: "A",
    7968663: "B",
    6885782: "C",
    15800095: "E",
    1120031: "F",
    15323542: "G",
    10067865: "H",
    14959694: "I",
    6916236: "J",
    9786201: "K",
    15798545: "L",
    6920598: "O",
    1145239: "P",
    9795991: "R",
    7889182: "S",
    6920601: "U",
    4475409: "Y",
    15803535: "Z",
}


def decode_letter(dots: Letter) -> str:
    """Decode a matrix of dots to a single letter.

    The matrix of dots should be 6 tall and 5 wide.
    """
    encoded = encode_letter(dots)
    if encoded not in LETTERS:
        print("Unrecognised letter:", encoded)
        for row in dots:
            for dot in row:
                print(" #"[dot], end="")
            print()
        print("Please consider contributing this to the lookup table:")
        print("https://github.com/starwort/aoc_helper")
        return "?"
    return LETTERS[encoded]


def decode_text(dots: typing.List[typing.List[bool]]) -> str:
    """Decode a matrix of dots to text.

    The matrix of dots should be 6 tall and 5n - 1 wide.
    """
    broken_rows = [list(chunk_default(row, 5, False)) for row in dots]
    letters = list(zip(*broken_rows))
    out = []
    for letter in letters:
        out.append(decode_letter(letter))
    if "?" in out:
        # prevent submitting malformed output
        raise ValueError("Unrecognised letter")
    return "".join(out)


def _default_classifier(char: str, /) -> int:
    if char in "0123456789":
        return int(char)
    elif char in ".#":
        return ".#".index(char)
    else:
        raise ValueError(f"Could not classify {char}. Please use a custom classifier.")


class Grid(typing.Generic[T]):
    data: list[list[T]]

    def __init__(self, data: list[list[T]]) -> None:
        self.data = data

    @typing.overload
    @classmethod
    def from_string(cls, data: str) -> "Grid[int]":
        ...

    @typing.overload
    @classmethod
    def from_string(cls, data: str, classify: typing.Callable[[str], T]) -> "Grid[T]":
        ...

    @classmethod
    def from_string(cls, data: str, classify=_default_classifier):
        """Create a grid from a string (e.g. a puzzle input).

        Can take a classifier to use a custom classification. The default will
        map numbers from 0 to 9 to themselves, and . and # to 0 and 1 respectively.
        """
        return Grid(list(data.splitlines()).mapped(lambda i: list(i).mapped(classify)))

    def dijkstras(
        self,
        start: typing.Optional[typing.Tuple[int, int]] = None,
        end: typing.Optional[typing.Tuple[int, int]] = None,
    ) -> int:
        """Use Dijkstra's algorithm to find the best path from
        start to end, and return the total cost.

        start defaults to the top left, and end defaults to the bottom right.
        """
        to_visit: typing.List[typing.Tuple[int, typing.Tuple[int, int]]] = []
        heappush(to_visit, (0, start or (0, 0)))
        visited = set()
        if end is None:
            target = len(self.data[0]) - 1, len(self.data) - 1
        else:
            target = end

        while True:
            cost, (x, y) = heappop(to_visit)
            if (x, y) in visited:
                continue
            if (x, y) == target:
                return cost
            visited.add((x, y))
            if x > 0:
                heappush(to_visit, (cost + self.data[y][x - 1], (x - 1, y)))
            if x < len(self.data[0]) - 1:
                heappush(to_visit, (cost + self.data[y][x + 1], (x + 1, y)))
            if y > 0:
                heappush(to_visit, (cost + self.data[y - 1][x], (x, y - 1)))
            if y < len(self.data) - 1:
                heappush(to_visit, (cost + self.data[y + 1][x], (x, y + 1)))

    def neighbours(self, x: int, y: int) -> list[T]:
        """Return the neighbours of a point in the grid (but not the point itself).

        Examples below:
        - A is the point (x, y)
        - * are points returned
        - . are other points in the grid

        ...........
        ..***......
        ..*A*......
        ..***......

        A*.........
        **.........
        ...........
        ...........
        """
        return (
            irange(max(y - 1, 0), min(y + 1, len(self.data) - 1))
            .map(
                lambda y_: irange(max(x - 1, 0), min(x + 1, len(self.data[0]) - 1))
                .filter(lambda x_: (x, y) != (x_, y_))
                .map(lambda x: self.data[y_][x])
            )
            .flatten(False)
        ).collect()

    def orthogonal_neighbours(self, x: int, y: int) -> list[T]:
        """Return the orthogonal neighbours of a point in the grid (but not the
        point itself).

        Examples below:
        - A is the point (x, y)
        - * are points returned
        - . are other points in the grid

        ...........
        ...*.......
        ..*A*......
        ...*.......

        A*.........
        *..........
        ...........
        ...........
        """
        rv = list()
        if x > 0:
            rv.append(self.data[y][x - 1])
        if x < len(self.data[0]) - 1:
            rv.append(self.data[y][x + 1])
        if y > 0:
            rv.append(self.data[y - 1][x])
        if y < len(self.data) - 1:
            rv.append(self.data[y + 1][x])
        return rv

    def deepcopy(self) -> "Grid[T]":
        return Grid(self.data.deepcopy())

    def __getitem__(self, index: int) -> list[T]:
        return self.data[index]


def dijkstras(
    grid: typing.List[typing.List[int]],
    start: typing.Optional[typing.Tuple[int, int]] = None,
    end: typing.Optional[typing.Tuple[int, int]] = None,
) -> int:
    """Use Dijkstra's algorithm to find the best path from
    start to end, and return the total cost.

    start defaults to the top left, and end defaults to the bottom right.
    """
    return Grid(list(grid).mapped(list)).dijkstras(start, end)


class PrioQueue(typing.Generic[T], typing.Iterator[T], typing.Iterable[T]):
    _data: builtins.list[T]

    def __init__(self, data: builtins.list[T]) -> None:
        self._data = data
        heapify(self._data)

    def __next__(self) -> T:
        if not self._data:
            raise StopIteration
        return heappop(self._data)

    def __iter__(self):
        return self

    def __bool__(self) -> bool:
        return bool(self._data)

    def next(self) -> T:
        return next(self)

    def push(self, val: T) -> None:
        heappush(self._data, val)
