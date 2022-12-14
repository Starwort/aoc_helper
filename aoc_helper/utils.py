import builtins
import collections
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
    SubtractableT,
    SupportsMean,
    SupportsProdNoDefaultT,
    SupportsRichComparison,
    SupportsRichComparisonT,
    SupportsSumNoDefaultT,
)

T = typing.TypeVar("T")
SpecialisationT = typing.TypeVar("SpecialisationT")
U = typing.TypeVar("U")
GenericU = typing.Generic[T]
P = ParamSpec("P")


MaybeIterator = typing.Union[T, typing.Iterable["MaybeIterator[T]"]]


def extract_ints(raw: str) -> "list[int]":
    """Utility function to extract all integers from some string.

    Some inputs can be directly parsed with this function.
    """
    return list(map(int, re.findall(r"((?:-|\+)?\d+)", raw)))


def extract_uints(raw: str) -> "list[int]":
    """Utility function to extract all integers from some string.

    Minus signs will be *ignored*; the output integers will all be positive.

    Some inputs can be directly parsed with this function.
    """
    return list(map(int, re.findall(r"(\d+)", raw)))


def _range_from_match(range: typing.Tuple[str, str]) -> builtins.range:
    if range[1]:
        return builtins.range(int(range[0]), int(range[1]))
    else:
        return builtins.range(int(range[0]), int(range[0]))


def _irange_from_match(range: typing.Tuple[str, str]) -> builtins.range:
    if range[1]:
        return builtins.range(int(range[0]), int(range[1]) + 1)
    else:
        return builtins.range(int(range[0]), int(range[0]) + 1)


def extract_ranges(raw: str) -> "list[builtins.range]":
    """Utility function to extract all ranges from some string.

    Ranges are interpreted as `start-stop` and are not inclusive.

    Some inputs can be directly parsed with this function.
    """
    return list(map(_range_from_match, re.findall(r"(\d+)(?:-(\d+))?", raw)))


def extract_iranges(raw: str) -> "list[builtins.range]":
    """Utility function to extract all ranges from some string.

    Ranges are interpreted as `start-stop` and are inclusive.

    Some inputs can be directly parsed with this function.
    """
    return list(map(_irange_from_match, re.findall(r"(\d+)(?:-(\d+))?", raw)))


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
        self: "list[typing.Iterable[SpecialisationT]]",
        func: typing.Callable[[SpecialisationT], U],
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

    def any(self, pred: typing.Union[typing.Callable[[T], bool], T] = bool) -> bool:
        """Consume this iterator and return True if any element satisfies the
        given predicate. The default predicate is bool; therefore by default this
        method returns True if any element is truthy.
        """
        if not callable(pred):
            pred = (lambda j: lambda i: i == j)(pred)
        return any(pred(item) for item in self)

    def all(self, pred: typing.Union[typing.Callable[[T], bool], T] = bool) -> bool:
        """Consume this iterator and return True if all elements satisfy the
        given predicate. The default predicate is bool; therefore by default this
        method returns True if all elements are truthy.
        """
        if not callable(pred):
            pred = (lambda j: lambda i: i == j)(pred)
        return all(pred(item) for item in self)

    def none(self, pred: typing.Union[typing.Callable[[T], bool], T] = bool) -> bool:
        """Consume this iterator and return True if no element satisfies the
        given predicate. The default predicate is bool; therefore by default this
        method returns True if no element is truthy.
        """
        if not callable(pred):
            pred = (lambda j: lambda i: i == j)(pred)
        return not any(pred(item) for item in self)

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
        self: "list[SupportsRichComparisonT]",
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
        self: "list[SupportsRichComparisonT]",
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
    def flat(self: "list[typing.Iterable[SpecialisationT]]") -> "list[SpecialisationT]":
        ...

    @typing.overload
    def flat(
        self: "list[typing.Iterable[SpecialisationT]]",
        recursive: typing.Literal[False] = False,
    ) -> "list[SpecialisationT]":
        ...

    @typing.overload
    def flat(
        self: "list[typing.Iterable[MaybeIterator[SpecialisationT]]]",
        recursive: typing.Literal[True] = True,
    ) -> "list[SpecialisationT]":
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

    def transposition(
        self: "list[list[SpecialisationT]]",
    ) -> "list[list[SpecialisationT]]":
        """Return the transposition of this list, which is assumed to be
        rectangular, not ragged.

        This operation looks similar to a 90° rotation followed by a reflection:

        ABC
        DEF
        HIJ
        KLM

        transposes to:

        ADHK
        BEIL
        CFJM
        """
        return list(zip(*self)).mapped(list)

    def into_grid(self: "list[list[SpecialisationT]]") -> "Grid[SpecialisationT]":
        """Convert this list, which is assumed to be rectangular, not ragged,
        into a Grid.

        This function converts directly; it doesn't copy - expect strange
        behaviour if you continue using self.
        """
        return Grid(self)

    def into_queue(self) -> "PrioQueue[T]":
        """Convert this list into a PrioQueue.

        This function converts directly; it doesn't copy - expect strange
        behaviour if you continue using self.
        """
        return PrioQueue(self.into_builtin())

    def into_builtin(self) -> typing.List[T]:
        """Unwrap this list into a builtins.list.

        This function converts directly; it doesn't copy - expect strange
        behaviour if you continue using self.
        """
        return self.data

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
        return next(self.it)

    def map(self, func: typing.Callable[[T], U]) -> "iter[U]":
        """Return an iterator containing the result of calling func on each
        element in this iterator.
        """
        return iter(map(func, self))

    def map_each(
        self: "iter[typing.Iterable[SpecialisationT]]",
        func: typing.Callable[[SpecialisationT], U],
    ) -> "iter[iter[U]]":
        """Return an iterator containing the result of calling func on each
        element in each element in this iterator.
        """
        return iter(self.map(lambda i: iter(i).map(func)))

    def filter(
        self, pred: typing.Union[typing.Callable[[T], bool], T] = bool
    ) -> "iter[T]":
        """Return an iterator containing only the elements for which pred
        returns True.

        If pred is a T (and T is not callable), return an iterator
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

    def any(self, pred: typing.Union[typing.Callable[[T], bool], T] = bool) -> bool:
        """Consume this iterator and return True if any element satisfies the
        given predicate. The default predicate is bool; therefore by default this
        method returns True if any element is truthy.
        """
        if not callable(pred):
            pred = (lambda j: lambda i: i == j)(pred)
        return any(pred(item) for item in self)

    def all(self, pred: typing.Union[typing.Callable[[T], bool], T] = bool) -> bool:
        """Consume this iterator and return True if all elements satisfy the
        given predicate. The default predicate is bool; therefore by default this
        method returns True if all elements are truthy.
        """
        if not callable(pred):
            pred = (lambda j: lambda i: i == j)(pred)
        return all(pred(item) for item in self)

    def none(self, pred: typing.Union[typing.Callable[[T], bool], T] = bool) -> bool:
        """Consume this iterator and return True if no element satisfies the
        given predicate. The default predicate is bool; therefore by default this
        method returns True if no element is truthy.
        """
        if not callable(pred):
            pred = (lambda j: lambda i: i == j)(pred)
        return not any(pred(item) for item in self)

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
    def collect(self, collection_type: typing.Type[U]) -> "U[T]":  # type: ignore
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
    def flatten(
        self: "iter[typing.Iterable[SpecialisationT]]",
    ) -> "iter[SpecialisationT]":
        ...

    @typing.overload
    def flatten(
        self: "iter[typing.Iterable[SpecialisationT]]",
        recursive: typing.Literal[False] = False,
    ) -> "iter[SpecialisationT]":
        ...

    @typing.overload
    def flatten(
        self: "iter[typing.Iterable[MaybeIterator[SpecialisationT]]]",
        recursive: typing.Literal[True] = True,
    ) -> "iter[SpecialisationT]":
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
    # 4475409: "Y",
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
    out = "".join(decode_letter(letter) for letter in letters)
    assert "?" not in out, f"Output {out} contained unrecognised letters!"
    return out


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

    @classmethod
    def from_string(
        cls, data: str, classify: typing.Callable[[str], U] = _default_classifier
    ) -> "Grid[U]":
        """Create a grid from a string (e.g. a puzzle input).

        Can take a classifier to use a custom classification. The default will
        map numbers from 0 to 9 to themselves, and . and # to 0 and 1 respectively.
        """
        return Grid(list(data.splitlines()).mapped(lambda i: list(i).mapped(classify)))

    def vertical_chunks(self, n: int) -> list["Grid[T]"]:
        """Create a list of grids formed by splitting this grid every n rows.

        Any extra rows that cannot form a group of n will be lost (see
        vertical_chunks_default)
        """
        chunked_rows = self.data.chunked(n)
        return chunked_rows.mapped(list).mapped(Grid)

    def vertical_chunks_default(self, n: int, fill_value: T) -> list["Grid[T]"]:
        """Create a list of grids formed by splitting this grid every n rows.

        Grids will be padded out to have n rows, where every cell in the padded
        rows is fill_value.
        """
        if self.data.len() == 0:
            return list()
        fill_row = list(fill_value for _ in self.data[0])
        chunked_rows = self.data.chunked_default(n, fill_row)
        result = chunked_rows.mapped(list)
        result[-1] = result[-1].mapped(lambda i: i.deepcopy() if i is fill_row else i)
        return result.mapped(Grid)

    def horizontal_chunks(self, n: int) -> list["Grid[T]"]:
        """Create a list of grids formed by splitting this grid every n columns.

        Any extra columns that cannot form a group of n will be lost (see
        horizontal_chunks_default)
        """
        chunked_data = [list(chunk(row, n)) for row in self.data]
        return list(zip(*chunked_data)).mapped(list).mapped(Grid)

    def horizontal_chunks_default(self, n: int, fill_value: T) -> list["Grid[T]"]:
        """Create a list of grids formed by splitting this grid every n columns.

        Rows will be padded out to have n values, where every cell in the padded
        columns is fill_value.
        """
        chunked_data = [list(chunk_default(row, n, fill_value)) for row in self.data]
        return list(zip(*chunked_data)).mapped(list).mapped(Grid)

    def transpose(self) -> "Grid[T]":
        """Create a grid that is the transposition of this grid.

        This operation looks similar to a 90° rotation followed by a reflection:

        ABC
        DEF
        HIJ
        KLM

        transposes to:

        ADHK
        BEIL
        CFJM
        """
        return Grid(self.data.transposition())

    def rotate_clockwise(self) -> "Grid[T]":
        """Create a new grid that is the clockwise rotation of this grid.

        self[0][0] is considered to be the top-left corner.
        """
        return Grid(self.data[::-1].transposition())

    def rotate_anticlockwise(self) -> "Grid[T]":
        """Create a new grid that is the anti-clockwise rotation of this grid.

        self[0][0] is considered to be the top-left corner.
        """
        return Grid(self.data.transposition()[::-1])

    def to_bool_grid(self, convert: typing.Callable[[T], bool] = bool) -> "Grid[bool]":
        """Create a new grid of booleans by using the given conversion function
        on self. The default conversion function is bool, converting via
        truthiness value.
        """
        # Would love to replace this with a mapped_each call but it doesn't type-check
        return Grid(self.data.mapped(lambda i: i.mapped(convert)))

    def decode_as_text(self: "Grid[bool]") -> str:
        """Decode self as a grid of letters using decode_text.

        This method will check that self is the correct dimensions and raise an
        AssertionError if not.
        """
        self = self.trim_to_content()
        assert (
            len(self.data) == 6
        ), f"Expected a height of 6, found height of {len(self.data)}"
        assert len(self.data[0]) % 5 == 4, (
            f"Expected a width of 5n + 4, found width of {len(self.data[0])} (5n +"
            f" {len(self.data[0]) % 5})"
        )
        return decode_text([[i for i in row] for row in self.data])

    def trim_to_content(self, keep: typing.Callable[[T], bool] = bool) -> "Grid[T]":
        """Create a new grid of booleans by using the given conversion function
        on self. The default conversion function is bool, converting via
        truthiness value."""
        trim_rows = self.data.mapped(lambda i: i.none(keep))
        if trim_rows.all():  # Trim out the entire grid
            return Grid(list())
        trim_cols = self.transpose().data.mapped(lambda i: i.none(keep))
        if trim_rows.none() and trim_cols.none():  # Nothing to trim
            return self.deepcopy()
        trim_cols = trim_cols.enumerated()
        trim_rows = trim_rows.enumerated()
        top = expect(trim_rows.find(lambda i: not i[1]))[0]
        bottom = expect(trim_rows[::-1].find(lambda i: not i[1]))[0]
        left = expect(trim_cols.find(lambda i: not i[1]))[0]
        right = expect(trim_cols[::-1].find(lambda i: not i[1]))[0]
        return Grid(self.data[top : bottom + 1].mapped(lambda i: i[left : right + 1]))

    def neighbours(
        self, x: int, y: int
    ) -> list[typing.Tuple[typing.Tuple[int, int], T]]:
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
                .map(lambda x: ((x, y_), self.data[y_][x]))
            )
            .flatten(False)
        ).collect()

    def orthogonal_neighbours(
        self, x: int, y: int
    ) -> list[typing.Tuple[typing.Tuple[int, int], T]]:
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
            rv.append(((x - 1, y), self.data[y][x - 1]))
        if x < len(self.data[0]) - 1:
            rv.append(((x + 1, y), self.data[y][x + 1]))
        if y > 0:
            rv.append(((x, y - 1), self.data[y - 1][x]))
        if y < len(self.data) - 1:
            rv.append(((x, y + 1), self.data[y + 1][x]))
        return rv

    def pathfind(
        self: "Grid[AddableT]",
        start=None,
        end=None,
        valid_traversal: typing.Callable[
            [AddableT, AddableT], bool
        ] = lambda i, j: True,
        cost_function: typing.Callable[[AddableT, AddableT], AddableT] = (
            lambda i, j: j - i  # type: ignore
        ),
        neighbour_type: typing.Literal["ortho", "full"] = "ortho",
        initial_cost: AddableT = 0,
    ) -> typing.Optional[AddableT]:
        """Use Dijkstra's algorithm to find the best path from start to end, and
        return the total cost.

        start defaults to the top left, and end defaults to the bottom right.

        valid_traversal is a function that takes the start value and the end
        value of a potential traversal, and returns whether that traversal is
        valid. The default is that any traversal is valid.

        cost_function is a function that takes the start value and the end value
        of a traversal, and returns the cost of that traversal. The default is
        that the cost is the difference between the two values.

        initial_cost is the zero-value of the cell type. You should only need to
        modify this if you're using a non-numeric type.
        """
        to_visit = PrioQueue([(typing.cast(AddableT, initial_cost), start or (0, 0))])
        visited = set()
        if end is None:
            target = len(self.data[0]) - 1, len(self.data) - 1
        else:
            target = end

        neighbours = (
            self.orthogonal_neighbours if neighbour_type == "ortho" else self.neighbours
        )

        for cost, (x, y) in to_visit:
            if (x, y) in visited:
                continue
            if (x, y) == target:
                return cost
            visited.add((x, y))
            for neighbour, value in neighbours(x, y):
                if valid_traversal(self.data[y][x], value):
                    to_visit.push(
                        (cost + cost_function(self.data[y][x], value), neighbour)
                    )

    dijkstras = pathfind

    def deepcopy(self) -> "Grid[T]":
        return Grid(self.data.deepcopy())

    def __getitem__(self, index: int) -> list[T]:
        return self.data[index]

    def __repr_row(self, row: list[T]) -> str:
        # Specialise output for empty, bool, and int
        if row.len() == 0:
            return "    [],\n"
        elif narrow_list(row, bool):
            return "    " + "".join(row.mapped("_█".__getitem__)) + "\n"
        elif narrow_list(row, int):
            if not hasattr(self, "_cached_int_width"):
                self._cached_int_width = (
                    typing.cast(Grid[int], self)
                    .data.mapped(lambda i: i.mapped(str).mapped(len).max())
                    .max()
                )
            return (
                "    "
                + " ".join(f"{{: >{self._cached_int_width}}}".format(i) for i in row)
                + "\n"
            )
        else:
            return "    " + repr(row.data) + ",\n"

    def __repr__(self) -> str:
        if self.data.len() == 0:
            return "Grid([])"
        else:
            out = "Grid([\n"
            for row in self.data:
                out += self.__repr_row(row)
            return out + "])"


def points_between(
    start: typing.Tuple[int, int], end: typing.Tuple[int, int]
) -> iter[typing.Tuple[int, int]]:
    """Return an iterator of points between start and end, inclusive.
    Start to end must be horizontal, vertical, or a perfect diagonal.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    assert abs(dx) == abs(dy) or dx == 0 or dy == 0
    if dx == 0:
        return iter(zip(itertools.repeat(start[0]), irange(start[1], end[1])))
    elif dy == 0:
        return iter(zip(irange(start[0], end[0]), itertools.repeat(start[1])))
    else:
        return iter(zip(irange(start[0], end[0]), irange(start[1], end[1])))


class SparseGrid(typing.Generic[T]):
    data: typing.DefaultDict[typing.Tuple[int, int], T]

    def __init__(self, default_factory: typing.Callable[[], T]) -> None:
        self.data = collections.defaultdict(default_factory)

    def draw_line(
        self,
        start: typing.Tuple[int, int],
        end: typing.Tuple[int, int],
        value: T,
    ) -> None:
        """Draw a line on a sparse grid, setting all points between start and end
        to value.
        """
        for x, y in points_between(start, end):
            self[x, y] = value

    def draw_lines(
        self,
        lines: typing.Iterable[typing.Tuple[int, int]],
        value: T,
    ) -> None:
        """Draw a series of lines on a sparse grid, setting all points between
        each pair of points to value.
        """
        _lines: list[typing.Tuple[int, int]] = list(lines)
        if _lines:
            x, y = _lines[0]  # allows for lists to be used instead of tuples
            self[x, y] = value
        for start, end in _lines.windowed(2):
            self.draw_line(start, end, value)

    def bounds(self, empty: builtins.list[T]) -> typing.Tuple[int, int, int, int]:
        """Return the bounds of a sparse grid, as a tuple of (min_x, min_y, max_x, max_y)."""
        if len(self) == 0:
            return 0, 0, 0, 0
        else:
            return (
                min(x for x, _ in filter(lambda i: self[i] not in empty, self)),
                min(y for _, y in filter(lambda i: self[i] not in empty, self)),
                max(x for x, _ in filter(lambda i: self[i] not in empty, self)),
                max(y for _, y in filter(lambda i: self[i] not in empty, self)),
            )

    def pretty_print(
        self, to_char: typing.Callable[[T], str], empty: builtins.list[T]
    ) -> None:
        """Print a sparse grid to the console."""
        min_x, min_y, max_x, max_y = self.bounds(empty)
        max_y_width = max(len(str(max_y)), len(str(min_y)))
        max_x_width = max(len(str(max_x)), len(str(min_x)))
        x_labels = [
            f"{x:={max_x_width}}" if x % 2 == 0 else (" " * max_x_width)
            for x in irange(min_x, max_x)
        ]
        for char in range(max_x_width):
            print(" " * max_y_width, end=" ")
            for label in x_labels:
                print(label[char], end="")
            print()
        for y in irange(min_y, max_y):
            print(f"{y:={max_y_width}}", end=" ")
            for x in irange(min_x, max_x):
                print(to_char(self[x, y]), end="")
            print()

    def __getitem__(self, index: typing.Tuple[int, int]) -> T:
        return self.data[index]

    def __setitem__(self, index: typing.Tuple[int, int], value: T) -> None:
        self.data[index] = value

    def __delitem__(self, index: typing.Tuple[int, int]) -> None:
        del self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> iter[typing.Tuple[int, int]]:
        return iter(self.data)

    def __repr__(self) -> str:
        return f"SparseGrid({self.data})"

    def keys(self) -> iter[typing.Tuple[int, int]]:
        return iter(self.data.keys())

    def values(self) -> iter[T]:
        return iter(self.data.values())

    def items(self) -> iter[typing.Tuple[typing.Tuple[int, int], T]]:
        return iter(self.data.items())


def expect(val: typing.Optional[T]) -> T:
    """Expect that a value is not None."""
    assert val is not None
    return val


def narrow_list(list: list, type: typing.Type[T]) -> typing.TypeGuard[list[T]]:
    """Narrow the type of list based on the passed type.

    Assumes that list is homogenous.
    """
    return isinstance(list[0], type)


def pathfind(
    grid: typing.List[typing.List[int]],
    start: typing.Optional[typing.Tuple[int, int]] = None,
    end: typing.Optional[typing.Tuple[int, int]] = None,
) -> typing.Optional[int]:
    """Use Dijkstra's algorithm to find the best path from start to end, and
    return the total cost.

    start defaults to the top left, and end defaults to the bottom right.
    """
    return Grid(list(grid).mapped(list)).pathfind(start, end)


dijkstras = pathfind


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

    def __repr__(self) -> str:
        return f"PrioQueue({self._data})"


def chinese_remainder_theorem(
    moduli: typing.List[int], residues: typing.List[int]
) -> int:
    from math import prod

    N = prod(moduli)

    return (
        sum(
            (div := (N // modulus)) * pow(div, -1, modulus) * residue
            for modulus, residue in zip(moduli, residues)
        )
        % N
    )
