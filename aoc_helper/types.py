from typing import Any, Protocol, TypeAlias, TypeVar, Union

# mostly stolen from typeshed

_T_contra = TypeVar("_T_contra", contravariant=True)
_T_co = TypeVar("_T_co", covariant=True)


class SupportsDunderLT(Protocol[_T_contra]):
    def __lt__(self, __other: _T_contra) -> bool:
        ...


class SupportsDunderGT(Protocol[_T_contra]):
    def __gt__(self, __other: _T_contra) -> bool:
        ...


class SupportsAdd(Protocol[_T_contra, _T_co]):
    def __add__(self, __x: _T_contra) -> _T_co:
        ...


class SupportsRAdd(Protocol[_T_contra, _T_co]):
    def __radd__(self, __x: _T_contra) -> _T_co:
        ...


class SupportsSub(Protocol[_T_contra, _T_co]):
    def __sub__(self, __x: _T_contra) -> _T_co:
        ...


class SupportsMul(Protocol[_T_contra, _T_co]):
    def __mul__(self, __x: _T_contra) -> _T_co:
        ...


class SupportsRMul(Protocol[_T_contra, _T_co]):
    def __rmul__(self, __x: _T_contra) -> _T_co:
        ...


class SupportsDiv(Protocol[_T_contra, _T_co]):
    def __div__(self, __x: _T_contra) -> _T_co:
        ...


class _SupportsSumWithNoDefaultGiven(
    SupportsAdd[Any, Any], SupportsRAdd[int, Any], Protocol
):
    ...


SupportsSumNoDefaultT = TypeVar(
    "SupportsSumNoDefaultT", bound=_SupportsSumWithNoDefaultGiven
)

AddableT = TypeVar("AddableT", bound=SupportsAdd[Any, Any])
AddableU = TypeVar("AddableU", bound=SupportsAdd[Any, Any])

SubtractableT = TypeVar("SubtractableT", bound=SupportsSub[Any, Any])


class _SupportsProdWithNoDefaultGiven(
    SupportsMul[Any, Any], SupportsRMul[int, Any], Protocol
):
    ...


SupportsProdNoDefaultT = TypeVar(
    "SupportsProdNoDefaultT", bound=_SupportsProdWithNoDefaultGiven
)

MultipliableT = TypeVar("MultipliableT", bound=SupportsMul[Any, Any])
MultipliableU = TypeVar("MultipliableU", bound=SupportsMul[Any, Any])


class _SupportsMean(_SupportsSumWithNoDefaultGiven, SupportsDiv[Any, Any], Protocol):
    ...


SupportsMean = TypeVar("SupportsMean", bound=_SupportsMean)

SupportsRichComparison: TypeAlias = Union[SupportsDunderLT[Any], SupportsDunderGT[Any]]
SupportsRichComparisonT = TypeVar(
    "SupportsRichComparisonT", bound=SupportsRichComparison
)  # noqa: Y001
