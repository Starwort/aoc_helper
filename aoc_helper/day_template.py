import aoc_helper
from aoc_helper import (
    Grid,
    PrioQueue,
    SparseGrid,
    decode_text,
    extract_ints,
    extract_iranges,
    extract_ranges,
    extract_uints,
    frange,
    irange,
    iter,
    list,
    map,
    range,
    tail_call,
)

raw = aoc_helper.fetch({day}, {year})


def parse_raw(raw):
    ...


data = parse_raw(raw)


def part_one(data):
    ...


aoc_helper.lazy_test(day={day}, year={year}, parse=parse_raw, solution=part_one)


def part_two(data):
    ...


aoc_helper.lazy_test(day={day}, year={year}, parse=parse_raw, solution=part_two)

aoc_helper.lazy_submit(day={day}, year={year}, solution=part_one, data=data)
aoc_helper.lazy_submit(day={day}, year={year}, solution=part_two, data=data)
