# `aoc_helper`

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

@salt-die's aoc_helper package, rewritten from the ground up

## Installation

Install `aoc_helper` with `pip`!

```bash
pip install aoc_helper
# install the dependencies required for the Command Line Interface
pip install aoc_helper[cli]
# install the dependencies required for colour
pip install aoc_helper[colour]
# install all additional dependencies
pip install aoc_helper[cli,colour]
```

## Configuration

When you first use any function that interfaces with Advent of Code, you will be prompted to enter your session token.

Your session token is stored as a *HTTPOnly cookie*. This means there is no way of extracting it with JavaScript, you either must
use a browser extension such as [EditThisCookie](http://www.editthiscookie.com/), or follow [this guide](https://github.com/wimglenn/advent-of-code-wim/issues/1)

This token is stored in `~/.config/aoc_helper/token.txt` (`C:\Users\YOUR_USERNAME\.config\aoc_helper\token.txt` on Windows,
probably), and other `aoc_helper` data is stored in this folder (such as your input and submission caches).

If, for whatever reason, you feel the need to clear your caches, you can do so by deleting the relevant folders in `aoc_helper`'s
configuration folder.

## Command Line Interface

`aoc_helper` has a command line interface, accessed by running `python -m aoc_helper` followed by the command line arguments. Its commands are detailed below:

### `fetch`

`python -m aoc_helper fetch <DAY> [--year <YEAR>]`

Fetch your input for a given day.  
YEAR is the current year by default.

Examples (written during 2020):

```bash
python -m aoc_helper fetch 2 # fetches input for 2020 day 2
python -m aoc_helper fetch 24 --year 2019 # fetches input for 2019 day 24
```

### `submit`

`python -m aoc_helper submit <DAY> <PART> <ANSWER> [--year <YEAR>]`

Submits your answer for a given day and part.  
YEAR is the current year by default.

Examples (written during 2020):

```bash
python -m aoc_helper submit 2 1 643 # submits 643 as the answer for 2020 day 2 part 1
python -m aoc_helper submit 24 1 12531574 --year 2019 # submits 12531574 as the answer for 2019 day 2 part 1
```

### `template`

`python -m aoc_helper template <DAYS> [--year YEAR]`

Generates templates for your advent of code folder.  
YEAR is the current year by default.

`DAYS` must be a comma-separated list of date ranges, which may be one of:

- `all`, to generate a template for every day in the year
- A single integer in the range \[1, 25] to generate a single template
- A pair of integers in the range \[1, 25], separated with a hyphen, to generate template for each day in the range

Note that even if a day is included in the list of days more than once (including implicitly, within ranges or `all`), it will only be generated once.

Filenames are formatted as `day_NUMBER.py` where `NUMBER` is the 2-digit day number.

Examples (written during 2020):

```bash
python -m aoc_helper template all # generates day_01.py to day_25.py, with aoc_helper methods referencing 2020, in the current folder
python -m aoc_helper template 3 --year 2019 # generates day_03.py, with aoc_helper methods referencing 2019, in the current folder
python -m aoc_helper template 3-5 --year 2017 # generates day_03.py to day_05.py, with aoc_helper methods referencing 2017, in the current folder
python -m aoc_helper template 3-5,7,9,9-10 # generates files for days 3, 4, 5, 7, 9, and 10
```
