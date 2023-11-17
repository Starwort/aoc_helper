# `aoc_helper`

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

@salt-die's aoc_helper package, rewritten from the ground up

## Automation

This project aims to be compliant with the [Advent of Code Automation Guidelines](https://www.reddit.com/r/adventofcode/wiki/faqs/automation). Here are the strategies it uses:

- Once inputs are downloaded, they are cached in `~/.config/aoc_helper/YEAR/DAY.in` (or a similar path for Windows users) - [`interface.fetch`](https://github.com/Starwort/aoc_helper/blob/master/aoc_helper/interface.py#L97-L155) (lines [107-108](https://github.com/Starwort/aoc_helper/blob/master/aoc_helper/interface.py#L107-L108), [152](https://github.com/Starwort/aoc_helper/blob/master/aoc_helper/interface.py#L152))
- The `User-Agent` header declares the package name, version, and my contact info - [`data.HEADERS`](https://github.com/Starwort/aoc_helper/blob/master/aoc_helper/data.py#L20-L25), [used](https://github.com/Starwort/aoc_helper/blob/master/aoc_helper/interface.py#L118) [in](https://github.com/Starwort/aoc_helper/blob/master/aoc_helper/interface.py#L139) [every](https://github.com/Starwort/aoc_helper/blob/master/aoc_helper/interface.py#L210) [outbound](https://github.com/Starwort/aoc_helper/blob/master/aoc_helper/interface.py#L266) [request](https://github.com/Starwort/aoc_helper/blob/master/aoc_helper/interface.py#L341)
- If requesting input before the puzzle unlocks, [the library will wait for unlock before sending any requests](https://github.com/Starwort/aoc_helper/blob/master/aoc_helper/interface.py#L130-L133) (except on day 1, where it will send a request to validate the session token)
- If sending an answer too soon after an incorrect one, [the library will wait the cooldown specified in the response](https://github.com/Starwort/aoc_helper/blob/master/aoc_helper/interface.py#L231-L234) (sending only one extra request; it *is* however possible for a user to send multiple requests in quick succession, by repeatedly calling `submit` before the cooldown is over)
- Advent of Code will not be queried at all [if the puzzle has already been solved](https://github.com/Starwort/aoc_helper/blob/master/aoc_helper/interface.py#L182-L190) or [if an answer has already been submitted](https://github.com/Starwort/aoc_helper/blob/master/aoc_helper/interface.py#L193-L199)
- If, for some reason, the user decides they wish to clear their cache (for example, if they believe their input to be corrupted) they can do so by using the [`aoc clean`](https://github.com/Starwort/aoc_helper/blob/master/aoc_helper/main.py#L91-L121) command.

## Installation

Install `aoc_helper` with `pip`!

```bash
pip install aoc_helper
# install the dependencies required for the Command Line Interface
pip install aoc_helper[cli]
# install the dependencies required for colour
pip install aoc_helper[fancy]
# install all additional dependencies
pip install aoc_helper[cli,fancy]
# or
pip install aoc_helper[full]
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

`aoc_helper` has a command line interface, accessed by running `python -m aoc_helper` or `aoc` followed by the command line arguments. Its commands are detailed below:

### `fetch`

`aoc fetch <DAY> [--year <YEAR>]`

Fetch your input for a given day.  
YEAR is the current year by default.

Examples (written during 2020):

```bash
aoc fetch 2 # fetches input for 2020 day 2
aoc fetch 24 --year 2019 # fetches input for 2019 day 24
```

### `submit`

`aoc submit <DAY> <PART> <ANSWER> [--year <YEAR>]`

Submits your answer for a given day and part.  
YEAR is the current year by default.

Examples (written during 2020):

```bash
aoc submit 2 1 643 # submits 643 as the answer for 2020 day 2 part 1
aoc submit 24 1 12531574 --year 2019 # submits 12531574 as the answer for 2019 day 2 part 1
```

### `template`

`aoc template <DAYS> [--year YEAR]`

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
aoc template all # generates day_01.py to day_25.py, with aoc_helper methods referencing 2020, in the current folder
aoc template 3 --year 2019 # generates day_03.py, with aoc_helper methods referencing 2019, in the current folder
aoc template 3-5 --year 2017 # generates day_03.py to day_05.py, with aoc_helper methods referencing 2017, in the current folder
aoc template 3-5,7,9,9-10 # generates files for days 3, 4, 5, 7, 9, and 10
```
