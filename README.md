# `aoc_helper`

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

@salt-die's aoc_helper package, rewritten from the ground up

## Installation

Install `aoc_helper` with `pip`!

```bash
pip install aoc_helper
```

## Configuration

When you first use any function that interfaces with Advent of Code, you will be prompted to enter your session token.

Your session token is stored as a *HTTPOnly cookie*. This means there is no way of extracting it with JavaScript, you either must
use a browser extension such as [EditThisCookie](http://www.editthiscookie.com/), or follow [this guide](https://github.com/wimglenn/advent-of-code-wim/issues/1)

This token is stored in `~/.config/aoc_helper/token.txt` (`C:\Users\YOUR_USERNAME\.config\aoc_helper\token.txt` on Windows,
probably), and other `aoc_helper` data is stored in this folder (such as your input and submission caches).

If, for whatever reason, you feel the need to clear your caches, you can do so by deleting the relevant folders in `aoc_helper`'s
configuration folder.
