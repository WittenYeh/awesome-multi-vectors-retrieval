#!/usr/bin/env python3
"""Regenerate the README banner with pyfiglet."""

from pathlib import Path

from pyfiglet import figlet_format


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START = "<!-- ascii-art:start -->"
END = "<!-- ascii-art:end -->"


def main() -> None:
    readme = README.read_text(encoding="utf-8")
    before, rest = readme.split(START, maxsplit=1)
    _, after = rest.split(END, maxsplit=1)
    art = "\n".join(
        line.rstrip()
        for line in figlet_format("Awesome MVR", font="standard").rstrip().splitlines()
    )
    block = f"{START}\n```text\n{art}\n```\n{END}"
    README.write_text(f"{before}{block}{after}", encoding="utf-8")


if __name__ == "__main__":
    main()
