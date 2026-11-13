"""User-study evaluation kit (FYP2): the Fitts'-law task, its analysis and SUS scoring.

    evaluation.task_runner      ISO 9241-9 style target-acquisition task
    evaluation.analyze_results  effective width, throughput, MT = a + b*ID
    evaluation.sus_score        System Usability Scale scoring

``read_table`` lives here because both command-line tools read participant data
that may have passed through Excel on its way here.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import NamedTuple


class Table(NamedTuple):
    columns: list[str]                       # lower-case, no surrounding spaces
    rows: list[tuple[int, dict[str, str]]]   # (line number in the file, {column: text})


def read_table(path: str | Path) -> Table:
    """Read a CSV the way Excel may have saved it.

    Copes with a UTF-8 byte-order mark, GBK text (plain "CSV" on a Chinese
    Windows), semicolon or tab separators, and rows of empty cells. A short row
    simply lacks the missing columns. Raises OSError if the file can't be read
    and ValueError if it isn't a table at all.
    """
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("gbk", errors="replace")
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    delimiter = max(",;\t", key=first_line.count)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    columns: list[str] = []
    rows: list[tuple[int, dict[str, str]]] = []
    try:
        for cells in reader:
            cells = [cell.strip() for cell in cells]
            if not any(cells):
                continue
            if not columns:
                columns = [cell.lower() for cell in cells]
            else:
                rows.append((reader.line_num, dict(zip(columns, cells))))
    except csv.Error as e:
        raise ValueError(str(e)) from e
    return Table(columns, rows)
