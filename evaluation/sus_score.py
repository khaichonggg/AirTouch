"""System Usability Scale scoring (Brooke 1996) with the adjective scale of
Bangor, Kortum & Miller (2009).

Each of the 10 statements is rated 1 (strongly disagree) to 5 (strongly agree).
Odd statements are worded positively and contribute (rating - 1); even ones are
worded negatively and contribute (5 - rating). The sum x 2.5 gives 0..100. The
result is a score, not a percentage.

Run:  python -m evaluation.sus_score responses.csv
      (one row per participant: columns q1..q10, plus an optional "participant")
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from airtouch.console import use_utf8_console
from evaluation import Table, read_table

NUM_ITEMS = 10
ITEM_COLUMNS = tuple(f"q{i}" for i in range(1, NUM_ITEMS + 1))

# (lowest score, adjective, published mean). Bangor et al. (2009) report the mean
# SUS of the systems that received each adjective, not boundaries between them,
# so each lower bound is the midpoint between two neighbouring means: a score
# takes the adjective whose published mean it is closest to.
ADJECTIVE_SCALE = (
    (0.0, "Worst imaginable", 12.5),
    (16.4, "Awful", 20.3),
    (28.0, "Poor", 35.7),
    (43.3, "OK", 50.9),
    (61.15, "Good", 71.4),
    (78.45, "Excellent", 85.5),
    (88.2, "Best imaginable", 90.9),
)
# Bangor et al.'s school-style grades.
GRADE_SCALE = ((90.0, "A"), (80.0, "B"), (70.0, "C"), (60.0, "D"), (0.0, "F"))
# Bangor, Kortum & Miller (2008): below 50 not acceptable, above 70 acceptable.
NOT_ACCEPTABLE_BELOW = 50.0
ACCEPTABLE_ABOVE = 70.0


class SusError(ValueError):
    """Bad input. The message is plain enough to show to the person as it is."""


def _rating(value) -> int | None:
    """``value`` as a whole number 1..5, or None. Excel may save 3 as "3" or "3.0"."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not number.is_integer() or not 1 <= number <= 5:     # NaN and inf fail is_integer
        return None
    return int(number)


def sus_score(responses) -> float:
    """Score one person's 10 answers (q1 first). Raises SusError on any bad answer."""
    if isinstance(responses, (str, bytes)):
        raise SusError("答案要一个一个分开给，不能是一整串文字。")
    try:
        answers = list(responses)
    except TypeError:
        raise SusError("答案要是 10 个数字。") from None
    if len(answers) != NUM_ITEMS:
        raise SusError(f"需要正好 {NUM_ITEMS} 个答案，现在有 {len(answers)} 个。")
    ratings = []
    for i, value in enumerate(answers, start=1):
        rating = _rating(value)
        if rating is None:
            raise SusError(f"第 {i} 题的答案 {value!r} 不对，只能是 1 到 5 的整数。")
        ratings.append(rating)
    total = sum(r - 1 if i % 2 == 1 else 5 - r for i, r in enumerate(ratings, start=1))
    return total * 2.5


def _check_score(score: float) -> float:
    number = float(score)
    if not math.isfinite(number) or not 0 <= number <= 100:
        raise SusError(f"SUS 分数必须在 0 到 100 之间，现在是 {score!r}。")
    return number


def adjective_rating(score: float) -> str:
    """Adjective for a score (individual or mean), see ``ADJECTIVE_SCALE``."""
    number = _check_score(score)
    return next(adj for low, adj, _mean in reversed(ADJECTIVE_SCALE) if number >= low)


def grade(score: float) -> str:
    number = _check_score(score)
    return next(letter for low, letter in GRADE_SCALE if number >= low)


def acceptability(score: float) -> str:
    number = _check_score(score)
    if number < NOT_ACCEPTABLE_BELOW:
        return "Not acceptable"
    return "Acceptable" if number > ACCEPTABLE_ABOVE else "Marginal"


@dataclass(frozen=True)
class SusResult:
    participant: str
    score: float

    @property
    def adjective(self) -> str:
        return adjective_rating(self.score)

    @property
    def grade(self) -> str:
        return grade(self.score)


def score_table(table: Table) -> list[SusResult]:
    """Score every row. Lists ALL problems at once, and scores nothing if there
    are any: one mistyped answer must not quietly change the reported mean."""
    missing = [c for c in ITEM_COLUMNS if c not in table.columns]
    if missing:
        raise SusError(
            "表格缺少这些列：" + ", ".join(missing) + "。\n"
            "第一行应该是 participant,q1,q2,...,q10（participant 这一列可以没有）。"
        )
    if not table.rows:
        raise SusError("表格里只有标题，没有任何人的答案。")
    problems, results, seen = [], [], set()
    for row_number, (line, row) in enumerate(table.rows, start=1):
        who = row.get("participant") or f"P{row_number}"
        where = f"第 {line} 行（{who}）"
        if who in seen:
            problems.append(f"{where}：这个编号前面已经出现过了。")
        seen.add(who)
        ok = True
        for i, column in enumerate(ITEM_COLUMNS, start=1):
            text = row.get(column, "")
            if not text:
                problems.append(f"{where}：第 {i} 题没有填。")
                ok = False
            elif _rating(text) is None:
                problems.append(f"{where}：第 {i} 题的答案“{text}”不对，只能是 1 到 5 的整数。")
                ok = False
        if ok:
            results.append(SusResult(who, sus_score(row[c] for c in ITEM_COLUMNS)))
    if problems:
        raise SusError("表格里有这些问题，改好后再算一次：\n  " + "\n  ".join(problems))
    return results


def score_csv(path: str | Path) -> list[SusResult]:
    path = Path(path)
    if not path.is_file():
        raise SusError(f"找不到这个文件：{path}")
    try:
        table = read_table(path)
    except (OSError, ValueError) as e:
        raise SusError(f"打不开这个文件：{path}（{e}）") from e
    return score_table(table)


def format_report(results: list[SusResult]) -> str:
    """Table for the report (standard English terms) with a plain-language ending."""
    scores = [r.score for r in results]
    mean = statistics.fmean(scores)
    width = max(len("Participant"), *(len(r.participant) for r in results))
    head = f"{'Participant':<{width}}  {'SUS':>5}  {'Adjective':<16}  Grade"
    lines = [head, "-" * len(head)]
    lines += [f"{r.participant:<{width}}  {r.score:5.1f}  {r.adjective:<16}  {r.grade}" for r in results]
    lines.append("-" * len(head))
    lines.append(f"{'Mean':<{width}}  {mean:5.1f}  {adjective_rating(mean):<16}  {grade(mean)}")
    if len(scores) > 1:
        lines.append(f"{'SD':<{width}}  {statistics.stdev(scores):5.1f}")
    else:
        lines.append(f"{'SD':<{width}}    n/a  (needs 2 or more people)")
    lines.append(f"{'n':<{width}}  {len(scores):5d}")
    lines.append(f"Acceptability of the mean: {acceptability(mean)}")
    lines.append("")
    lines.append(f"一共 {len(scores)} 人，平均 {mean:.1f} 分（满分 100，这不是百分比）。")
    lines.append("大致标准：70 分以上算可以接受，50 分以下算不能接受，中间是勉强。")
    return "\n".join(lines)


def main(argv=None) -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser(description="计算 SUS 问卷分数（每人一行，q1 到 q10 填 1 到 5）。")
    ap.add_argument("csv", help="问卷结果表格（.csv 文件）")
    args = ap.parse_args(argv)
    try:
        results = score_csv(args.csv)
    except SusError as e:
        print(e, file=sys.stderr)
        return 2
    print(format_report(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
