"""Analyse the session logs written by ``evaluation.task_runner``.

For each input condition this gives movement time, error rate and the ISO 9241-9
measures, as described by Soukoreff & MacKenzie (2004):

    dx   the click's signed offset from the target centre, projected on the task
         axis (start -> target); positive = overshoot, sideways error is ignored
    We   = 4.133 x SD(dx)            effective width: what the person really hit
    Ae   = mean(|start->target| + dx) effective amplitude: distance really moved
    IDe  = log2(Ae / We + 1)         effective difficulty (bits)
    TP   = IDe / MT                  throughput (bits/s): speed and accuracy in one

These are computed per sequence of trials (one person, one A x W, one block) and
then averaged - sequences -> A x W conditions -> people - which is MacKenzie's
"mean of means". The start of a trial is the previous click, so the serial-task
adjustment to Ae (adding the previous trial's dx) is already built in.

Every trial counts, misses included: We is what makes throughput fair to a fast
but sloppy participant. ``--exclude-outliers`` drops accidental clicks using the
rule in Wobbrock, Shinohara & Jansen (2011); the count is always printed.

Run:  python -m evaluation.analyze_results [folder-or-csv ...] [--detail] [--plot [PNG]]
"""

from __future__ import annotations

import argparse
import io
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from airtouch.console import use_utf8_console
from evaluation import read_table

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REQUIRED_COLUMNS = (
    "participant", "input", "block", "amplitude", "width",
    "target_x", "target_y", "start_x", "start_y", "click_x", "click_y",
    "hit", "movement_time_ms",
)
EFFECTIVE_WIDTH_FACTOR = 4.133      # +-2.066 SD: 96% of a normal spread, i.e. 4% errors
MIN_SEQUENCE_TRIALS = 3             # an aborted sequence this short has no usable spread
INPUT_ORDER = ("gesture", "mouse")  # fixed, so tables and colours never swap between runs

# Plot palette (dataviz reference palette, light mode). Each input keeps its colour.
SERIES_COLORS = {"gesture": "#2a78d6", "mouse": "#eb6834"}
OTHER_SERIES_COLOR = "#898781"
SURFACE, INK, INK_2, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def fitts_id(amplitude: float, width: float) -> float:
    """Shannon formulation of Fitts' index of difficulty, in bits."""
    return math.log2(amplitude / width + 1)


# ------------------------------------------------------------------ loading
@dataclass(frozen=True)
class Trial:
    participant: str
    input: str
    block: int
    amplitude: float
    width: float
    target: tuple[float, float]
    start: tuple[float, float]
    click: tuple[float, float]
    hit: bool
    mt_ms: float
    source: str = ""        # file name, so two sessions never merge into one sequence

    @property
    def axis_length(self) -> float:
        return math.dist(self.start, self.target)

    @property
    def deviation(self) -> float:
        """dx: signed offset of the click from the target along the task axis."""
        ax, ay = self.target[0] - self.start[0], self.target[1] - self.start[1]
        ox, oy = self.click[0] - self.target[0], self.click[1] - self.target[1]
        return (ox * ax + oy * ay) / self.axis_length

    @property
    def is_spatial_outlier(self) -> bool:
        """A miss that barely moved (under A/2) or landed far away (over 2W from
        the target centre): an accidental click, not an attempt at the target."""
        return not self.hit and (math.dist(self.start, self.click) < self.amplitude / 2
                                 or math.dist(self.click, self.target) > 2 * self.width)


def _number(row: dict, column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value):
        raise ValueError(column)
    return value


def _parse_hit(text: str) -> bool:
    value = text.strip().lower()
    if value in ("1", "true", "yes"):
        return True
    if value in ("0", "false", "no"):
        return False
    raise ValueError("hit")


def parse_row(row: dict, source: str = "") -> Trial:
    """One CSV row -> Trial. Raises (ValueError, KeyError) if the row is unusable."""
    trial = Trial(
        participant=row["participant"],
        input=row["input"],
        block=int(_number(row, "block")),
        amplitude=_number(row, "amplitude"),
        width=_number(row, "width"),
        target=(_number(row, "target_x"), _number(row, "target_y")),
        start=(_number(row, "start_x"), _number(row, "start_y")),
        click=(_number(row, "click_x"), _number(row, "click_y")),
        hit=_parse_hit(row["hit"]),
        mt_ms=_number(row, "movement_time_ms"),
        source=source,
    )
    if not trial.participant or not trial.input:
        raise ValueError("empty name")
    if trial.amplitude <= 0 or trial.width <= 0 or trial.mt_ms <= 0 or trial.axis_length <= 0:
        raise ValueError("not a real movement")
    return trial


def find_csv_files(paths) -> tuple[list[Path], list[str]]:
    """Expand folders (not recursively) -> (files, plain messages about problems)."""
    files, notes = [], []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found = sorted(path.glob("*.csv"))
            if not found:
                notes.append(f"这个文件夹里没有结果文件（.csv）：{path}")
            files += found
        elif path.is_file():
            files.append(path)
        else:
            notes.append(f"找不到这个文件或文件夹：{path}")
    return files, notes


def load_trials(files) -> tuple[list[Trial], list[str]]:
    """Read every file it can -> (trials, plain messages about what was skipped)."""
    trials, notes = [], []
    for path in map(Path, files):
        try:
            table = read_table(path)
        except (OSError, ValueError) as e:
            notes.append(f"打不开 {path.name}，已跳过（{e}）。")
            continue
        missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
        if missing:
            notes.append(f"{path.name} 不是测试结果文件，已跳过（缺少这些列：{', '.join(missing)}）。")
            continue
        bad_lines = []
        for line, row in table.rows:
            try:
                trials.append(parse_row(row, path.name))
            except (ValueError, KeyError):
                bad_lines.append(line)
        if bad_lines:
            shown = ", ".join(map(str, bad_lines[:5])) + (" ..." if len(bad_lines) > 5 else "")
            notes.append(f"{path.name} 有 {len(bad_lines)} 行数据不对，已跳过（第 {shown} 行）。")
    return trials, notes


# ------------------------------------------------------------------ measures
@dataclass(frozen=True)
class SequenceStats:
    """One sequence of trials. ``ide`` and ``tp`` are None if every click fell
    at the same depth (no spread), which gives no effective width to divide by."""

    participant: str
    input: str
    amplitude: float
    width: float
    n: int
    mt_ms: float
    error_rate: float
    sd_x: float
    we: float
    ae: float
    ide: float | None
    tp: float | None


def sequence_stats(trials: list[Trial]) -> SequenceStats:
    """Measures of one sequence (at least 2 trials, all with the same A x W)."""
    first = trials[0]
    deviations = [t.deviation for t in trials]
    mt_ms = statistics.fmean(t.mt_ms for t in trials)
    sd_x = statistics.stdev(deviations)
    we = EFFECTIVE_WIDTH_FACTOR * sd_x
    ae = statistics.fmean(t.axis_length + dx for t, dx in zip(trials, deviations))
    ide = math.log2(ae / we + 1) if we > 1e-9 and ae > 0 else None
    return SequenceStats(
        participant=first.participant,
        input=first.input,
        amplitude=first.amplitude,
        width=first.width,
        n=len(trials),
        mt_ms=mt_ms,
        error_rate=sum(not t.hit for t in trials) / len(trials),
        sd_x=sd_x,
        we=we,
        ae=ae,
        ide=ide,
        tp=ide / (mt_ms / 1000) if ide is not None else None,
    )


def split_sequences(trials: list[Trial]) -> tuple[list[SequenceStats], int]:
    """Group trials into sequences -> (stats, how many were too short to use)."""
    groups = defaultdict(list)
    for t in trials:
        groups[t.source, t.participant, t.input, t.block, t.amplitude, t.width].append(t)
    stats, too_short = [], 0
    for group in groups.values():
        if len(group) < MIN_SEQUENCE_TRIALS:
            too_short += 1
        else:
            stats.append(sequence_stats(group))
    return stats, too_short


@dataclass(frozen=True)
class Fit:
    """MT = a + b * x, MT in ms and x in bits. ``r2`` is NaN if MT never varies."""

    a: float
    b: float
    r2: float
    n: int


def linear_fit(x, y) -> Fit | None:
    """Least squares line, or None without at least two different x values."""
    x, y = list(x), list(y)
    if len(x) != len(y) or len(set(x)) < 2:
        return None
    mean_x, mean_y = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((v - mean_x) ** 2 for v in x)
    b = sum((u - mean_x) * (v - mean_y) for u, v in zip(x, y)) / sxx
    a = mean_y - b * mean_x
    ss_total = sum((v - mean_y) ** 2 for v in y)
    ss_residual = sum((v - (a + b * u)) ** 2 for u, v in zip(x, y))
    return Fit(a, b, 1 - ss_residual / ss_total if ss_total > 0 else float("nan"), len(x))


def _mean(values) -> float | None:
    usable = [v for v in values if v is not None]
    return statistics.fmean(usable) if usable else None


def _grouped(items, key) -> dict:
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return dict(groups)


def _person_mean(seqs: list[SequenceStats], attr: str) -> float | None:
    """One person: sequences -> A x W conditions -> a single number."""
    cells = _grouped(seqs, lambda s: (s.amplitude, s.width))
    return _mean(_mean(getattr(s, attr) for s in cell) for cell in cells.values())


def _people_mean(seqs: list[SequenceStats], attr: str) -> float | None:
    """The mean of means: every person counts once, however much data they have."""
    people = _grouped(seqs, lambda s: s.participant)
    return _mean(_person_mean(p, attr) for p in people.values())


@dataclass(frozen=True)
class Measures:
    trials: int
    mt_ms: float
    error_rate: float
    we: float
    ae: float
    ide: float | None
    tp: float | None


def _measures(seqs: list[SequenceStats], mean_of=_people_mean) -> Measures:
    values = {a: mean_of(seqs, a) for a in ("mt_ms", "error_rate", "we", "ae", "ide", "tp")}
    return Measures(trials=sum(s.n for s in seqs), **values)


@dataclass(frozen=True)
class ConditionRow:
    amplitude: float
    width: float
    measures: Measures

    @property
    def id(self) -> float:
        return fitts_id(self.amplitude, self.width)


@dataclass(frozen=True)
class InputSummary:
    input: str
    overall: Measures
    conditions: list[ConditionRow]
    people: dict[str, Measures]
    fit_id: Fit | None          # MT = a + b * ID   (nominal)
    fit_ide: Fit | None         # MT = a + b * IDe  (effective)


def summarize(sequences: list[SequenceStats]) -> list[InputSummary]:
    """One summary per input condition, gesture first."""
    out = []
    by_input = _grouped(sequences, lambda s: s.input)
    for name in sorted(by_input, key=lambda n: (INPUT_ORDER.index(n) if n in INPUT_ORDER else 99, n)):
        seqs = by_input[name]
        cells = _grouped(seqs, lambda s: (s.amplitude, s.width))
        conditions = [ConditionRow(a, w, _measures(cells[a, w])) for a, w in sorted(cells)]
        people = {p: _measures(ps, _person_mean)
                  for p, ps in sorted(_grouped(seqs, lambda s: s.participant).items())}
        effective = [c for c in conditions if c.measures.ide is not None]
        out.append(InputSummary(
            input=name,
            overall=_measures(seqs),
            conditions=conditions,
            people=people,
            fit_id=linear_fit([c.id for c in conditions], [c.measures.mt_ms for c in conditions]),
            fit_ide=linear_fit([c.measures.ide for c in effective], [c.measures.mt_ms for c in effective]),
        ))
    return out


@dataclass(frozen=True)
class PairedTest:
    first: str
    second: str
    n: int
    mean_difference: float      # first - second
    t: float
    df: int
    p: float | None             # None if SciPy is not available


def paired_throughput_test(summaries: list[InputSummary]) -> PairedTest | None:
    """Paired t-test on per-person throughput, for exactly two input conditions.
    None without 2 people who did both, or if their differences are all equal."""
    if len(summaries) != 2:
        return None
    one, two = summaries
    shared = sorted(p for p in set(one.people) & set(two.people)
                    if one.people[p].tp is not None and two.people[p].tp is not None)
    if len(shared) < 2:
        return None
    diffs = [one.people[p].tp - two.people[p].tp for p in shared]
    sd = statistics.stdev(diffs)
    if sd <= 1e-12:
        return None
    mean_diff = statistics.fmean(diffs)
    t = mean_diff / (sd / math.sqrt(len(diffs)))
    df = len(diffs) - 1
    try:
        from scipy.stats import t as student_t      # comes with scikit-learn
        p = float(2 * student_t.sf(abs(t), df))
    except ImportError:
        p = None
    return PairedTest(one.input, two.input, len(diffs), mean_diff, t, df, p)


# ------------------------------------------------------------------ reporting
def _table(headers: list[str], rows: list[list[str]], left: int = 1) -> list[str]:
    """Plain text table: the first ``left`` columns left-aligned, the rest right."""
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]

    def line(cells):
        return "  ".join(str(c).ljust(w) if i < left else str(c).rjust(w)
                         for i, (c, w) in enumerate(zip(cells, widths))).rstrip()

    return [line(headers), "-" * len(line(headers)), *(line(r) for r in rows)]


def _fmt(value: float | None, decimals: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{decimals}f}"


def _measure_cells(m: Measures) -> list[str]:
    return [str(m.trials), _fmt(m.mt_ms), _fmt(m.error_rate * 100), _fmt(m.we), _fmt(m.ae),
            _fmt(m.ide, 2), _fmt(m.tp, 2)]


_MEASURE_HEADERS = ["Trials", "MT(ms)", "Error%", "We(px)", "Ae(px)", "IDe(bits)", "TP(bits/s)"]


def _fit_text(fit: Fit | None, symbol: str) -> str:
    if fit is None:
        return "needs at least 2 different difficulty levels"
    r2 = "n/a" if math.isnan(fit.r2) else f"{fit.r2:.3f}"
    return f"MT = {fit.a:.1f} {'+' if fit.b >= 0 else '-'} {abs(fit.b):.1f} x {symbol}   R^2 = {r2}   ({fit.n} points)"


def format_report(summaries: list[InputSummary], detail: bool = False) -> str:
    people = {p for s in summaries for p in s.people}
    trials = sum(s.overall.trials for s in summaries)
    lines = [f"Fitts' law results: {len(people)} participant(s), {trials} trials", ""]
    lines += _table(["Input", "People", *_MEASURE_HEADERS],
                    [[s.input, str(len(s.people)), *_measure_cells(s.overall)] for s in summaries])

    lines += ["", "Regression on the A x W condition means (MT in ms):"]
    for s in summaries:
        lines.append(f"  {s.input}")
        lines.append(f"    nominal    {_fit_text(s.fit_id, 'ID')}")
        lines.append(f"    effective  {_fit_text(s.fit_ide, 'IDe')}")

    lines += ["", "Per participant:"]
    lines += _table(["Participant", "Input", *_MEASURE_HEADERS],
                    [[p, s.input, *_measure_cells(m)] for s in summaries for p, m in s.people.items()],
                    left=2)

    test = paired_throughput_test(summaries)
    if test is not None:
        p_text = "p = n/a" if test.p is None else ("p < 0.001" if test.p < 0.001 else f"p = {test.p:.3f}")
        lines += ["", f"Paired t-test on TP ({test.first} minus {test.second}, {test.n} people): "
                      f"mean difference = {test.mean_difference:+.2f} bits/s, "
                      f"t({test.df}) = {test.t:.2f}, {p_text}"]

    if detail:
        lines += ["", "Per A x W condition:"]
        lines += _table(["Input", "A", "W", "ID", *_MEASURE_HEADERS],
                        [[s.input, f"{c.amplitude:g}", f"{c.width:g}", f"{c.id:.2f}", *_measure_cells(c.measures)]
                         for s in summaries for c in s.conditions])

    lines += ["", "怎么看这些表：",
              "  MT = 平均每次用时（越小越快）；Error% = 没点中的比例（越小越准）",
              "  We = 点击实际散开的范围；Ae = 实际移动的距离；IDe = 目标的实际难度",
              "  TP = 把快和准合在一起的总分（越大越好），比较两种操作方式主要看这一列",
              "  R^2 = 用时和难度有多贴合一条直线（越接近 1 越贴合）"]
    return "\n".join(lines)


def save_plot(summaries: list[InputSummary], path: str | Path) -> Path:
    """MT against nominal ID, one line per input condition.

    Written through BytesIO + write_bytes so non-ASCII folders are safe, and with
    the Figure class directly so no GUI backend is ever started.
    """
    import matplotlib
    from matplotlib.figure import Figure

    style = {"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10}
    with matplotlib.rc_context(style):
        fig = Figure(figsize=(7.2, 4.4), dpi=150, facecolor=SURFACE)
        ax = fig.add_subplot()
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.75, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
            ax.spines[side].set_linewidth(0.75)

        tops = [c.measures.mt_ms for s in summaries for c in s.conditions]
        for s in summaries:
            color = SERIES_COLORS.get(s.input, OTHER_SERIES_COLOR)
            xs = [c.id for c in s.conditions]
            ys = [c.measures.mt_ms for c in s.conditions]
            label = s.input
            if s.fit_id is not None:
                ends = [min(xs), max(xs)]
                line_y = [s.fit_id.a + s.fit_id.b * x for x in ends]
                ax.plot(ends, line_y, color=color, linewidth=1.5, solid_capstyle="round", zorder=2)
                if not math.isnan(s.fit_id.r2):
                    label = (f"{s.input}: MT = {s.fit_id.a:.0f} {'+' if s.fit_id.b >= 0 else '-'} "
                             f"{abs(s.fit_id.b):.0f} x ID, R² = {s.fit_id.r2:.2f}")
                ax.annotate(s.input, (ends[1], line_y[1]), xytext=(8, 0), textcoords="offset points",
                            va="center", color=INK_2, fontsize=9)
            ax.plot(xs, ys, linestyle="none", marker="o", markersize=6.5, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3)

        ax.set_xlabel("Index of difficulty, ID = log2(A / W + 1) (bits)", color=INK_2)
        ax.set_ylabel("Movement time (ms)", color=INK_2)
        ax.set_ylim(bottom=0, top=max(tops) * 1.12)
        ax.margins(x=0.12)
        ax.set_title("Movement time against target difficulty", loc="left", color=INK, fontsize=12, pad=12)
        if len(summaries) >= 2:
            legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
            for text in legend.get_texts():
                text.set_color(INK_2)
        else:
            ax.text(0, 1.01, summaries[0].input, transform=ax.transAxes, color=INK_2, fontsize=9)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())
    return path


# ---------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser(description="分析目标点击测试的结果（task_runner 产生的 .csv 文件）。")
    ap.add_argument("paths", nargs="*", default=[str(RESULTS_DIR)],
                    help="结果文件或文件夹（默认 evaluation/results）")
    ap.add_argument("--detail", action="store_true", help="再列出每一种距离和大小组合的结果")
    ap.add_argument("--exclude-outliers", action="store_true", help="去掉明显的误触再算")
    ap.add_argument("--plot", nargs="?", const="", default=None, metavar="PNG",
                    help="同时画一张图（默认保存在第一个结果文件夹里，名字是 fitts_plot.png）")
    args = ap.parse_args(argv)

    files, notes = find_csv_files(args.paths)
    trials, load_notes = load_trials(files)
    for note in notes + load_notes:
        print(note)
    if not trials:
        print("没有可以分析的数据。请先用 task_runner 做一次测试：\n"
              "  python -m evaluation.task_runner --participant P01 --input gesture")
        return 1

    outliers = [t for t in trials if t.is_spatial_outlier]
    if args.exclude_outliers:
        trials = [t for t in trials if not t.is_spatial_outlier]

    sequences, too_short = split_sequences(trials)
    if not sequences:
        print("每组数据都太少（一组至少要 3 次点击），没法计算。请完整做完一组再分析。")
        return 1
    summaries = summarize(sequences)
    print(format_report(summaries, args.detail))
    print()
    if too_short:
        print(f"有 {too_short} 组数据太少（不到 {MIN_SEQUENCE_TRIALS} 次点击），没有算进去。")
    no_spread = sum(s.ide is None for s in sequences)
    if no_spread:
        print(f"有 {no_spread} 组的点击完全没有散开，算不出 TP，这些组的 TP 没有算进去。")
    if args.exclude_outliers:
        print(f"已去掉 {len(outliers)} 次明显的误触。")
    elif outliers:
        print(f"其中有 {len(outliers)} 次明显的误触（还算在里面）。加上 --exclude-outliers 可以去掉它们再算一次。")

    if args.plot is not None:
        if args.plot:
            target = Path(args.plot)
        else:
            folder = next((Path(p) for p in args.paths if Path(p).is_dir()), files[0].parent)
            target = folder / "fitts_plot.png"
        print(f"图已保存：{save_plot(summaries, target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
