"""Measure a saved gesture model on labelled hands and draw its confusion matrix.

Predictions are made the way the live classifier makes them: argmax of
``predict_proba``, and with a threshold anything less certain becomes "none"
(the app then does nothing for that frame), which counts as an error for the
true class. Macro-F1 averages over the classes that actually occur in the test
data, so an unused "none" column cannot drag it down.

The PNG goes through ``io.BytesIO`` + ``Path.write_bytes`` because file writers
that take a path can fail on non-ASCII Windows folders.

CLI:  python -m ml.evaluate --model models/gesture_model.joblib --data CSV
          [--out docs/figures/ml_confusion_matrix.png] [--threshold 0.55]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from airtouch.config import GESTURE_MODEL_PATH
from airtouch.console import use_utf8_console
from airtouch.features import DEFAULT_ASPECT, feature_matrix
from airtouch.gesture_classifier import _single_thread
from ml.synthetic_data import ALL_GESTURES, load_csv

NONE = "none"
LIVE_THRESHOLD = 0.55       # Config.ml_confidence default, so the CLI matches the app out of the box

# Report-figure tokens: one-hue sequential ramp on a light surface; ink stays neutral.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_RAMP = ("#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")


def load_bundle(bundle_or_path) -> dict:
    return bundle_or_path if isinstance(bundle_or_path, dict) else joblib.load(Path(bundle_or_path))


def _label_order(present: set[str]) -> list[str]:
    """Gestures in their usual order, then anything unknown, "none" last."""
    known = [g for g in ALL_GESTURES if g in present]
    other = sorted(present - set(ALL_GESTURES) - {NONE})
    return known + other + ([NONE] if NONE in present else [])


def evaluate(bundle_or_path, hands, labels, threshold: float | None = None) -> dict:
    """Score a bundle (or a path to one) on ``hands``; ``threshold=None`` never rejects.

    Returns accuracy, macro_f1, per_class (precision / recall / f1 / support),
    confusion_matrix (rows = true, columns = predicted) and labels (their order).
    """
    bundle = load_bundle(bundle_or_path)
    model = bundle["model"]
    _single_thread(model)
    X = feature_matrix(hands, bundle.get("aspect", DEFAULT_ASPECT), **bundle.get("feature_flags", {}))
    proba = model.predict_proba(X)
    classes = [str(c) for c in model.classes_]
    best = np.argmax(proba, axis=1)
    y_pred = [classes[i] for i in best]
    if threshold is not None:
        y_pred = [NONE if p < threshold else c for c, p in zip(y_pred, proba.max(axis=1))]
    y_true = [str(label) for label in labels]

    order = _label_order(set(classes) | set(y_true) | set(y_pred))
    occurring = [label for label in order if label in set(y_true)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=order, zero_division=0)
    return {
        "model_name": bundle.get("name", type(model).__name__),
        "threshold": threshold,
        "n_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=occurring, average="macro", zero_division=0)),
        "rejected": y_pred.count(NONE),
        "labels": order,
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(order)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=order).tolist(),
    }


def format_report(result: dict) -> str:
    threshold = "off" if result["threshold"] is None else f"{result['threshold']:.2f}"
    lines = [
        f"Model: {result['model_name']}   samples: {result['n_samples']}   threshold: {threshold}",
        f"Accuracy: {result['accuracy']:.4f}   macro-F1: {result['macro_f1']:.4f}   "
        f"rejected as 'none': {result['rejected']}",
        "",
        f"{'gesture':<14}{'precision':>11}{'recall':>9}{'F1':>9}{'support':>9}",
    ]
    for label in result["labels"]:
        c = result["per_class"][label]
        lines.append(f"{label:<14}{c['precision']:>11.4f}{c['recall']:>9.4f}{c['f1']:>9.4f}{c['support']:>9d}")
    width = max(len(label) for label in result["labels"]) + 2
    lines += ["", "Confusion matrix (rows = true, columns = predicted):",
              " " * width + "".join(f"{label[:9]:>10}" for label in result["labels"])]
    for label, row in zip(result["labels"], result["confusion_matrix"]):
        lines.append(f"{label:<{width}}" + "".join(f"{v:>10d}" for v in row))
    return "\n".join(lines)


def plot_confusion(result: dict, out: str | Path, title: str | None = None) -> Path:
    """Save the matrix as a PNG. Colour = share of that true gesture; rows with no samples are left out."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    labels = result["labels"]
    cm = np.asarray(result["confusion_matrix"], dtype=np.float64)
    rows = [i for i in range(len(labels)) if cm[i].sum() > 0]
    counts = cm[rows]
    share = counts / counts.sum(axis=1, keepdims=True)
    n_rows, n_cols = counts.shape

    fig = Figure(figsize=(max(6.4, 1.02 * n_cols + 2.2), max(4.2, 0.72 * n_rows + 1.9)), dpi=160,
                 facecolor=_SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ramp = LinearSegmentedColormap.from_list("airtouch_blue", _RAMP)
    image = ax.imshow(share, cmap=ramp, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n_cols), labels, rotation=0 if n_cols <= 7 else 35,
                  ha="center" if n_cols <= 7 else "right", fontsize=9, color=_INK_SECONDARY)
    ax.set_yticks(range(n_rows), [labels[i] for i in rows], fontsize=9, color=_INK_SECONDARY)
    ax.set_xlabel("Predicted gesture", fontsize=10, color=_INK_SECONDARY, labelpad=8)
    ax.set_ylabel("True gesture", fontsize=10, color=_INK_SECONDARY, labelpad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # a surface-coloured gap between cells, instead of ink grid lines
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color=_SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    for r in range(n_rows):
        for c in range(n_cols):
            if counts[r, c] == 0:       # empty cells stay quiet so the mistakes stand out
                continue
            ink = "#ffffff" if share[r, c] >= 0.5 else _INK
            ax.text(c, r, f"{int(counts[r, c])}\n{share[r, c]:.1%}", ha="center", va="center",
                    fontsize=9, color=ink, linespacing=1.25)

    threshold = "no threshold" if result["threshold"] is None else f"threshold {result['threshold']:.2f}"
    ax.set_title(title or f"Gesture recogniser: {result['model_name']}", loc="left", fontsize=12,
                 fontweight="bold", color=_INK, pad=24)
    ax.text(0.0, 1.02, f"{result['n_samples']} hands, {threshold}: accuracy {result['accuracy']:.1%}, "
                       f"macro-F1 {result['macro_f1']:.3f}",
            transform=ax.transAxes, fontsize=9, color=_INK_SECONDARY)

    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 0.5, 1])
    bar.ax.set_yticklabels(["0%", "50%", "100%"], fontsize=8, color=_INK_SECONDARY)
    bar.set_label("Share of the true gesture", fontsize=9, color=_INK_SECONDARY)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=_SURFACE)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buf.getvalue())
    return out


def main(argv=None) -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(GESTURE_MODEL_PATH), help="saved model bundle")
    ap.add_argument("--data", required=True, help="CSV of labelled hands (synthetic or recorded; same format)")
    ap.add_argument("--out", help="save the confusion matrix as this PNG")
    ap.add_argument("--threshold", type=float, default=LIVE_THRESHOLD,
                    help="less certain predictions become 'none', like the live app (0 = never)")
    a = ap.parse_args(argv)

    try:
        hands, labels = load_csv(a.data)
        if len(labels) == 0:
            raise ValueError(f"文件里没有数据：{a.data}")
        result = evaluate(a.model, hands, labels, a.threshold)
        print(format_report(result))
        if a.out:
            print(f"\n混淆矩阵图已保存：{plot_confusion(result, a.out)}")
    except FileNotFoundError as e:
        print(f"找不到文件：{e.filename}", file=sys.stderr)
        return 1
    except (ValueError, OSError, KeyError) as e:
        print(f"评估没有成功：{e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
