"""Train the candidate gesture classifiers and deploy the right one.

    landmarks -> features -> stratified 80/20 split
    -> 5-fold CV on the 80% (macro-F1) -> fit on the 80% -> score on the 20%
    -> single-sample latency -> choose -> refit the winner on ALL data -> save

Deployment rule: the model runs on every camera frame, so among the models
whose test macro-F1 is within ``TIE_TOLERANCE`` of the best, the FASTEST wins
(the original project found RandomForest tied with the MLP on accuracy, at about
45 ms per prediction against 0.6 ms).

The winner is refitted on all the data before saving: the 80/20 split exists
only to compare the models honestly, and once that decision is made, holding
20% of the examples back would only make the deployed model worse. The scores
in the report therefore describe the 80% fit, a slightly conservative estimate
of the saved model.

Scores use the argmax of ``predict_proba`` because that is the only call the
live classifier makes (an SVM's ``predict`` can disagree with it slightly).

CLI:  python -m ml.train_model [--data CSV | --synthetic N] [--extended]
          [--difficulty normal|hard|very_hard] [--out models/gesture_model.joblib]
          [--report models/training_report.json] [--seed 0] [--aspect 1.3333]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from airtouch.config import GESTURE_MODEL_PATH
from airtouch.console import use_utf8_console
from airtouch.features import DEFAULT_ASPECT, feature_matrix
from airtouch.gesture_classifier import _single_thread
from ml.models import make_models
from ml.synthetic_data import ALL_GESTURES, DIFFICULTY, GESTURES, generate_dataset, load_csv

TIE_TOLERANCE = 0.005
TEST_FRACTION = 0.2
CV_FOLDS = 5
LATENCY_CALLS = 200
MIN_PER_CLASS = 10          # below this the split + 5 folds leave a class with too few examples
DEFAULT_FLAGS = {"translate": True, "rotate": True, "scale": True}


def _predict(model, X: np.ndarray) -> np.ndarray:
    """Labels exactly as the live classifier derives them."""
    return np.asarray(model.classes_)[np.argmax(model.predict_proba(X), axis=1)]


def _macro_f1_scorer(model, X: np.ndarray, y: np.ndarray) -> float:
    return float(f1_score(y, _predict(model, X), average="macro"))


def measure_latency(model, X: np.ndarray, n: int = LATENCY_CALLS) -> float:
    """Median milliseconds for one single-sample ``predict_proba`` (n_jobs forced to 1)."""
    _single_thread(model)
    X = np.atleast_2d(X)
    times = []
    for i in range(n):
        x = X[i % len(X)][None, :]
        t0 = time.perf_counter()
        model.predict_proba(x)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000.0)


def _tied(rows: list[dict], tol: float) -> list[dict]:
    best = max(r["test_f1"] for r in rows)
    return [r for r in rows if r["test_f1"] >= best - tol - 1e-12]   # 1e-12: float fuzz at the edge


def choose_model(report_rows: list[dict], tol: float = TIE_TOLERANCE) -> str:
    """Name of the fastest model among those within ``tol`` of the best test macro-F1."""
    if not report_rows:
        raise ValueError("no models to choose from")
    return min(_tied(report_rows, tol), key=lambda r: r["latency_ms"])["name"]


def _reason(rows: list[dict], chosen: str, tol: float) -> str:
    tied = _tied(rows, tol)
    best = max(r["test_f1"] for r in rows)
    if len(tied) == 1:
        return f"{chosen} has the best test macro-F1 ({best:.4f}); no other model is within {tol}."
    names = ", ".join(r["name"] for r in tied)
    latency = next(r["latency_ms"] for r in rows if r["name"] == chosen)
    return (f"{names} are within {tol} of the best test macro-F1 ({best:.4f}); {chosen} is the "
            f"fastest of them at {latency:.2f} ms per prediction, and the model runs on every "
            "camera frame.")


def _check_labels(y: np.ndarray) -> None:
    names, counts = np.unique(y, return_counts=True)
    if len(names) < 2:
        raise ValueError("训练数据里至少要有 2 种手势。")
    short = [f"{name}（{count} 条）" for name, count in zip(names, counts) if count < MIN_PER_CLASS]
    if short:
        raise ValueError(f"训练数据不够：每种手势至少要 {MIN_PER_CLASS} 条，这些太少了：{'、'.join(short)}。")


def train_and_select(
    hands,
    labels,
    seed: int = 0,
    aspect: float = DEFAULT_ASPECT,
    feature_flags: dict | None = None,
    data_source: str = "in-memory",
    tol: float = TIE_TOLERANCE,
) -> tuple[dict, dict]:
    """Compare the candidates and refit the winner on everything.

    Returns (bundle, report): ``bundle`` is what ``GestureClassifier`` loads,
    ``report`` is JSON-ready.
    """
    flags = {**DEFAULT_FLAGS, **(feature_flags or {})}
    y = np.asarray([str(label) for label in labels])
    if len(y) != len(hands):
        raise ValueError("手部数据和手势名称的数量对不上。")
    _check_labels(y)
    X = feature_matrix(hands, aspect, **flags)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=seed)
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)

    rows = []
    for name, model in make_models(seed).items():
        cv = cross_val_score(model, X_train, y_train, cv=folds, scoring=_macro_f1_scorer)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        y_pred = _predict(model, X_test)
        rows.append({
            "name": name,
            "cv_f1": round(float(cv.mean()), 6),
            "cv_f1_std": round(float(cv.std()), 6),
            "test_f1": round(float(f1_score(y_test, y_pred, average="macro")), 6),
            "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
            "latency_ms": round(measure_latency(model, X_test), 4),
            "fit_seconds": round(fit_s, 3),
        })

    chosen = choose_model(rows, tol)
    final = make_models(seed)[chosen]
    final.fit(X, y)
    _single_thread(final)       # the bundle is loaded for one-frame-at-a-time prediction

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    label_names = [str(c) for c in final.classes_]
    metrics = {r["name"]: {k: v for k, v in r.items() if k != "name"} for r in rows}
    bundle = {
        "model": final,
        "name": chosen,
        "feature_flags": flags,
        "aspect": float(aspect),
        "labels": label_names,
        "metrics": metrics,
        "trained_at": trained_at,
        "n_samples": int(len(y)),
        "data_source": data_source,
        "sklearn_version": sklearn.__version__,    # a pickle only loads reliably on a matching version
    }
    report = {
        "trained_at": trained_at,
        "data_source": data_source,
        "n_samples": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "labels": label_names,
        "seed": seed,
        "aspect": float(aspect),
        "feature_flags": flags,
        "cv_folds": CV_FOLDS,
        "test_fraction": TEST_FRACTION,
        "tie_tolerance": tol,
        "models": rows,
        "deployed": chosen,
        "reason": _reason(rows, chosen, tol),
        "refit_on_all_data": True,
        "note": ("Scores come from the 80% fit; the saved model was then refitted on all "
                 f"{len(y)} samples, so they are a slightly conservative estimate of it."),
    }
    return bundle, report


def save_outputs(bundle: dict, report: dict, out: str | Path,
                 report_path: str | Path | None = None) -> tuple[Path, Path]:
    """Write the model (via a temporary name, so a crash cannot leave half a file) and the JSON report."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    joblib.dump(bundle, tmp)
    tmp.replace(out)
    report_path = Path(report_path) if report_path else out.with_name("training_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out, report_path


def format_table(report: dict) -> str:
    head = f"{'model':<14}{'CV macro-F1':>20}{'test macro-F1':>16}{'test accuracy':>16}{'ms/predict':>13}"
    lines = [head, "-" * len(head)]
    for r in report["models"]:
        mark = "  <- deployed" if r["name"] == report["deployed"] else ""
        cv = f"{r['cv_f1']:.4f} +/- {r['cv_f1_std']:.4f}"
        lines.append(f"{r['name']:<14}{cv:>20}{r['test_f1']:>16.4f}{r['test_accuracy']:>16.4f}"
                     f"{r['latency_ms']:>13.2f}{mark}")
    return "\n".join(lines)


def main(argv=None) -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--data", help="CSV of labelled hands (synthetic or recorded; same format)")
    source.add_argument("--synthetic", type=int, metavar="N", help="generate N hands per gesture (default: 300)")
    ap.add_argument("--extended", action="store_true", help="with --synthetic: add the 4 extra gestures")
    ap.add_argument("--difficulty", choices=list(DIFFICULTY), default="normal", help="with --synthetic")
    ap.add_argument("--out", default=str(GESTURE_MODEL_PATH), help="where to save the model bundle")
    ap.add_argument("--report", help="where to save the JSON report (default: training_report.json next to --out)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aspect", type=float, default=DEFAULT_ASPECT, help="camera width / height of the data")
    a = ap.parse_args(argv)

    try:
        if a.data:
            hands, labels = load_csv(a.data)
            if len(labels) == 0:
                raise ValueError(f"文件里没有数据：{a.data}")
            data_source = a.data
        else:
            n = 300 if a.synthetic is None else a.synthetic
            gestures = ALL_GESTURES if a.extended else GESTURES
            hands, labels = generate_dataset(n, gestures, a.difficulty, a.seed, a.aspect)
            data_source = (f"synthetic (n_per_class={n}, gestures={len(gestures)}, "
                           f"difficulty={a.difficulty}, seed={a.seed})")
        bundle, report = train_and_select(hands, labels, seed=a.seed, aspect=a.aspect, data_source=data_source)
        out, report_path = save_outputs(bundle, report, a.out, a.report)
    except FileNotFoundError:
        print(f"找不到数据文件：{a.data}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as e:
        print(f"训练没有成功：{e}", file=sys.stderr)
        return 1

    print(f"Data: {report['data_source']}  ({report['n_train']} train / {report['n_test']} test)")
    print(format_table(report))
    print(f"Deployed: {report['deployed']}. {report['reason']}")
    print(f"Refitted on all {report['n_samples']} samples before saving.")
    print(f"模型已保存：{out}")
    print(f"训练报告已保存：{report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
