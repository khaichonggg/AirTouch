"""Synthetic hand-landmark generator.

Builds a simple 3-D kinematic hand (palm + 3-joint fingers + thumb), poses
it for a gesture, then randomly rotates / scales / moves / mirrors it and
adds per-landmark jitter, finally projecting to MediaPipe-style normalised
image coordinates. This lets the whole pipeline (rules, ML training,
simulator, research studies) be developed and verified with no webcam.

Real recordings from ``ml/record_gestures.py`` use the exact same CSV
format, so they are a drop-in replacement.

CLI:  python -m ml.synthetic_data --n 300 --out data/synthetic.csv
      [--extended] [--difficulty normal|hard|very_hard] [--seed 0]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GESTURES = ("point", "pinch_index", "pinch_middle", "scroll", "fist", "open_palm")
EXTENDED_GESTURES = ("three_finger", "rock", "thumbs_up", "ok_sign")
ALL_GESTURES = GESTURES + EXTENDED_GESTURES

DIFFICULTY = {
    #             jitter   in-plane / tilt (deg)       curl-range overlap, joint noise (deg)
    "normal": dict(noise=0.0025, roll=35, pitch=15, yaw=20, overlap=0.00, joint=4.0),
    "hard": dict(noise=0.0050, roll=50, pitch=25, yaw=30, overlap=0.15, joint=8.0),
    "very_hard": dict(noise=0.0090, roll=65, pitch=35, yaw=40, overlap=0.25, joint=12.0),
}

# --- hand geometry (hand units: wrist -> middle knuckle = 1; x right, y up, z towards camera)
_MCP = {"index": (-0.30, 0.95), "middle": (0.0, 1.0), "ring": (0.27, 0.93), "pinky": (0.50, 0.82)}
_LEN = {
    "index": (0.42, 0.25, 0.20),
    "middle": (0.46, 0.29, 0.22),
    "ring": (0.43, 0.27, 0.21),
    "pinky": (0.34, 0.20, 0.18),
}
_SPLAY = {"index": -8.0, "middle": 0.0, "ring": 7.0, "pinky": 15.0}
_IDX = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
_FLEX = np.array([85.0, 105.0, 75.0])          # MCP, PIP, DIP flexion at curl = 1
_THUMB_CMC = np.array([-0.20, 0.22, 0.04])
_THUMB_MCP = np.array([-0.42, 0.42, 0.10])
_PALM_CENTER = np.array([0.094, 0.74, 0.0])

# curl ranges (0 = straight, 1 = fully curled)
EXT, CURL, RELAX, PINCH = (0.0, 0.12), (0.80, 1.0), (0.45, 1.0), (0.35, 0.55)

# finger -> list of allowed curl ranges (one is picked at random); thumb mode
SPEC = {
    "point": dict(index=[EXT], middle=[CURL], ring=[CURL], pinky=[CURL], thumb=("tuck", "middle")),
    "pinch_index": dict(index=[PINCH], middle=[RELAX], ring=[RELAX], pinky=[RELAX], thumb=("pinch", "index")),
    "pinch_middle": dict(index=[EXT, CURL], middle=[PINCH], ring=[RELAX], pinky=[RELAX], thumb=("pinch", "middle")),
    "scroll": dict(index=[EXT], middle=[EXT], ring=[CURL], pinky=[CURL], thumb=("tuck", "ring")),
    "fist": dict(index=[CURL], middle=[CURL], ring=[CURL], pinky=[CURL], thumb=("tuck", "fist")),
    "open_palm": dict(index=[EXT], middle=[EXT], ring=[EXT], pinky=[EXT], thumb=("out", -50.0)),
    "three_finger": dict(index=[EXT], middle=[EXT], ring=[EXT], pinky=[CURL], thumb=("tuck", "pinky")),
    "rock": dict(index=[EXT], middle=[CURL], ring=[CURL], pinky=[EXT], thumb=("tuck", "middle")),
    "thumbs_up": dict(index=[CURL], middle=[CURL], ring=[CURL], pinky=[CURL], thumb=("out", -12.0)),
    "ok_sign": dict(index=[PINCH], middle=[EXT], ring=[EXT], pinky=[EXT], thumb=("pinch", "index")),
}


@dataclass
class HandPose:
    """Where the hand is and how it is oriented in the camera image."""

    cx: float = 0.5          # palm centre, normalised image x
    cy: float = 0.55         # palm centre, normalised image y
    size: float = 0.16       # wrist->middle-knuckle length, as a fraction of image height
    roll: float = 0.0        # in-plane rotation (deg)
    pitch: float = 0.0       # tilt towards/away from camera (deg)
    yaw: float = 0.0         # turn left/right (deg)
    mirror: bool = False     # True = the other hand


def random_pose(rng: np.random.Generator, difficulty: str = "normal") -> HandPose:
    d = DIFFICULTY[difficulty]
    return HandPose(
        cx=rng.uniform(0.3, 0.7),
        cy=rng.uniform(0.4, 0.7),
        size=rng.uniform(0.10, 0.22),
        roll=rng.uniform(-d["roll"], d["roll"]),
        pitch=rng.uniform(-d["pitch"], d["pitch"]),
        yaw=rng.uniform(-d["yaw"], d["yaw"]),
        mirror=bool(rng.random() < 0.5),
    )


def _widen(rng_range, overlap):
    lo, hi = rng_range
    if rng_range == EXT:
        return lo, hi + overlap
    if rng_range in (CURL, RELAX):
        return lo - overlap, hi
    if rng_range == PINCH:
        return lo - overlap / 2, hi + overlap / 2
    return lo, hi


def _finger(name, curl, rng, joint_noise):
    mcp = np.array([*_MCP[name], 0.0])
    a = np.radians(_SPLAY[name] + rng.normal(0, 2.0))
    d = np.array([np.sin(a), np.cos(a), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    angles = _FLEX * float(np.clip(curl, 0.0, 1.0)) + rng.normal(0, joint_noise, 3)
    angles = np.radians(np.clip(angles, -10.0, 125.0))
    pts, phi = [mcp], 0.0
    for length, th in zip(_LEN[name], angles):
        phi += th
        pts.append(pts[-1] + length * (np.cos(phi) * d + np.sin(phi) * z))
    return pts  # mcp, pip, dip, tip


def _thumb_out(angle_deg, rng, joint_noise):
    a = np.radians(angle_deg + rng.normal(0, joint_noise))
    d = np.array([np.sin(a), np.cos(a), 0.15])
    d /= np.linalg.norm(d)
    ip = _THUMB_MCP + 0.30 * d
    return ip, ip + 0.26 * d


def _thumb_to(target, rng, joint_noise):
    tip = np.asarray(target, dtype=np.float64)
    bulge = np.array([-0.07, 0.0, 0.08]) * (1 + rng.normal(0, joint_noise / 40))
    ip = (_THUMB_MCP + tip) / 2 + bulge
    return ip, tip


def _rot(roll, pitch, yaw):
    r, p, y = np.radians([roll, pitch, yaw])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return rz @ rx @ ry


def make_hand(
    gesture: str,
    rng: np.random.Generator | None = None,
    difficulty: str = "normal",
    aspect: float = 4 / 3,
    pose: HandPose | None = None,
    thumb_gap: float | None = None,
    curls: dict | None = None,
    noise: float | None = None,
) -> np.ndarray:
    """One synthetic hand as a (21, 3) array of normalised image coordinates.

    thumb_gap: for pinch gestures, thumb-tip to finger-tip distance in hand
               units (default: small random touch). Lets the simulator animate
               a pinch closing and opening.
    curls:     override curl of named fingers, e.g. {"index": 0.7}.
    noise:     override per-landmark jitter (normalised units).
    """
    if gesture not in SPEC:
        raise ValueError(f"unknown gesture {gesture!r}")
    rng = rng if rng is not None else np.random.default_rng()
    diff = DIFFICULTY[difficulty]
    spec = SPEC[gesture]
    pose = pose if pose is not None else random_pose(rng, difficulty)

    pts = np.zeros((21, 3))
    chains = {}
    for name in ("index", "middle", "ring", "pinky"):
        if curls and name in curls:
            c = curls[name]
        else:
            choices = spec[name]
            lo, hi = _widen(choices[rng.integers(len(choices))], diff["overlap"])
            c = rng.uniform(lo, hi)
        chain = _finger(name, c, rng, diff["joint"])
        chains[name] = chain
        for idx, p in zip(_IDX[name], chain):
            pts[idx] = p

    mode, arg = spec["thumb"]
    if mode == "out":
        ip, tip = _thumb_out(arg, rng, diff["joint"])
    elif mode == "pinch":
        target = chains[arg][3]
        gap = thumb_gap if thumb_gap is not None else rng.uniform(0.0, 0.08)
        u = rng.normal(0, 1, 3)
        u /= np.linalg.norm(u)
        ip, tip = _thumb_to(target + gap * u, rng, diff["joint"])
    else:  # tuck: thumb tip rests over the middle phalanx of a curled finger
        if arg == "fist":
            base = (chains["index"][1] + chains["middle"][1]) / 2 + np.array([0.0, -0.05, 0.12])
        else:
            c = chains[arg]
            base = (c[1] + c[2]) / 2 + np.array([-0.04, 0.0, 0.10])
        ip, tip = _thumb_to(base + rng.normal(0, 0.02 + diff["overlap"] * 0.1, 3), rng, diff["joint"])
    pts[1], pts[2], pts[3], pts[4] = _THUMB_CMC, _THUMB_MCP, ip, tip
    pts[0] = 0.0

    # pose: mirror, rotate about the palm centre, scale, place, project
    p = pts - _PALM_CENTER
    if pose.mirror:
        p[:, 0] = -p[:, 0]
    p = p @ _rot(pose.roll, pose.pitch, pose.yaw).T
    s = pose.size
    out = np.empty_like(p)
    out[:, 0] = (pose.cx * aspect + s * p[:, 0]) / aspect   # back to width-normalised x
    out[:, 1] = pose.cy - s * p[:, 1]                       # image y grows downwards
    out[:, 2] = -s * p[:, 2]                                # smaller z = closer to camera
    sigma = diff["noise"] if noise is None else noise
    if sigma > 0:
        out += rng.normal(0, sigma, out.shape)
    return out


def generate_dataset(
    n_per_class: int = 300,
    gestures=GESTURES,
    difficulty: str = "normal",
    seed: int = 0,
    aspect: float = 4 / 3,
):
    """Return (hands (N, 21, 3), labels (N,)) - a balanced, shuffled set."""
    rng = np.random.default_rng(seed)
    hands, labels = [], []
    for g in gestures:
        for _ in range(n_per_class):
            hands.append(make_hand(g, rng, difficulty=difficulty, aspect=aspect))
            labels.append(g)
    order = rng.permutation(len(labels))
    return np.asarray(hands)[order], np.asarray(labels)[order]


# --- CSV ---------------------------------------------------------------------
HEADER = ["label"] + [f"{a}{i}" for i in range(21) for a in "xyz"]


def save_csv(path, hands, labels) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for h, lab in zip(hands, labels):
            w.writerow([lab] + [f"{v:.6f}" for v in np.asarray(h).reshape(-1)])


def load_csv(path):
    hands, labels = [], []
    with Path(path).open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) != 64:
                continue
            labels.append(row[0])
            hands.append(np.asarray(row[1:], dtype=np.float64).reshape(21, 3))
    return np.asarray(hands), np.asarray(labels)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="samples per gesture")
    ap.add_argument("--out", default="data/synthetic.csv")
    ap.add_argument("--extended", action="store_true", help="add the 4 confusable extra gestures")
    ap.add_argument("--difficulty", choices=list(DIFFICULTY), default="normal")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    gestures = ALL_GESTURES if a.extended else GESTURES
    hands, labels = generate_dataset(a.n, gestures, a.difficulty, a.seed)
    save_csv(a.out, hands, labels)
    print(f"wrote {len(labels)} samples ({len(gestures)} gestures, {a.difficulty}) -> {a.out}")


if __name__ == "__main__":
    main()
