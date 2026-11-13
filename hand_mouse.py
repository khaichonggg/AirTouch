"""AirTouch entry point: control the mouse with your hand in front of a webcam.

Run:  python hand_mouse.py [--lite] [--engine rule|ml] [--camera N]
                           [--two-hands] [--dwell] [--config PATH]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from airtouch.console import use_utf8_console


def main(argv=None) -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser(description="AirTouch：对着摄像头用手势控制鼠标。", add_help=False)
    ap.add_argument("-h", "--help", action="help", help="显示这份说明")
    ap.add_argument("--lite", action="store_true", help="不显示画面窗口，最省电脑资源（按 Ctrl+C 退出）")
    ap.add_argument("--engine", choices=("rule", "ml"), help="识别方式：rule 是默认方式，ml 是训练好的手势模型")
    ap.add_argument("--camera", type=int, metavar="N", help="用第几个摄像头（0 是第一个）")
    ap.add_argument("--two-hands", action="store_true", help="同时认两只手，可以用两只手缩放")
    ap.add_argument("--dwell", action="store_true", help="打开停留点击：鼠标停住一小会儿就自动点一下")
    ap.add_argument("--config", metavar="PATH", help="设置文件的位置（默认是这个文件夹里的 config.json）")
    args = ap.parse_args(argv)

    # Imported after the arguments are read, so --help answers instantly.
    from airtouch import app
    from airtouch.config import Config

    overrides = {}
    if args.engine:
        overrides["engine"] = args.engine
    if args.camera is not None:
        overrides["camera_index"] = args.camera
    if args.two_hands:
        overrides["max_hands"] = 2
    if args.dwell:
        overrides["dwell_enabled"] = True
    cfg = dataclasses.replace(Config.load(args.config), **overrides)
    return app.main(cfg, show_window=not args.lite)


if __name__ == "__main__":
    sys.exit(main())
