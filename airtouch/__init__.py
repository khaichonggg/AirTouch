"""AirTouch - control the Windows mouse with hand gestures in front of a webcam.

Pipeline: webcam frame -> MediaPipe hand landmarks -> gesture recognition
(rule engine or ML engine) -> debounce -> mouse controller (move / click /
drag / right-click / scroll / two-hand zoom).
"""

__version__ = "2.0.0"
