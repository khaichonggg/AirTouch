"""Changing fake displays only; never queries or changes a real desktop."""
import runpy

import pytest

from airtouch.config import PROJECT_ROOT, Config
from airtouch.controller import FakeBackend, MouseController

helpers = runpy.run_path(str(PROJECT_ROOT / "tests/test_controller.py"))
POINT, TOUCHING, TOUCHING_RIGHT = (helpers[name] for name in ("POINT", "TOUCHING", "TOUCHING_RIGHT"))
moved = helpers["moved"]


class ChangingDisplay(FakeBackend):
    def __init__(self, size=(1920, 1080), bounds=(0, 0, 1920, 1080), pos=(960, 540)):
        super().__init__(*size)
        self.bounds = bounds
        self.pos = pos
        self.size_queries = self.bounds_queries = 0

    def screen_size(self):
        self.size_queries += 1
        return self.size

    def screen_bounds(self):
        self.bounds_queries += 1
        return self.bounds


@pytest.mark.parametrize("old_bounds,new_bounds,old_pos,new_pos", [
    ((-1280, 0, 3200, 1080), (0, 0, 1920, 1080), (-600, 400), (0, 400)),
    ((0, 0, 1920, 1080), (-1280, 0, 3200, 1080), (960, 540), (-600, 400)),
])
def test_relative_topology_change_reanchors_without_a_move(old_bounds, new_bounds, old_pos, new_pos):
    backend = ChangingDisplay(bounds=old_bounds, pos=old_pos)
    control = MouseController(Config(cursor_mode="relative"), backend)
    control.update("point", POINT, 0)
    backend.bounds, backend.pos = new_bounds, new_pos
    backend.events.clear()

    result = control.update("point", POINT, .3)

    assert result.cursor == new_pos
    assert backend.pos == new_pos
    assert backend.events == []
    control.update("point", moved(POINT, dx=.01), .34)
    assert backend.pos[0] > new_pos[0]
    assert new_bounds[0] <= backend.pos[0] < new_bounds[0] + new_bounds[2]


def test_relative_primary_resize_updates_geometry_without_retargeting_still_hand():
    backend = ChangingDisplay()
    control = MouseController(Config(cursor_mode="relative"), backend)
    control.update("point", POINT, 0)
    backend.size, backend.bounds, backend.pos = (1280, 1024), (0, 0, 1280, 1024), (500, 500)
    backend.events.clear()

    result = control.update("point", POINT, .3)

    assert result.cursor == (500, 500)
    assert backend.events == []
    # 4:3 camera -> 5:4 primary: a 0.7-high active region is 0.65625 wide.
    assert control.active_box == pytest.approx((.171875, .15, .65625, .7))
    control.update("point", moved(POINT, dx=.01), .34)
    assert 500 < backend.pos[0] < 550


def test_absolute_mapping_uses_new_primary_size():
    backend = ChangingDisplay()
    control = MouseController(Config(), backend)
    control.update("point", POINT, 0)
    backend.size, backend.bounds = (1280, 1024), (0, 0, 1280, 1024)

    control.update("point", moved(POINT, dx=2, dy=2), .3)

    assert backend.pos == (1279, 1023)
    assert backend.count("down") == 0


@pytest.mark.parametrize("pose,button", [(TOUCHING, "left"), (TOUCHING_RIGHT, "right")])
def test_topology_change_preserves_held_drag_without_duplicate_press_or_jump(pose, button):
    backend = ChangingDisplay(bounds=(-1280, 0, 3200, 1080), pos=(-600, 400))
    control = MouseController(Config(cursor_mode="relative"), backend)
    control.update("pinch_index" if button == "left" else "pinch_middle", pose, 0)
    control.update("pinch_index" if button == "left" else "pinch_middle", pose, .3)
    assert backend.down == {button}
    backend.bounds, backend.pos = (0, 0, 1920, 1080), (300, 400)
    backend.events.clear()

    result = control.update("pinch_index" if button == "left" else "pinch_middle", pose, .6)

    assert result.cursor == (300, 400)
    assert result.button == button and backend.down == {button}
    assert backend.events == []
    control.update("point", moved(pose, dx=.01), .64)
    assert backend.pos[0] > 300
    assert backend.count("down") == backend.count("up") == 0


@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_display_remap_cannot_start_a_new_click(mode):
    backend = ChangingDisplay()
    control = MouseController(Config(cursor_mode=mode), backend)
    control.update("point", POINT, 0)
    backend.size, backend.bounds = (1280, 1024), (0, 0, 1280, 1024)
    backend.events.clear()

    result = control.update("pinch_index", TOUCHING, .3)

    assert result.button is None
    assert backend.count("down") == 0


def test_display_remap_restarts_pending_dwell_instead_of_clicking():
    backend = ChangingDisplay()
    control = MouseController(Config(cursor_mode="relative", dwell_enabled=True, dwell_ms=200), backend)
    control.update("point", POINT, 0)
    backend.size, backend.bounds = (1280, 1024), (0, 0, 1280, 1024)
    backend.events.clear()
    control.update("point", POINT, .3)
    control.update("point", POINT, .34)
    control.update("point", POINT, .45)
    assert backend.count("down") == 0
    control.update("point", POINT, .56)
    assert backend.count("down", "left") == 1


@pytest.mark.parametrize("size,bounds", [
    ((0, 1080), (0, 0, 1920, 1080)),
    ((1920, 0), (0, 0, 1920, 1080)),
    ((1920, 1080), (0, 0, 0, 1080)),
    ((1920, 1080), (0, 0, 1920, -1)),
])
def test_unavailable_geometry_stops_and_releases_then_allows_safe_recovery(size, bounds):
    backend = ChangingDisplay()
    control = MouseController(Config(cursor_mode="relative"), backend)
    control.update("pinch_index", TOUCHING, 0)
    control.update("pinch_index", TOUCHING, .3)
    assert backend.down == {"left"}
    backend.size, backend.bounds = size, bounds
    backend.events.clear()

    with pytest.raises(RuntimeError):
        control.update("pinch_index", TOUCHING, .6)

    assert backend.down == set()
    assert backend.events == [("up", "left")]
    backend.size, backend.bounds, backend.pos = (1920, 1080), (0, 0, 1920, 1080), (300, 400)
    backend.events.clear()
    returned = control.update("point", POINT, .64)
    assert returned.cursor == (300, 400)
    assert backend.events == []


def test_display_queries_are_bounded_during_normal_movement():
    backend = ChangingDisplay()
    control = MouseController(Config(cursor_mode="relative"), backend)
    for frame in range(60):
        control.update("point", POINT, frame / 30)
    assert backend.size_queries <= 12
    assert backend.bounds_queries <= 12


def test_releasing_without_a_hand_does_not_require_available_display_geometry():
    backend = ChangingDisplay()
    control = MouseController(Config(cursor_mode="relative"), backend)
    control.update("pinch_index", TOUCHING, 0)
    control.update("pinch_index", TOUCHING, .3)
    backend.size, backend.bounds = (0, 0), (0, 0, 0, 0)
    backend.events.clear()

    result = control.update("none", None, .6)

    assert result.button is None
    assert backend.down == set()
    assert backend.events == [("up", "left")]


@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_new_press_checks_changed_geometry_inside_regular_poll_window(mode):
    backend = ChangingDisplay()
    control = MouseController(Config(cursor_mode=mode), backend)
    control.update("point", POINT, 0)
    control.update("point", POINT, .26)
    backend.size, backend.bounds, backend.pos = (1280, 1024), (0, 0, 1280, 1024), (500, 500)
    backend.events.clear()

    result = control.update("point", TOUCHING, .30)

    assert result.button is None
    assert backend.count("down") == 0


def test_due_dwell_checks_changed_geometry_inside_regular_poll_window():
    backend = ChangingDisplay()
    control = MouseController(Config(cursor_mode="relative", dwell_enabled=True, dwell_ms=300), backend)
    control.update("point", POINT, 0)
    control.update("point", POINT, .26)
    backend.size, backend.bounds, backend.pos = (1280, 1024), (0, 0, 1280, 1024), (500, 500)
    backend.events.clear()

    control.update("point", POINT, .31)

    assert backend.events == []
    control.update("point", POINT, .33)
    control.update("point", POINT, .64)
    assert backend.count("down", "left") == 1


def test_new_stroke_refreshes_bounds_inside_regular_poll_window():
    backend = ChangingDisplay(bounds=(-1280, 0, 3200, 1080), pos=(-600, 400))
    control = MouseController(Config(cursor_mode="relative"), backend)
    control.update("point", POINT, 0)
    control.update("fist", POINT, .03)
    backend.bounds, backend.pos = (0, 0, 1920, 1080), (0, 400)
    backend.events.clear()
    control.update("point", POINT, .10)
    control.update("point", moved(POINT, dx=-.01), .13)
    assert backend.pos == (0, 400)
    assert not any(event[0] == "move" and event[1] < 0 for event in backend.events)


def test_pending_tap_rechecks_display_before_clicking_at_an_old_target():
    backend = ChangingDisplay()
    control = MouseController(Config(cursor_mode="relative"), backend)
    control.update("pinch_index", TOUCHING, 0)
    assert backend.down == set()
    backend.bounds, backend.pos = (-1280, 0, 3200, 1080), (-600, 400)
    backend.events.clear()

    control.update("point", POINT, .1)
    control.update("point", POINT, .13)

    assert backend.count("down") == backend.count("up") == 0
    assert backend.pos == (-600, 400)


def test_pending_drag_rechecks_display_when_hold_expires_inside_poll_window():
    backend = ChangingDisplay()
    control = MouseController(Config(cursor_mode="relative"), backend)
    control.update("point", POINT, 0)
    control.update("pinch_index", TOUCHING, .1)
    control.update("pinch_index", TOUCHING, .3)
    assert backend.down == set()
    backend.bounds, backend.pos = (-1280, 0, 3200, 1080), (-600, 400)
    backend.events.clear()

    control.update("pinch_index", TOUCHING, .36)

    assert backend.count("down") == 0
    assert backend.pos == (-600, 400)
