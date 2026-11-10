"""Read the same camera and bench geometry as Jewel Tracker."""
import json
from pathlib import Path

import jewel_tracker_5 as original


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def tracker_roi():
    return read_json(original.ZONES_FILE).get("roi", [])


def tracker_camera(source):
    if not isinstance(source, int):
        return {}
    saved = read_json(original.CAMERA_FILE).get("usb", {})
    return dict(resolution=saved.get("resolution", "2560x1440"),
                antiflicker={**original.AF_DEFAULT, **saved.get("antiflicker", {})})


def tracker_source():
    saved = read_json(original.CAMERA_FILE)
    mode = saved.get("mode", "usb")
    if mode == "usb":
        return str(saved.get("usb", {}).get("index", 0))
    if mode == "custom":
        return str(saved.get("custom", {}).get("url", "0"))
    camera = saved.get(mode, {})
    return original.dahua_url(camera) if camera.get("ip") else "0"
