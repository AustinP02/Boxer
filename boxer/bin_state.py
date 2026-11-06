"""Persistent bin ownership and a write-ahead placement journal.

No new placement may start after an interrupted attempt until its physical outcome
is resolved. A bin is committed to a color only after confirmed release.
"""

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .settings import BIN_IDS, COLORS, SOURCE_IDS, SetupError
from .storage import atomic_json


def workcell_key(settings):
    # Moving a source/camera does not change bin contents; moving/relabeling output bins does.
    identity = {"address": settings.data["address"], "bins": settings.data["bins"]}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class BinState:
    def __init__(self, settings, path, data):
        self.path = Path(path) if path is not None else None
        self.data = data
        if not isinstance(data, dict):
            raise SetupError("Invalid bin-state document")
        if data.get("schema_version") != 1 or data.get("workcell_key") != workcell_key(settings):
            raise SetupError(
                "Bin state belongs to another output-bin layout; inspect contents before reinitializing"
            )
        if set(data.get("bins", {})) != set(BIN_IDS):
            raise SetupError("Bin state has invalid bin IDs")
        assigned = []
        for bin_id, entry in data["bins"].items():
            if not isinstance(entry, dict) or entry.get("color") not in (*COLORS, None):
                raise SetupError(f"Invalid color assignment in {bin_id}")
            count = entry.get("released_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise SetupError(f"Invalid release count in {bin_id}")
            if (entry["color"] is None) != (count == 0):
                raise SetupError(f"Inconsistent color and release count in {bin_id}")
            if entry["color"] is not None:
                assigned.append(entry["color"])
        if len(assigned) != len(set(assigned)):
            raise SetupError("A color has been assigned to multiple bins")
        pending = data.get("pending")
        if pending is not None:
            if not isinstance(pending, dict) or (
                pending.get("bin_id") not in BIN_IDS
                or pending.get("color") not in COLORS
                or pending.get("source_id") not in SOURCE_IDS
                or pending.get("stage") not in ("reserved", "grasped", "release_pending")
            ):
                raise SetupError("Invalid pending placement journal")
            if data["bins"][pending["bin_id"]]["color"] not in (None, pending["color"]):
                raise SetupError("Pending placement conflicts with bin ownership")
            if any(
                b != pending["bin_id"] and e["color"] == pending["color"] for b, e in data["bins"].items()
            ):
                raise SetupError("Pending placement would duplicate another bin's color")

    @classmethod
    def empty(cls, settings, path=None):
        return cls(
            settings,
            path,
            {
                "schema_version": 1,
                "workcell_key": workcell_key(settings),
                "bins": {b: {"color": None, "released_count": 0} for b in BIN_IDS},
                "pending": None,
            },
        )

    @classmethod
    def load(cls, settings, path):
        path = Path(path)
        if not path.exists():
            raise SetupError(
                "Bin state is uninitialized. Inspect that all four bins are empty, then run bins init --empty-confirmed"
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            raise SetupError(
                f"Cannot read bin state; inspect the state file: {type(error).__name__}"
            ) from error
        return cls(settings, path, data)

    def save(self):
        if self.path is None:
            raise SetupError("An in-memory dry-run state cannot be used for live sorting")
        self.data["updated_at"] = datetime.now(UTC).isoformat()
        atomic_json(self.path, self.data)

    def _commit(self, updated):
        previous = self.data
        self.data = updated
        try:
            self.save()
        except (OSError, TypeError, ValueError):
            self.data = previous
            raise

    def require_resolved(self):
        if self.data["pending"] is not None:
            p = self.data["pending"]
            raise SetupError(
                f"Unresolved {p['color']} placement into {p['bin_id']} ({p['stage']}); inspect the robot/bins and use bins resolve"
            )

    def available_bins(self, color):
        self.require_resolved()
        if color not in COLORS:
            raise SetupError("Unsupported color")
        matching = [b for b in BIN_IDS if self.data["bins"][b]["color"] == color]
        return matching or [b for b in BIN_IDS if self.data["bins"][b]["color"] is None]

    def begin(self, color, bin_id, source_id):
        if bin_id not in self.available_bins(color):
            raise SetupError(f"Cannot place {color} into {bin_id}")
        if source_id not in SOURCE_IDS:
            raise SetupError("Unknown source")
        updated = copy.deepcopy(self.data)
        updated["pending"] = {"color": color, "bin_id": bin_id, "source_id": source_id, "stage": "reserved"}
        self._commit(updated)

    def stage(self, value):
        p = self.data["pending"]
        if p is None or value not in ("grasped", "release_pending"):
            raise SetupError("No valid placement in progress")
        updated = copy.deepcopy(self.data)
        updated["pending"]["stage"] = value
        self._commit(updated)

    def confirm_release(self):
        p = self.data["pending"]
        if p is None or p["stage"] != "release_pending":
            raise SetupError("No release awaiting confirmation")
        self.resolve(released=True)

    def resolve(self, released):
        p = self.data["pending"]
        if p is None:
            raise SetupError("No interrupted placement to resolve")
        updated = copy.deepcopy(self.data)
        if released:
            entry = updated["bins"][p["bin_id"]]
            if entry["color"] not in (None, p["color"]):
                raise SetupError("Release conflicts with existing bin color")
            entry["color"] = p["color"]
            entry["released_count"] += 1
        updated["pending"] = None
        self._commit(updated)

    def summary(self):
        return copy.deepcopy(self.data)
