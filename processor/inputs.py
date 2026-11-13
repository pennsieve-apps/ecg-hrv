"""
Finding recordings, channels and events under INPUT_DIR, and reading NWB TimeSeries by slice.

The normal input is the `physio` port of physio-import (recordings.csv, channels.csv, events.tsv
and one NWB file per recording) and, optionally, stim-events' `events.tsv`. Files are found by name
and content, recursively and following symlinks, so the ports may arrive in subfolders or merged.
Any NWB 2.x file with ECG TimeSeries in /acquisition also works without the index tables.

NWB is read with h5py only: each TimeSeries' `data` (scaled by `conversion` and `offset`), with
either `starting_time` + `rate` or `timestamps`. Data are read a window at a time, never whole.
"""
from __future__ import annotations

import csv
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger("ecg-hrv")

# conventions.md "Typing" name heuristics, for NWB files without channels.csv
NAME_TYPES = [
    ("ecg", re.compile(r"ECG|EKG|Lead[ _-]?(I|II|III)\b", re.I)),
    ("hr", re.compile(r"^HR$|heart.?rate", re.I)),
    ("bp", re.compile(r"\bBP|ABP|ART|NIBP|Finapres|pressure", re.I)),
    ("resp", re.compile(r"RESP|breath|thorax|abdomen|airflow|pneumo", re.I)),
    ("ppg", re.compile(r"PPG|pleth", re.I)),
    ("stim", re.compile(r"stim|trig|TTL|marker|pulse|StimAmp|CtrlAmp", re.I)),
]
TYPED_NAME = re.compile(r"^(ecg|bp|resp|ppg|nerve|emg|stim|hr|temp|pressure|other)(-\d+)?$")
STIM_TYPES = ("stimulation", "block")
GAP_FACTOR = 1.5          # a timestamp step longer than 1.5 sample periods starts a new segment


@dataclass
class Channel:
    name: str
    type: str
    unit: str = ""
    fs: float = float("nan")
    calibrated: bool = True
    status: str = "good"
    original_name: str = ""
    status_description: str = ""
    derived_from: str = ""      # channels.csv: empty for a raw channel, else its source channel


@dataclass
class Recording:
    rid: str
    path: Path | None
    subject: str = ""
    session: str = ""
    species: str = ""
    channels: list = field(default_factory=list)
    source: str = "nwb"
    error: str | None = None

    def of_type(self, t: str) -> list[Channel]:
        return [c for c in self.channels if c.type == t]


def walk(root: Path):
    for d, dirs, files in os.walk(root, followlinks=True):
        dirs[:] = sorted(x for x in dirs if not x.startswith("."))
        for f in sorted(files):
            if not f.startswith("."):
                yield Path(d) / f


def _truthy(v) -> bool:
    return str(v).strip().lower() not in ("false", "0", "no", "n")


def read_csv(path: Path, delimiter: str = ",") -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(fh, delimiter=delimiter)]


def nwb_species(path: Path) -> str:
    try:
        with h5py.File(path, "r") as f:
            d = f.get("general/subject/species")
            if d is not None:
                v = d[()]
                return v.decode() if isinstance(v, bytes) else str(v)
    except (OSError, KeyError):
        pass
    return ""


def nwb_channels(path: Path) -> list[Channel]:
    """TimeSeries in /acquisition (and one level down), typed by their names."""
    out = []
    with h5py.File(path, "r") as f:
        acq = f.get("acquisition")
        if acq is None:
            return out
        groups = []
        for name, g in acq.items():
            if isinstance(g, h5py.Group) and "data" in g:
                groups.append((name, g))
            elif isinstance(g, h5py.Group):
                groups += [(f"{name}/{n}", s) for n, s in g.items() if isinstance(s, h5py.Group) and "data" in s]
        for name, g in groups:
            leaf = name.split("/")[-1]
            m = TYPED_NAME.match(leaf)
            typ = m.group(1) if m else next((t for t, rx in NAME_TYPES if rx.search(leaf)), "other")
            unit = g["data"].attrs.get("unit", "")
            unit = unit.decode() if isinstance(unit, bytes) else str(unit)
            fs = float("nan")
            if "starting_time" in g:
                fs = float(g["starting_time"].attrs.get("rate", float("nan")))
            elif "timestamps" in g and g["timestamps"].shape[0] > 1:
                ts = g["timestamps"][: min(2001, g["timestamps"].shape[0])]
                fs = 1.0 / float(np.median(np.diff(ts)))
            out.append(Channel(name, typ, unit, fs, calibrated=unit.lower() not in ("", "a.u.", "au"),
                               original_name=leaf))
    return out


def find_recordings(root: Path, species_param: str | None = None) -> list[Recording]:
    """Recordings from every recordings.csv (physio-import), plus NWB files no table lists."""
    files = list(walk(root))
    recs: dict[str, Recording] = {}
    listed: set[Path] = set()
    chan_rows: dict[str, list[dict]] = {}
    for f in files:
        if f.name == "channels.csv":
            for row in read_csv(f):
                if row.get("recording_id") and row.get("channel"):
                    chan_rows.setdefault(row["recording_id"], []).append(row)
    for f in files:
        if f.name != "recordings.csv":
            continue
        for row in read_csv(f):
            rid = row.get("recording_id", "")
            if not rid or rid in recs:
                continue
            rel = row.get("path", "")
            p = (f.parent / rel) if rel else None
            if p is not None and not p.exists():
                p = next((x for x in files if x.name == Path(rel).name), None)
            rec = Recording(rid, p, row.get("subject", ""), row.get("session", ""), row.get("species", ""),
                            source="recordings.csv")
            if p is None:
                rec.error = f"NWB file {rel!r} listed in recordings.csv was not found"
            else:
                listed.add(p.resolve())
            recs[rid] = rec
    for f in files:
        if f.suffix.lower() != ".nwb" or f.resolve() in listed:
            continue
        rid = re.sub(r"[^A-Za-z0-9_-]+", "-", f.stem)
        k, base = 2, rid
        while rid in recs:
            rid, k = f"{base}-{k}", k + 1
        recs[rid] = Recording(rid, f, source="nwb")
    for rid, rec in recs.items():
        if rec.path is None:
            continue
        try:
            found = nwb_channels(rec.path)
        except OSError as e:
            rec.error = f"cannot open NWB file: {e}"
            continue
        if rid in chan_rows:
            by_name = {c.name: c for c in found}
            for row in chan_rows[rid]:
                c = by_name.get(row["channel"])
                if c is None:
                    continue
                c.type = row.get("type") or c.type
                c.unit = row.get("unit") or c.unit
                c.calibrated = _truthy(row.get("calibrated", "true"))
                c.status = row.get("status") or "good"
                c.original_name = row.get("original_name") or c.original_name
                c.derived_from = row.get("derived_from", "") or ""
                c.status_description = row.get("status_description", "") or ""
                try:
                    c.fs = float(row.get("sampling_rate_hz") or c.fs)
                except ValueError:
                    pass
        rec.channels = found
        if not rec.species:
            rec.species = nwb_species(rec.path)
    return list(recs.values())


# --------------------------------------------------------------------------- reading signals

@dataclass
class Segment:
    start: int          # first sample index in the dataset
    stop: int           # one past the last
    t0: float           # time of the first sample (s from recording start)
    fs: float

    @property
    def duration(self) -> float:
        return (self.stop - self.start) / self.fs


class SeriesReader:
    """One NWB TimeSeries: contiguous segments and windowed, scaled reads."""

    def __init__(self, path: Path, name: str):
        self.f = h5py.File(path, "r")
        self.g = self.f["acquisition"][name]
        self.data = self.g["data"]
        a = self.data.attrs
        self.conversion = float(a.get("conversion", 1.0))
        self.offset = float(a.get("offset", 0.0))
        self.n = int(self.data.shape[0])
        self.segments = self._segments()

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _segments(self) -> list[Segment]:
        if self.n == 0:
            return []
        if "starting_time" in self.g:
            st = self.g["starting_time"]
            fs = float(st.attrs["rate"])
            return [Segment(0, self.n, float(st[()]), fs)]
        ts = self.g["timestamps"]
        segs, start, block = [], 0, 5_000_000
        prev_last = None
        bounds = []
        diffs = []
        for i in range(0, self.n, block):
            t = ts[i:i + block].astype(np.float64)
            if prev_last is not None:
                t_full = np.concatenate([[prev_last], t])
                d = np.diff(t_full)
                off = i - 1
            else:
                d = np.diff(t)
                off = i
            diffs.append(np.median(d) if d.size else np.nan)
            prev_last = t[-1]
            bounds.append((off, d))
        period = float(np.nanmedian(diffs)) if diffs else 1.0
        if not math.isfinite(period) or period <= 0:
            period = 1.0
        cuts = []
        for off, d in bounds:
            idx = np.nonzero(d > GAP_FACTOR * period)[0]
            cuts.extend((off + idx + 1).tolist())
        edges = [0] + cuts + [self.n]
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a < 2:
                continue
            seg_t = ts[a:b]
            fs = (b - a - 1) / float(seg_t[-1] - seg_t[0]) if seg_t[-1] > seg_t[0] else 1.0 / period
            segs.append(Segment(a, b, float(seg_t[0]), fs))
        return segs

    def read(self, start: int, stop: int) -> np.ndarray:
        x = self.data[start:stop]
        if x.ndim > 1:
            x = x[:, 0]
        x = x.astype(np.float64)
        if self.conversion != 1.0:
            x *= self.conversion
        if self.offset:
            x += self.offset
        return x

    def times(self, seg: Segment, start: int, stop: int) -> np.ndarray:
        if "timestamps" in self.g:
            return self.g["timestamps"][start:stop].astype(np.float64)
        return seg.t0 + (np.arange(start, stop) - seg.start) / seg.fs


# --------------------------------------------------------------------------- events

@dataclass
class Event:
    rid: str
    event_id: str
    onset: float
    duration: float
    trial_type: str
    params: dict


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def find_events(root: Path) -> tuple[dict[str, list[Event]], list[str]]:
    """Stimulation and block trains from stim-events' events.tsv. physio-import's events.tsv (the
    one beside a recordings.csv: raw markers) is never read, as conventions.md asks; any other
    events.tsv is, with files in a `stim-events/` folder first."""
    files = list(walk(root))
    index_dirs = {f.parent.resolve() for f in files if f.name == "recordings.csv"}
    tsvs = [f for f in files if f.name == "events.tsv" and f.parent.resolve() not in index_dirs]
    tsvs.sort(key=lambda p: 0 if "stim-events" in p.parts else 1)
    out: dict[str, list[Event]] = {}
    used: list[str] = []
    seen_rids: set[str] = set()
    for f in tsvs:
        try:
            rows = read_csv(f, delimiter="\t")
        except (OSError, csv.Error) as e:
            log.warning("cannot read %s: %s", f, e)
            continue
        file_rids = set()
        n = 0
        for i, row in enumerate(rows):
            rid = row.get("recording_id", "")
            tt = row.get("trial_type", "").lower()
            if not rid or tt not in STIM_TYPES or rid in seen_rids:
                continue
            onset, dur = _num(row.get("onset")), _num(row.get("duration"))
            if not math.isfinite(onset):
                continue
            eid = row.get("event_id") or f"{tt[:5]}-{i + 1:04d}"
            params = {k: v for k, v in row.items() if k not in ("recording_id", "onset", "duration", "trial_type",
                                                                   "event_id") and v not in ("", "n/a")}
            out.setdefault(rid, []).append(Event(rid, eid, onset, dur if math.isfinite(dur) else 0.0, tt, params))
            file_rids.add(rid)
            n += 1
        seen_rids |= file_rids
        if n:
            used.append(str(f))
    for evs in out.values():
        evs.sort(key=lambda e: e.onset)
    return out, used
