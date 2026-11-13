"""
ecg-hrv: R-peaks, beat labels, heart rate and heart rate variability from ECG, with species presets.

  INPUT_DIR   physio-import's `physio` port (recordings.csv, channels.csv, NWB per recording) or any
              NWB files with ECG; optionally stim-events' events.tsv. Found recursively.
  OUTPUT_DIR/ecg-hrv/
      beats.csv  series.csv  hrv.csv  qc.csv  features.csv  figures/*.png  summary.json  provenance.json

One recording at a time, the ECG read a window at a time. A recording without ECG is skipped (not a
failure); a recording that fails is logged and skipped.
Exit codes: 0 ok, 1 no usable input / every recording with ECG failed, 2 bad parameters.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepare_environment() -> None:
    """Lambda-safe start: HOME and TMPDIR may be fresh, empty folders and everything else
    read-only. Writable caches go under TMPDIR; nothing is written to the image."""
    tmp = Path(os.environ.get("TMPDIR") or "/tmp")
    for var, default in (("TMPDIR", tmp), ("HOME", tmp / "home")):
        p = Path(os.environ.get(var) or default)
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        os.environ[var] = str(p)
    mpl = Path(os.environ["TMPDIR"]) / "mpl"
    baked = Path(os.environ.get("ECG_HRV_MPL_CACHE", "/opt/ecg-hrv/matplotlib"))
    if not os.environ.get("MPLCONFIGDIR") or not os.access(os.environ["MPLCONFIGDIR"], os.W_OK):
        mpl.mkdir(parents=True, exist_ok=True)
        if baked.is_dir():  # the font cache built at image build time: copy instead of rebuilding
            for f in baked.glob("fontlist-*.json"):
                target = mpl / f.name
                if not target.exists():
                    target.write_bytes(f.read_bytes())
        os.environ["MPLCONFIGDIR"] = str(mpl)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(os.environ["TMPDIR"]) / "numba"))


_prepare_environment()

from processor import resources as R  # noqa: E402  (no numpy import)

_requested = os.environ.get("THREADS", "0").strip() or "0"
try:
    _THREADS, _THREADS_WHY = R.resolve_workers(int(_requested), per_worker_mb=300, reserve_mb=500)
except ValueError:
    _THREADS, _THREADS_WHY = 1, f"THREADS={_requested!r} is not an integer"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(_THREADS)

import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from importlib import metadata as importlib_metadata  # noqa: E402

import numpy as np  # noqa: E402

from processor import ecg as E  # noqa: E402
from processor import figures as F  # noqa: E402
from processor import inputs as I  # noqa: E402
from processor import presets as P  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
log = logging.getLogger("ecg-hrv")
log.setLevel(logging.INFO)
for _noisy in ("matplotlib", "h5py", "numba"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

APP_NAME = "ecg-hrv"
REPO_ROOT = Path(__file__).resolve().parent.parent
TRUE, FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")

CITATIONS = ["10.3758/s13428-020-01516-y", "10.1161/01.CIR.93.5.1043", "10.1080/03091902.2019.1640306",
             "10.13026/9njx-6322", "10.1161/01.CIR.101.23.e215", "10.3389/fphys.2018.01390"]
SPECIES_CITATIONS = {"human": ["10.1161/01.CIR.93.5.1043"], "dog": ["10.3389/fphys.2018.01390"],
                     "rabbit": ["10.3389/fphys.2018.01390"],
                     "mouse": ["10.3389/fphys.2018.01390", "10.3389/fphys.2018.01001", "10.1113/expphysiol.2007.040733"],
                     "rat": ["10.3389/fphys.2023.1170320"],
                     "pig": ["10.1016/j.physbeh.2011.01.019", "10.1371/journal.pone.0134387"]}
DETECTOR_CITATIONS = {"pantompkins1985": ["10.1109/TBME.1985.325532"]}

DEFAULTS = {
    "SPECIES": "auto", "DETECTOR": "neurokit", "ECG_CHANNEL": "auto", "INVERT": "auto",
    "ARTIFACT_CORRECTION": "kubios", "ARTIFACT_MIN_PCT": "10", "SECOND_DETECTOR": "true", "FILTER_BAND": "auto", "REFRACTORY_S": "0",
    "RR_RANGE_S": "auto", "VLF_BAND": "auto", "LF_BAND": "auto", "HF_BAND": "auto", "PNN_THRESHOLD_MS": "0",
    "SERIES_HZ": "0", "MAX_GAP_S": "0", "BASELINE_S": "0", "AFTER_S": "0", "POWERLINE_HZ": "auto",
    "FIGURES": "true", "CHUNK_S": "300", "THREADS": "0",
}

HEADERS = {
    "beats": ["recording_id", "beat", "time_s", "rr_ms", "label", "detector"],
    "series": ["recording_id", "time_s", "variable", "value", "valid"],
    "hrv": ["recording_id", "scope", "metric", "value", "n_beats", "window_s"],
    "qc": ["recording_id", "detector", "n_beats", "pct_ectopic", "pct_artifact", "agreement_with_second_detector",
           "signal_quality", "channel", "sampling_rate_hz", "species", "inverted", "mean_hr_bpm",
           "agreement_other_lead", "other_lead", "hr_device_mae_bpm", "hr_device_r", "xqrs_offset_ms",
           "xqrs_timing_jitter_ms", "agreement_tight", "drr_lag1_autocorr", "pct_lipponen_flagged_ungated", "notes"],
    "features": ["recording_id", "channel", "feature", "value"],
}


class ParamError(ValueError):
    pass


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("neurokit2", "wfdb", "numpy", "scipy", "pandas", "h5py", "matplotlib"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def _bool(name: str, v: str) -> bool:
    s = str(v).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    raise ParamError(f"{name} must be true or false, got {v!r}")


def _float(name: str, v: str, lo: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ParamError(f"{name} must be a number, got {v!r}") from None
    if not math.isfinite(x) or x < lo:
        raise ParamError(f"{name} must be >= {lo:g}, got {v!r}")
    return x


def _range(name: str, v: str) -> tuple | None:
    s = str(v).strip().lower()
    if s in ("", "auto"):
        return None
    try:
        a, b = (float(x) for x in s.replace(",", "-", 1).split("-", 1))
    except ValueError:
        raise ParamError(f"{name} must be auto or two numbers like 0.04-0.15, got {v!r}") from None
    if not (0 <= a < b):
        raise ParamError(f"{name}: the lower edge must be below the upper edge, got {v!r}")
    return (a, b)


def read_params(env=os.environ) -> dict:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v).strip() for k, v in DEFAULTS.items()}
    p = {"raw": raw}
    sp = raw["SPECIES"].lower()
    if sp != "auto" and sp not in P.PRESETS:
        raise ParamError(f"SPECIES must be auto or one of {', '.join(P.SPECIES)}, got {raw['SPECIES']!r}")
    p["species"] = sp
    det = raw["DETECTOR"].lower()
    if det not in E.DETECTORS:
        raise ParamError(f"DETECTOR must be one of {', '.join(E.DETECTORS)}, got {raw['DETECTOR']!r}")
    p["detector"] = det
    p["ecg_channel"] = raw["ECG_CHANNEL"]
    inv = raw["INVERT"].lower()
    if inv not in ("auto", "true", "false"):
        raise ParamError("INVERT must be auto, true or false")
    p["invert"] = inv
    art = raw["ARTIFACT_CORRECTION"].lower()
    if art not in ("kubios", "range", "none"):
        raise ParamError("ARTIFACT_CORRECTION must be kubios, range or none")
    p["artifact"] = art
    p["artifact_min_pct"] = _float("ARTIFACT_MIN_PCT", raw["ARTIFACT_MIN_PCT"])
    p["second"] = _bool("SECOND_DETECTOR", raw["SECOND_DETECTOR"])
    p["figures"] = _bool("FIGURES", raw["FIGURES"])
    p["filter_band"] = _range("FILTER_BAND", raw["FILTER_BAND"])
    p["rr_range"] = _range("RR_RANGE_S", raw["RR_RANGE_S"])
    p["bands"] = {k: _range(f"{k.upper()}_BAND", raw[f"{k.upper()}_BAND"]) for k in ("vlf", "lf", "hf")}
    pl = raw["POWERLINE_HZ"].lower()
    if pl == "auto":
        p["powerline_hz"] = -1.0
    elif pl in ("0", "off", "none", "50", "60"):
        p["powerline_hz"] = 0.0 if pl in ("0", "off", "none") else float(pl)
    else:
        raise ParamError(f"POWERLINE_HZ must be auto, 0, 50 or 60, got {raw['POWERLINE_HZ']!r}")
    for k in ("REFRACTORY_S", "PNN_THRESHOLD_MS", "SERIES_HZ", "MAX_GAP_S", "BASELINE_S", "AFTER_S"):
        p[k.lower()] = _float(k, raw[k])
    p["chunk_s"] = _float("CHUNK_S", raw["CHUNK_S"], lo=10.0)
    try:
        int(raw["THREADS"])
    except ValueError:
        raise ParamError("THREADS must be an integer") from None
    return p


def settings_for(preset: P.Preset, p: dict) -> tuple[E.Settings, dict]:
    """The preset with the user's overrides; also returns the resolved numbers for provenance."""
    rr = p["rr_range"] or (preset.rr_min_s, preset.rr_max_s)
    bands = {k: p["bands"][k] or getattr(preset, f"{k}_hz") for k in ("vlf", "lf", "hf")}
    s = E.Settings(preset=preset, detector=p["detector"], filter_band=p["filter_band"] or preset.filter_band_hz,
                   refractory_s=p["refractory_s"] or preset.refractory_s, rr_min_s=rr[0], rr_max_s=rr[1],
                   bands=bands, pnn_ms=p["pnn_threshold_ms"] or preset.pnn_ms, invert=p["invert"],
                   second_detector=p["second"], chunk_s=p["chunk_s"], powerline_hz=p["powerline_hz"],
                   artifact_method=p["artifact"], artifact_min_pct=p["artifact_min_pct"])
    extra = {"series_hz": p["series_hz"] or preset.series_hz, "max_gap_s": p["max_gap_s"] or preset.max_gap_s,
             "baseline_s": p["baseline_s"] or preset.baseline_s, "after_s": p["after_s"] or preset.after_s}
    used = {"filter_band_hz": list(s.filter_band), "refractory_s": s.refractory_s, "rr_range_s": [rr[0], rr[1]],
            "vlf_hz": list(bands["vlf"]), "lf_hz": list(bands["lf"]), "hf_hz": list(bands["hf"]),
            "pnn_threshold_ms": s.pnn_ms, "spectrum_resampling_hz": E.interp_rate(s), **extra,
            "min_window_s": {"hf": round(P.min_window_s(bands["hf"], "hf"), 1),
                             "lf": round(P.min_window_s(bands["lf"], "lf"), 1),
                             "vlf": round(P.min_window_s(bands["vlf"], "vlf"), 1)}}
    return s, {**extra, "used": used}


def fmt(x, nd: int = 6) -> str:
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "true" if x else "false"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "" if not math.isfinite(xf) else f"{xf:.{nd}g}"


def num(x, nd: int = 4):
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return round(xf, nd) if math.isfinite(xf) else None


class Writer:
    """CSV tables, opened once, appended a recording at a time."""

    def __init__(self, out: Path):
        self.files, self.w = {}, {}
        for name, header in HEADERS.items():
            fh = open(out / f"{name}.csv", "w", newline="", encoding="utf-8")
            self.files[name] = fh
            self.w[name] = csv.writer(fh, lineterminator="\n")
            self.w[name].writerow(header)

    def rows(self, name: str, rows):
        self.w[name].writerows(rows)

    def close(self):
        for fh in self.files.values():
            fh.close()


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def _independent(c: I.Channel, chosen: I.Channel) -> bool:
    """A second lead only when both are raw and neither is a copy of the other."""
    return (c is not chosen and not c.derived_from and not chosen.derived_from
            and c.status not in ("flat", "saturated"))


def choose_ecg(rec: I.Recording, wanted: str) -> tuple[I.Channel | None, list[I.Channel], str, str]:
    """(channel, independent second leads, reason when none, note). Raw channels are preferred over
    processed copies (channels.csv derived_from), even when the raw one is noisy; a copy is used
    only when no usable raw ECG exists, and never as a second lead."""
    ecgs = rec.of_type("ecg")
    if wanted and wanted.lower() != "auto":
        named = [c for c in rec.channels if wanted in (c.name, c.original_name)]
        if not named:
            return None, [], f"ECG_CHANNEL {wanted!r} not in this recording", ""
        ch = named[0]
        note = f"ECG_CHANNEL={wanted} is derived from {ch.derived_from}" if ch.derived_from else ""
        return ch, [c for c in ecgs if _independent(c, ch)], "", note
    if not ecgs:
        return None, [], "no ECG channel", ""
    usable = [c for c in ecgs if c.status not in ("flat", "saturated")]
    if not usable:
        return None, [], "every ECG channel is flat or saturated (channels.csv status)", ""
    rank = {"good": 0, "noisy": 1}
    raw = sorted((c for c in usable if not c.derived_from), key=lambda c: (rank.get(c.status, 2), c.name))
    if raw:
        ch = raw[0]
        note = ""
        if ch.status != "good":
            note = f"raw ECG {ch.name} is {ch.status} ({ch.status_description or 'channels.csv'}): used, as raw is preferred over processed copies"
        return ch, [c for c in raw[1:] if _independent(c, ch)], "", note
    derived = sorted(usable, key=lambda c: (rank.get(c.status, 2), c.name))
    ch = derived[0]
    return ch, [], "", f"no usable raw ECG: used {ch.name}, a processed copy (derived_from {ch.derived_from})"


def event_windows(events: list, baseline_s: float, after_s: float, t_end: float) -> list[tuple]:
    """(scope, start, end) per event phase; baseline and after are cut at the neighbouring trains."""
    out = []
    for i, ev in enumerate(events):
        end = ev.onset + ev.duration
        prev_end = events[i - 1].onset + events[i - 1].duration if i > 0 else -math.inf
        next_on = events[i + 1].onset if i + 1 < len(events) else math.inf
        b0 = max(ev.onset - baseline_s, prev_end, 0.0)
        if ev.onset - b0 > 0:
            out.append((f"{ev.event_id}:baseline", b0, ev.onset))
        if ev.duration > 0:
            out.append((f"{ev.event_id}:during", ev.onset, end))
            a1 = min(end + after_s, next_on, t_end)
            if a1 > end:
                out.append((f"{ev.event_id}:after", end, a1))
    return out


def process(rec: I.Recording, p: dict, events: list, out: Path, writer: Writer, figs: list, warnings_: list) -> dict:
    t_start = time.time()
    sp_from = "SPECIES"
    species = p["species"] if p["species"] != "auto" else None
    if species is None:
        species = P.resolve(rec.species)
        sp_from = "recordings.csv / NWB Subject.species" if species else "default (human)"
        if species is None:
            species = "human"
            if rec.species:
                warnings_.append(f"{rec.rid}: species {rec.species!r} has no preset; used human (set SPECIES)")
    preset = P.PRESETS[species]
    s, extra = settings_for(preset, p)
    ch, others, why, ch_note = choose_ecg(rec, p["ecg_channel"])
    if ch is None:
        log.info("%s: skipped (%s); channel types: %s", rec.rid, why,
                 "|".join(sorted({c.type for c in rec.channels})) or "none")
        return {"status": "skipped", "reason": why, "species": species, "seconds": round(time.time() - t_start, 2)}
    notes = [ch_note] if ch_note else []
    if not ch.calibrated:
        notes.append("ECG uncalibrated: timing only, no amplitudes")
    if s.detector != "neurokit" and species not in ("human",):
        notes.append(f"{s.detector} keeps NeuroKit2's human constants; only the refractory period follows the preset")
    with I.SeriesReader(rec.path, ch.name) as rd:
        if not rd.segments:
            return {"status": "skipped", "reason": f"ECG channel {ch.name} is empty", "species": species,
                    "seconds": round(time.time() - t_start, 2)}
        total = sum(g.stop - g.start for g in rd.segments)
        last = [time.time()]

        def progress(b, seg):
            if time.time() - last[0] > 60:
                last[0] = time.time()
                log.info("%s: %.0f %% of %s", rec.rid, 100.0 * (b - rd.segments[0].start) / max(total, 1), ch.name)

        det = E.detect(rd, s, progress)
    if det.mains_hz:
        notes.append(f"{det.mains_hz:g} Hz mains interference notched")
    beats = E.label_beats(det, s)
    n = int(beats.times.size)
    rid = rec.rid
    if n < 3:
        raise RuntimeError(f"only {n} R-peaks found on {ch.name}: is it an ECG?")
    t_end = max(g.t0 + (g.stop - g.start - 1) / g.fs for g in det.segments)
    t0 = min(g.t0 for g in det.segments)
    recorded = sum(g.duration for g in det.segments)

    # beats.csv
    writer.rows("beats", [(rid, i + 1, fmt(t, 9), fmt(r, 6), lab, s.detector)
                          for i, (t, r, lab) in enumerate(zip(beats.times, beats.rr_ms, beats.labels))])
    # hrv.csv: whole recording, then each stimulation phase
    reasons: Counter = Counter()
    hrv_rows = []
    vals, why, nb = E.hrv_window(beats.nn_ms, beats.nn_t, beats.nn_start, t0, t_end, s)
    win_rec = recorded
    if len(det.segments) > 1:
        notes.append(f"{len(det.segments)} segments: recording-level spectra interpolate across the gaps "
                     f"between segments; compare per-stimulation windows instead")
    rec_vals = vals
    for m, v in vals.items():
        hrv_rows.append((rid, "recording", m, fmt(v), nb, fmt(win_rec, 6)))
        if v is None:
            notes.append(f"recording {m}: {why[m]}")
    windows = event_windows(events, extra["baseline_s"], extra["after_s"], t_end)
    for scope, a, b in windows:
        v2, w2, nb2 = E.hrv_window(beats.nn_ms, beats.nn_t, beats.nn_start, a, b, s)
        for m, v in v2.items():
            hrv_rows.append((rid, scope, m, fmt(v), nb2, fmt(b - a, 6)))
            if v is None:
                reasons[(m, w2[m].split(" (")[0] if "window" in w2[m] else w2[m])] += 1
    writer.rows("hrv", hrv_rows)
    if windows:
        by_metric: dict = {}
        for (m, r), c in reasons.items():
            by_metric.setdefault(r, []).append(f"{m} {c}")
        for r, ms in by_metric.items():
            notes.append(f"per-stimulation windows left empty ({r}): " + ", ".join(ms))

    # series.csv
    srows = E.grid_series(beats, det.segments, extra["series_hz"], extra["max_gap_s"])
    dev = None
    dev_mae = dev_r = None
    hr_ch = [c for c in rec.of_type("hr") if c.status != "flat"]
    if hr_ch:
        with I.SeriesReader(rec.path, hr_ch[0].name) as rd2:
            dev = E.device_hr(rd2, extra["series_hz"])
        if dev[0].size:
            srows += [(t, "hr_device_bpm", v, bool(np.isfinite(v))) for t, v in zip(*dev)]
            ecg_hr = {round(t * extra["series_hz"]): v for t, var, v, ok in srows if var == "hr_bpm" and ok}
            pairs = [(ecg_hr[k], v) for t, v in zip(*dev) if (k := round(t * extra["series_hz"])) in ecg_hr
                     and np.isfinite(v)]
            if len(pairs) >= 10:
                a_, b_ = np.array(pairs).T
                dev_mae = float(np.mean(np.abs(a_ - b_)))
                dev_r = float(np.corrcoef(a_, b_)[0, 1]) if np.std(a_) > 0 and np.std(b_) > 0 else None
    writer.rows("series", [(rid, fmt(t, 9), var, fmt(v), "true" if ok else "false") for t, var, v, ok in srows])

    # QC
    pct_ect = 100.0 * float(np.mean(beats.labels == "ectopic"))
    pct_art = 100.0 * float(np.mean(beats.labels == "artifact"))
    agree = E.agreement(det.times, det.xqrs_times, s.tolerance_s) if det.xqrs_times is not None else None
    other_agree = None
    if others:
        s2 = E.Settings(**{**s.__dict__, "second_detector": False})
        with I.SeriesReader(rec.path, others[0].name) as rd3:
            if rd3.segments:
                det2 = E.detect(rd3, s2)
                other_agree = E.agreement(det.times, det2.times, s.tolerance_s)
    xq_off = xq_jit = agree_tight = None
    if det.xqrs_times is not None:
        o, j = E.timing_jitter(det.times, det.xqrs_times, s.tolerance_s)
        if np.isfinite(o):
            xq_off, xq_jit = 1000 * o, 1000 * j
            agree_tight = E.agreement(det.times, det.xqrs_times - o, min(0.01, 0.1 * s.rr_min_s))
    ac1 = E.drr_autocorr(beats)
    if ac1 is not None and ac1 < -0.4:
        warnings_.append(f"{rid}: successive RR changes alternate (lag-1 autocorrelation {ac1:.2f}); beat timing "
                         f"may jump between parts of the QRS: check the ECG figure")
    ungated = 100.0 * beats.counts["lipponen_flagged_ungated"] / max(n, 1)
    if agree is not None and agree < 0.9:
        warnings_.append(f"{rid}: the two R-peak detectors agree on only {100 * agree:.0f} % of beats; check the ECG figure")
    if pct_art > 10:
        warnings_.append(f"{rid}: {pct_art:.0f} % of beats labelled artifact")
    mean_hr = rec_vals.get("mean_hr")
    if mean_hr is not None and not (60.0 / s.rr_max_s * 1.2 <= mean_hr <= 60.0 / s.rr_min_s / 1.2):
        warnings_.append(f"{rid}: mean heart rate {mean_hr:.0f} bpm is near the edge of the {species} preset's range; "
                         f"check SPECIES")
    writer.rows("qc", [(rid, s.detector, n, fmt(pct_ect, 4), fmt(pct_art, 4), fmt(agree, 4), fmt(det.quality, 4),
                        ch.name, fmt(det.fs, 6), species, "true" if det.inverted else "false", fmt(mean_hr, 5),
                        fmt(other_agree, 4), others[0].name if (others and other_agree is not None) else "",
                        fmt(dev_mae, 4), fmt(dev_r, 4), fmt(xq_off, 4), fmt(xq_jit, 4), fmt(agree_tight, 4),
                        fmt(ac1, 3), fmt(ungated, 4), "; ".join(n_ for n_ in notes if n_))])
    # features.csv
    feats = [(rid, "", f"hrv_{m}", fmt(v)) for m, v in rec_vals.items() if v is not None]
    feats += [(rid, ch.name, "ecg_n_beats", n), (rid, ch.name, "ecg_pct_ectopic", fmt(pct_ect, 4)),
              (rid, ch.name, "ecg_pct_artifact", fmt(pct_art, 4))]
    if agree is not None:
        feats.append((rid, ch.name, "ecg_detector_agreement", fmt(agree, 4)))
    writer.rows("features", feats)

    # figures
    if p["figures"]:
        fd = out / "figures"
        fd.mkdir(exist_ok=True)
        f1 = fd / f"{rid}_heart-rate.png"
        F.tachogram(f1, rid, det, beats, events, dev, s, species, ch.calibrated)
        figs.append({"path": f"figures/{f1.name}", "caption": (
            f"{rid}: heart rate from the ECG ({ch.name}) over the whole recording, beat by beat (blue); red crosses "
            f"are beats labelled ectopic or artifact and left out of HRV"
            + ("; grey is the acquisition device's own heart-rate channel, which should overlie the blue line"
               if dev is not None and dev[0].size else "")
            + ("; orange bands are stimulation trains" if events else "")
            + ". Below: the first 10 s of ECG with the detected R-peaks; every QRS complex should carry one circle."
            + (" The lead was inverted before detection." if det.inverted else ""))})
        f2 = fd / f"{rid}_hrv.png"
        if F.hrv_figure(f2, rid, beats, s, E.interp_rate(s)):
            figs.append({"path": f"figures/{f2.name}", "caption": (
                f"{rid}: left, Poincare plot of successive NN intervals: the width of the cloud across the "
                f"diagonal reflects beat-to-beat (vagal) variability (SD1), its length the slower variability "
                f"(SD2); isolated points far from the cloud are beats the labelling missed. Right, spectrum of "
                f"the RR series with the {species} frequency bands shaded: in a resting recording the HF peak "
                f"sits at the breathing rate.")})
    secs = time.time() - t_start
    log.info("%s: %s %s @ %.0f Hz, %d segment(s), %.0f s recorded: %d beats, HR %s bpm, %.1f %% artifact, "
             "%.1f %% ectopic, agreement %s, %d stimulation window(s), %.1f s",
             rid, species, ch.name, det.fs, len(det.segments), recorded, n, fmt(mean_hr, 4), pct_art, pct_ect,
             fmt(agree, 3), len(windows), secs)
    head = {"n_beats": n, "mean_hr_bpm": num(mean_hr, 2), "sdnn_ms": num(rec_vals.get("sdnn"), 2),
            "rmssd_ms": num(rec_vals.get("rmssd"), 2), "lf_hf": num(rec_vals.get("lf_hf"), 3),
            "pct_artifact": num(pct_art, 2), "pct_ectopic": num(pct_ect, 2),
            "detector_agreement": num(agree, 4), "signal_quality": num(det.quality, 3),
            "n_stimulation_windows": len(windows)}
    if dev_mae is not None:
        head["hr_device_mae_bpm"] = num(dev_mae, 2)
    return {"status": "ok", "seconds": round(secs, 2), "species": species, "species_from": sp_from,
            "channel": ch.name, "calibrated": ch.calibrated, "headline": head, "preset_used": extra["used"],
            "beat_labels": {"method": (f"Lipponen & Tarvainen 2019 classes (NeuroKit2 signal_fixpeaks, Kubios) with a "
                                       f"Pennsieve gate: a flag is kept only when the RR interval differs from the "
                                       f"median of the 11 around it by more than {s.artifact_min_pct:g} %; plus "
                                       f"RR outside {s.rr_min_s:g}-{s.rr_max_s:g} s")
                            if s.artifact_method == "kubios" else s.artifact_method,
                            "counts": beats.counts}}


def main() -> int:
    started = datetime.now(timezone.utc)
    t_wall = time.time()
    try:
        p = read_params()
    except ParamError as e:
        log.error("bad parameter: %s", e)
        return 2
    in_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    out_root = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    out = out_root / APP_NAME
    out.mkdir(parents=True, exist_ok=True)
    log.info("ecg-hrv %s; threads %d (%s)", app_version(), _THREADS, _THREADS_WHY)
    recs = I.find_recordings(in_dir) if in_dir.is_dir() else []
    events, ev_files = I.find_events(in_dir) if in_dir.is_dir() else ({}, [])
    log.info("%d recording(s), stimulation events for %d recording(s) from %s", len(recs), len(events),
             ", ".join(ev_files) or "no events.tsv")
    writer = Writer(out)
    figs, warnings_, prov_recs, heads = [], [], {}, {}
    used_species: set = set()
    for rec in recs:
        if rec.error:
            log.error("%s: failed: %s", rec.rid, rec.error)
            prov_recs[rec.rid] = {"status": "failed", "error": rec.error}
            continue
        try:
            r = process(rec, p, events.get(rec.rid, []), out, writer, figs, warnings_)
        except Exception as e:
            log.error("%s: failed: %s\n%s", rec.rid, e, traceback.format_exc())
            prov_recs[rec.rid] = {"status": "failed", "error": str(e)}
            continue
        if r["status"] == "ok":
            heads[rec.rid] = r.pop("headline")
            used_species.add(r["species"])
        elif r["status"] == "skipped":
            warnings_.append(f"{rec.rid}: skipped: {r['reason']}")
        prov_recs[rec.rid] = r
    writer.close()
    unmatched = sorted(set(events) - set(prov_recs))
    if unmatched:
        warnings_.append(f"events.tsv names recordings not in the input: {', '.join(unmatched[:10])}")
    n_ok = sum(1 for r in prov_recs.values() if r["status"] == "ok")
    n_fail = sum(1 for r in prov_recs.values() if r["status"] == "failed")
    n_skip = sum(1 for r in prov_recs.values() if r["status"] == "skipped")
    if not prov_recs:
        warnings_.append("no recordings found under INPUT_DIR")
    elif n_ok == 0 and n_fail == 0:
        warnings_.append("no recording has an ECG channel: tables are empty")
    cites = list(CITATIONS)
    for sp in sorted(used_species):
        cites += [c for c in SPECIES_CITATIONS.get(sp, []) if c not in cites]
    cites += [c for c in DETECTOR_CITATIONS.get(p["detector"], []) if c not in cites]
    uncal = [rid for rid, r in prov_recs.items() if r.get("status") == "ok" and not r.get("calibrated", True)]
    notice = ("Research output: heart rate and HRV computed automatically from the ECG, for research; not a "
              "clinical ECG or rhythm interpretation.")
    if uncal:
        notice += f" ECG uncalibrated in {len(uncal)} recording(s): timing-derived measures only."
    tables = [
        {"path": "beats.csv", "caption": "One row per detected R-peak: time, the RR interval ending at it, and its "
                                         "label (N, ectopic, artifact)."},
        {"path": "series.csv", "caption": f"Heart rate and RR interval on a regular grid (shared shape with bp-resp; "
                                          f"read by stim-dose-response); valid=false across artifacts and gaps."},
        {"path": "hrv.csv", "caption": "HRV metrics (Task Force time domain, species-band frequency domain, "
                                       "non-linear) for the whole recording and for each stimulation phase; "
                                       "empty where the window is too short (reason in qc.csv notes)."},
        {"path": "qc.csv", "caption": "Per recording: beats, ectopic and artifact percentages, agreement between "
                                      "the two R-peak detectors, signal quality, device heart-rate agreement."},
        {"path": "features.csv", "caption": "One number per recording and feature (hrv_*, ecg_*), for group comparisons."},
    ]
    summary = {"app": APP_NAME, "version": app_version(), "recordings": heads, "figures": figs, "tables": tables,
               "warnings": warnings_, "notice": notice}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    params = dict(p["raw"])
    params["SPECIES_PRESETS"] = {sp: P.PRESETS[sp].as_dict() for sp in sorted(used_species)}
    inputs = []
    for rec in recs:
        if rec.path is not None:
            try:
                inputs.append(str(rec.path.relative_to(in_dir)))
            except ValueError:
                inputs.append(str(rec.path))
    for f in ev_files:
        try:
            inputs.append(str(Path(f).relative_to(in_dir)))
        except ValueError:
            inputs.append(f)
    prov = {"app": APP_NAME, "version": app_version(), "variant": None, "tools": tool_versions(),
            "parameters": params, "inputs": inputs, "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_seconds": round(time.time() - t_wall, 2), "peak_rss_mb": round(peak_rss_mb(), 1),
            "threads": {"requested": int(p["raw"]["THREADS"]), "used": _THREADS, "why": _THREADS_WHY},
            "recordings": prov_recs, "citations": cites, "licence_notice": None}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    log.info("done: %d ok, %d skipped, %d failed; %.1f s, peak RSS %.0f MB", n_ok, n_skip, n_fail,
             time.time() - t_wall, peak_rss_mb())
    for w in warnings_:
        log.warning(w)
    if not prov_recs:
        return 1
    if n_fail and n_ok == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
