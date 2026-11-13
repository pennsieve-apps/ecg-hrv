"""
Synthetic input for ecg-hrv, written to the physio-import contract (docs/conventions.md of
workflow-physiology), with h5py and numpy only (the review generator image has no pynwb).

    python -m tests.make_fixture <out>

writes
    <out>/physio/                 physio-import's `physio` port: recordings.csv, channels.csv,
                                  events.tsv, provenance.json, sub-*/<recording_id>.nwb
    <out>/events/stim-events/events.tsv   stim-events' output (stimulation trains)
    <out>/truth.json              the planted R-peak times and effects (not an input)

Recordings:
- sub-h01_rec-rest   human, 500 Hz, 6 min: ECG (V, float32) at ~65 bpm with respiratory sinus
                     arrhythmia at 0.25 Hz and a 0.1 Hz (LF) rhythm, one premature (ectopic) beat;
                     respiration and blood pressure channels.
- sub-r01_rec-vns    rat, 1 kHz, 4 min: inverted ECG lead with a double-lobed QRS (two lobes of
                     near-equal, varying size 7 ms apart), stored as int16 with a conversion factor,
                     ~360 bpm, three 10 s stimulation trains with a bradycardia that grows with
                     amplitude (10, 20, 35 % longer RR); a stimulation channel.
- sub-p01_rec-grid   pig, 1 kHz, trial-by-trial (NWB timestamps, gaps between three 25 s trials),
                     ~100 bpm, a 5 s train in each trial; a device heart-rate channel at 500 Hz.
- sub-h02_rec-noecg  human, respiration and blood pressure only: no ECG (the "missing signal" path).
"""
from __future__ import annotations

import argparse
import csv
import json
import uuid
from pathlib import Path

import h5py
import numpy as np

VLEN = h5py.string_dtype()
START = "2026-01-01T09:00:00+00:00"


def _attrs(obj, ntype: str, ns: str = "core", **kw):
    obj.attrs["neurodata_type"] = ntype
    obj.attrs["namespace"] = ns
    obj.attrs["object_id"] = str(uuid.uuid4())
    for k, v in kw.items():
        obj.attrs[k] = v


def _nwb(path: Path, rid: str, subject: str, species: str):
    f = h5py.File(path, "w")
    f.attrs["nwb_version"] = "2.8.0"
    _attrs(f, "NWBFile")
    f.create_dataset("identifier", data=rid, dtype=VLEN)
    f.create_dataset("session_description", data=f"synthetic ecg-hrv fixture {rid}", dtype=VLEN)
    f.create_dataset("session_start_time", data=START, dtype=VLEN)
    f.create_dataset("timestamps_reference_time", data=START, dtype=VLEN)
    f.create_dataset("file_create_date", data=[START], dtype=VLEN)
    for g in ("acquisition", "analysis", "processing", "stimulus/presentation", "stimulus/templates",
              "intervals", "general"):
        f.require_group(g)
    sub = f.create_group("general/subject")
    _attrs(sub, "Subject")
    sub.create_dataset("subject_id", data=subject, dtype=VLEN)
    sub.create_dataset("species", data=species, dtype=VLEN)
    return f


def _series(f, name: str, data: np.ndarray, unit: str, fs: float | None = None, timestamps=None,
            conversion: float = 1.0, description: str = ""):
    g = f["acquisition"].create_group(name)
    _attrs(g, "TimeSeries", description=description or name, comments="no comments")
    d = g.create_dataset("data", data=data, chunks=(min(len(data), 65536),), compression="gzip")
    d.attrs["unit"] = unit
    d.attrs["conversion"] = conversion
    d.attrs["offset"] = 0.0
    d.attrs["resolution"] = -1.0
    if timestamps is None:
        st = g.create_dataset("starting_time", data=0.0)
        st.attrs["rate"] = float(fs)
        st.attrs["unit"] = "seconds"
    else:
        ts = g.create_dataset("timestamps", data=np.asarray(timestamps, dtype=np.float64))
        ts.attrs["interval"] = 1
        ts.attrs["unit"] = "seconds"


def beat_times(t0: float, t1: float, rr0: float, rng, hf=(0.25, 0.04), lf=(0.1, 0.03), slow=None) -> np.ndarray:
    """Beat times from an RR that is modulated at a respiratory (HF) and an LF frequency, times an
    optional slow factor slow(t) (stimulation bradycardia)."""
    out = [t0 + rr0 * rng.uniform(0.2, 0.8)]
    while True:
        t = out[-1]
        rr = rr0 * (1 + hf[1] * np.sin(2 * np.pi * hf[0] * t) + lf[1] * np.sin(2 * np.pi * lf[0] * t + 1.0))
        rr *= 1 + rng.normal(0, 0.005)
        if slow is not None:
            rr *= slow(t)
        if t + rr >= t1:
            break
        out.append(t + rr)
    return np.array(out)


def ecg_wave(t: np.ndarray, beats: np.ndarray, qrs_s: float, qt_s: float, rng, amp_mv: float = 1.0,
             noise_mv: float = 0.02) -> np.ndarray:
    """ECG in volts: P, Q, R, S, T Gaussians around each beat, baseline wander and noise."""
    x = np.zeros_like(t)
    fs = 1.0 / np.median(np.diff(t))
    sr = qrs_s / 6
    for b in beats:
        a, z = np.searchsorted(t, [b - 0.25 * qt_s - 4 * sr, b + 1.4 * qt_s])
        tt = t[a:z] - b
        x[a:z] += (amp_mv * np.exp(-0.5 * (tt / sr) ** 2)
                   - 0.15 * amp_mv * np.exp(-0.5 * ((tt + 1.6 * sr) / sr) ** 2)
                   - 0.25 * amp_mv * np.exp(-0.5 * ((tt - 1.8 * sr) / sr) ** 2)
                   + 0.25 * amp_mv * np.exp(-0.5 * ((tt - 0.75 * qt_s) / (0.12 * qt_s)) ** 2)
                   + 0.1 * amp_mv * np.exp(-0.5 * ((tt + 0.45 * qt_s) / (0.06 * qt_s)) ** 2))
    x += 0.1 * amp_mv * np.sin(2 * np.pi * 0.3 * t + 0.5) + rng.normal(0, noise_mv, t.size)
    del fs
    return x / 1000.0


def resp_wave(t, rate_hz=0.25):
    return np.sin(2 * np.pi * rate_hz * t).astype(np.float32)


def bp_wave(t, beats, sbp=120.0, dbp=80.0):
    x = np.full_like(t, dbp)
    for b in beats:
        a, z = np.searchsorted(t, [b, b + 0.6])
        tt = t[a:z] - b
        x[a:z] += (sbp - dbp) * np.sin(np.pi * np.clip(tt / 0.3, 0, 1)) * np.exp(-tt / 0.4)
    return x.astype(np.float32)


def write(out: Path, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    physio = out / "physio"
    physio.mkdir(parents=True, exist_ok=True)
    recs, chans, truth = [], [], {}
    stim_rows = []

    def chan(rid, name, typ, orig, unit, fs, n, calibrated=True, scale=1.0):
        chans.append({"recording_id": rid, "channel": name, "type": typ, "original_name": orig, "unit": unit,
                      "sampling_rate_hz": fs, "n_samples": n, "calibrated": "true" if calibrated else "false",
                      "scale": scale, "site": "", "typing": "name", "status": "good", "status_description": ""})

    # ---------------------------------------------------------------- human, resting
    rid, fs, dur = "sub-h01_rec-rest", 500.0, 360.0
    t = np.arange(int(dur * fs)) / fs
    b = beat_times(0.0, dur, 60 / 65.0, rng)
    ect = int(np.searchsorted(b, 100.0))
    b = b.copy()
    b[ect] = b[ect - 1] + 0.6 * (b[ect] - b[ect - 1])          # premature beat, compensatory pause follows
    x = ecg_wave(t, b, 0.08, 0.36, rng)
    p = physio / "sub-h01" / f"{rid}.nwb"
    p.parent.mkdir(parents=True, exist_ok=True)
    with _nwb(p, rid, "h01", "Homo sapiens") as f:
        _series(f, "ecg", x.astype(np.float32), "V", fs, description="ECG lead II (synthetic)")
        _series(f, "resp", resp_wave(t), "a.u.", fs)
        _series(f, "bp", bp_wave(t, b), "mmHg", fs)
    for name, typ, orig, unit in (("ecg", "ecg", "ECG II", "V"), ("resp", "resp", "Resp", "a.u."), ("bp", "bp", "ABP", "mmHg")):
        chan(rid, name, typ, orig, unit, fs, t.size, calibrated=unit != "a.u.")
    recs.append({"recording_id": rid, "subject": "h01", "session": "", "label": "rest", "path": f"sub-h01/{rid}.nwb",
                 "species": "human", "start_time": "2026-01-01T09:00:00", "duration_s": dur, "recorded_s": dur,
                 "n_segments": 1, "channel_types": "ecg|resp|bp", "source_format": "nwb", "source_path": "synthetic",
                 "metadata_source": "file", "age": "", "sex": ""})
    truth[rid] = {"beats_s": b.tolist(), "ectopic_index": ect, "species": "human", "mean_hr_bpm": 65.0,
                  "hf_hz": 0.25, "lf_hz": 0.1}

    # ---------------------------------------------------------------- rat, VNS trains, inverted int16 lead
    rid, fs, dur = "sub-r01_rec-vns", 1000.0, 240.0
    t = np.arange(int(dur * fs)) / fs
    trains = [(60.0, 10.0, 100.0, 0.10), (120.0, 10.0, 200.0, 0.20), (180.0, 10.0, 400.0, 0.35)]

    def slow(tt):
        f_ = 1.0
        for on, d, _, eff in trains:
            if on <= tt < on + d:
                f_ *= 1 + eff * min(1.0, (tt - on) / 2.0)
            elif on + d <= tt < on + d + 4:
                f_ *= 1 + eff * (1 - (tt - on - d) / 4.0)
        return f_

    b = beat_times(0.0, dur, 60 / 360.0, rng, hf=(1.2, 0.02), lf=(0.4, 0.02), slow=slow)
    # inverted lead with a double-lobed QRS as in dataset 655: two negative lobes 7 ms apart whose
    # sizes vary independently around the same value, so the deepest sample jumps between them
    x = 0.05e-3 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.01e-3, t.size)
    sig = 0.0015
    for bt in b:
        a_, z_ = np.searchsorted(t, [bt - 0.01, bt + 0.08])
        tt = t[a_:z_] - bt
        x[a_:z_] += 1e-3 * (-rng.uniform(0.45, 0.55) * np.exp(-0.5 * (tt / sig) ** 2)
                            - rng.uniform(0.45, 0.55) * np.exp(-0.5 * ((tt - 0.007) / sig) ** 2)
                            + 0.12 * np.exp(-0.5 * ((tt - 0.045) / 0.007) ** 2))
    conv = 1e-6                                                       # int16 counts of 1 uV
    xi = np.clip(np.round(x / conv), -32768, 32767).astype(np.int16)
    stim = np.zeros(t.size, dtype=np.float32)
    for on, d, amp, _ in trains:
        stim[(t >= on) & (t < on + d)] = amp
    p = physio / "sub-r01" / f"{rid}.nwb"
    p.parent.mkdir(parents=True, exist_ok=True)
    with _nwb(p, rid, "r01", "Rattus norvegicus") as f:
        _series(f, "ecg", xi, "V", fs, conversion=conv, description="ECG (synthetic, inverted)")
        _series(f, "stim", stim, "uA", fs, description="StimAmp envelope (synthetic)")
    chan(rid, "ecg", "ecg", "3 ECG", "V", fs, t.size, scale=conv)
    chan(rid, "stim", "stim", "4 StimAmp", "uA", fs, t.size)
    recs.append({"recording_id": rid, "subject": "r01", "session": "", "label": "vns", "path": f"sub-r01/{rid}.nwb",
                 "species": "rat", "start_time": "", "duration_s": dur, "recorded_s": dur, "n_segments": 1,
                 "channel_types": "ecg|stim", "source_format": "csv", "source_path": "synthetic",
                 "metadata_source": "mapping", "age": "", "sex": ""})
    for i, (on, d, amp, eff) in enumerate(trains):
        stim_rows.append((rid, on, d, f"stim-{i + 1:04d}", amp, 20.0, eff))
    truth[rid] = {"beats_s": b.tolist(), "species": "rat", "mean_hr_bpm": 360.0, "inverted": True,
                  "trains": [{"onset": on, "duration": d, "amplitude": a, "rr_increase": e} for on, d, a, e in trains]}

    # ---------------------------------------------------------------- pig, trial-by-trial with gaps
    rid, fs = "sub-p01_rec-grid", 1000.0
    trial_starts = [0.0, 55.0, 110.0]
    ts_all, x_all, hr_t, hr_v, beats_all = [], [], [], [], []
    for k, t0 in enumerate(trial_starts):
        tt = t0 + np.arange(int(25.0 * fs)) / fs
        on = t0 + 10.0

        def slow_p(s_, on=on, eff=0.15 * (k + 1)):
            return 1 + eff if on <= s_ < on + 5.0 else 1.0

        bk = beat_times(t0, t0 + 25.0, 0.6, rng, hf=(0.3, 0.02), lf=(0.05, 0.02), slow=slow_p)
        ts_all.append(tt)
        x_all.append(ecg_wave(tt, bk, 0.06, 0.3, rng, amp_mv=1.5))
        beats_all.append(bk)
        th = t0 + np.arange(int(25.0 * 500)) / 500.0
        inst = np.interp(th, bk[1:], 60.0 / np.diff(bk))
        hr_t.append(th)
        hr_v.append(inst + rng.normal(0, 0.5, th.size))
        stim_rows.append((rid, on, 5.0, f"stim-{k + 1:04d}", [500.0, 1000.0, 2000.0][k], 10.0, 0.15 * (k + 1)))
    p = physio / "sub-p01" / f"{rid}.nwb"
    p.parent.mkdir(parents=True, exist_ok=True)
    with _nwb(p, rid, "p01", "Sus scrofa") as f:
        _series(f, "ecg", np.concatenate(x_all).astype(np.float32), "V", timestamps=np.concatenate(ts_all),
                description="raw_ecg (synthetic)")
        _series(f, "hr", np.concatenate(hr_v).astype(np.float32), "bpm", timestamps=np.concatenate(hr_t),
                description="device heart rate (synthetic)")
    n = sum(a.size for a in ts_all)
    chan(rid, "ecg", "ecg", "raw_ecg", "V", fs, n)
    chan(rid, "hr", "hr", "hr", "bpm", 500.0, sum(a.size for a in hr_t))
    recs.append({"recording_id": rid, "subject": "p01", "session": "", "label": "grid", "path": f"sub-p01/{rid}.nwb",
                 "species": "pig", "start_time": "", "duration_s": 135.0, "recorded_s": 75.0, "n_segments": 3,
                 "channel_types": "ecg|hr", "source_format": "hdf5", "source_path": "synthetic",
                 "metadata_source": "sds", "age": "", "sex": ""})
    truth[rid] = {"beats_s": np.concatenate(beats_all).tolist(), "species": "pig", "mean_hr_bpm": 100.0,
                  "segments": [[s_, s_ + 25.0] for s_ in trial_starts]}

    # ---------------------------------------------------------------- human without ECG
    rid, fs, dur = "sub-h02_rec-noecg", 100.0, 120.0
    t = np.arange(int(dur * fs)) / fs
    b = beat_times(0.0, dur, 0.9, rng)
    p = physio / "sub-h02" / f"{rid}.nwb"
    p.parent.mkdir(parents=True, exist_ok=True)
    with _nwb(p, rid, "h02", "Homo sapiens") as f:
        _series(f, "resp", resp_wave(t), "a.u.", fs)
        _series(f, "bp", bp_wave(t, b), "mmHg", fs)
    chan(rid, "resp", "resp", "Resp", "a.u.", fs, t.size, calibrated=False)
    chan(rid, "bp", "bp", "Finapres", "mmHg", fs, t.size)
    recs.append({"recording_id": rid, "subject": "h02", "session": "", "label": "noecg", "path": f"sub-h02/{rid}.nwb",
                 "species": "human", "start_time": "", "duration_s": dur, "recorded_s": dur, "n_segments": 1,
                 "channel_types": "resp|bp", "source_format": "nwb", "source_path": "synthetic",
                 "metadata_source": "file", "age": "", "sex": ""})
    truth[rid] = {"beats_s": [], "species": "human", "no_ecg": True}

    # ---------------------------------------------------------------- tables
    def write_csv(path, rows, header, delimiter=","):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=header, delimiter=delimiter, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    write_csv(physio / "recordings.csv", recs, list(recs[0]))
    write_csv(physio / "channels.csv", chans, list(chans[0]))
    ev_header = ["recording_id", "onset", "duration", "trial_type", "event_id", "amplitude", "amplitude_unit",
                 "pulse_width_us", "frequency_hz", "train_duration_s", "train_on_s", "train_off_s", "n_pulses",
                 "polarity", "waveform", "site", "contact", "condition", "label", "source"]
    na = {k: "n/a" for k in ev_header}
    write_csv(physio / "events.tsv", [{**na, "recording_id": "sub-h01_rec-rest", "onset": 30.0, "duration": "n/a",
                                       "trial_type": "marker", "event_id": "marker-0001", "label": "eyes closed",
                                       "source": "marker"}], ev_header, "\t")
    (physio / "provenance.json").write_text(json.dumps({"app": "physio-import", "version": "fixture"}) + "\n")
    ev_dir = out / "events" / "stim-events"
    ev_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rid_, on, d, eid, amp, freq, _ in stim_rows:
        rows.append({**na, "recording_id": rid_, "onset": on, "duration": d, "trial_type": "stimulation",
                     "event_id": eid, "amplitude": amp, "amplitude_unit": "uA", "pulse_width_us": 200,
                     "frequency_hz": freq, "train_duration_s": d, "waveform": "biphasic", "source": "envelope"})
    write_csv(ev_dir / "events.tsv", rows, ev_header, "\t")
    (out / "truth.json").write_text(json.dumps(truth) + "\n")
    return truth


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    write(a.out, a.seed)
    print(f"fixture written to {a.out}")


if __name__ == "__main__":
    main()
