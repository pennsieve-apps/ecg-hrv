"""
Validation of ecg-hrv's outputs against references in the demo data (not part of the image).

    python validation/validate.py fantasia <ecg-hrv output dir> <folder with the Fantasia WFDB records>
    python validation/validate.py aging    <ecg-hrv output dir>
    python validation/validate.py d655     <ecg-hrv output dir> <Discover 655 files/ folder>
    python validation/validate.py d349     <ecg-hrv output dir>

fantasia: R-peaks against the reference beat annotations (.ecg), sensitivity and positive
          predictivity within +-50 ms as wfdb.processing.compare_annotations computes them.
aging:    agreement between the two ECG leads (qc.csv agreement_other_lead).
d655:     the app's heart rate against the authors' heart rate (derivative/HR/*.xlsx; normalised,
          0.5 s steps): Pearson r after the best lag, per trial.
d349:     the app's heart rate against the acquisition device's heart-rate channel (qc.csv).
Prints one CSV line per recording.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


def rows(p: Path):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def fantasia(out: Path, data: Path):
    import wfdb
    from wfdb import processing as wp
    beats = rows(out / "beats.csv")
    print("recording,n_reference,n_detected,sensitivity,positive_predictivity,pct_artifact_or_ectopic,"
          "n_reference_non_N,of_which_labelled_non_N")
    for rid in sorted({r["recording_id"] for r in beats}):
        rec = rid.split("_")[0].replace("sub-", "")
        ann = wfdb.rdann(str(data / rec), "ecg")
        fs = ann.fs or wfdb.rdheader(str(data / rec)).fs
        beat_syms = set("NLRBAaJSVrFejnE/fQ?")
        ref = ann.sample[np.isin(ann.symbol, list(beat_syms))]
        mine = [r for r in beats if r["recording_id"] == rid]
        det = np.round(np.array([float(r["time_s"]) for r in mine]) * fs).astype(int)
        c = wp.compare_annotations(ref, det, int(round(0.05 * fs)))
        nonn = np.mean([r["label"] != "N" for r in mine]) * 100
        sym = np.asarray(ann.symbol)[np.isin(ann.symbol, list(beat_syms))]
        ref_x = ref[sym != "N"]
        lab_x = det[np.array([r["label"] != "N" for r in mine])]
        hit = sum(np.any(np.abs(lab_x - r) <= int(round(0.05 * fs))) for r in ref_x)
        print(f"{rid},{ref.size},{det.size},{c.sensitivity:.4f},{c.positive_predictivity:.4f},{nonn:.2f},"
              f"{ref_x.size},{hit}")


def aging(out: Path):
    print("recording,agreement_with_second_detector,agreement_other_lead,signal_quality,mean_hr_bpm")
    for r in rows(out / "qc.csv"):
        print(f"{r['recording_id']},{r['agreement_with_second_detector']},{r['agreement_other_lead']},"
              f"{r['signal_quality']},{r['mean_hr_bpm']}")


def d655(out: Path, files: Path):
    import pandas as pd
    perf = pd.read_excel(files / "performances.xlsx")
    series = rows(out / "series.csv")
    print("recording,authors_column,lag_s,pearson_r,n_points,app_mean_hr_bpm")
    for rid in sorted({r["recording_id"] for r in series}):
        tag = rid.split("rec-")[-1]                                 # set01-trial01-gain02
        row = perf[perf["performance id"].str.endswith(tag) & perf["performance id"].str.contains("Bimodal01")]
        if row.empty:
            continue
        trial = int(row["internal trial number"].iloc[0])
        gain = int(tag.split("gain")[-1])
        x = pd.read_excel(files / f"derivative/HR/{gain}x_HR.xlsx")
        col = next((c for c in x.columns if c.endswith(f"Vagus161_01252024_t{trial:03d}")), None)
        if col is None:
            print(f"{rid},not found,,,,")
            continue
        ta, va = x["Time"].to_numpy(float), x[col].to_numpy(float)
        ok = np.isfinite(ta) & np.isfinite(va)
        ta, va = ta[ok], va[ok]
        s = [(float(r["time_s"]), float(r["value"])) for r in series
             if r["recording_id"] == rid and r["variable"] == "hr_bpm" and r["valid"] == "true"]
        t, v = np.array(s).T
        best = (-2, 0)
        for lag in np.arange(0, t[-1] - ta[-1], 0.5):
            mine = np.interp(ta + lag, t, v)
            r_ = np.corrcoef(mine, va)[0, 1]
            if r_ > best[0]:
                best = (r_, lag)
        print(f"{rid},{col},{best[1]:.1f},{best[0]:.4f},{ta.size},{np.mean(v):.1f}")


def d349(out: Path):
    print("recording,hr_device_mae_bpm,hr_device_r,mean_hr_bpm,n_beats")
    for r in rows(out / "qc.csv"):
        print(f"{r['recording_id']},{r['hr_device_mae_bpm']},{r['hr_device_r']},{r['mean_hr_bpm']},{r['n_beats']}")


if __name__ == "__main__":
    kind, out = sys.argv[1], Path(sys.argv[2]) / "ecg-hrv"
    {"fantasia": lambda: fantasia(out, Path(sys.argv[3])), "aging": lambda: aging(out),
     "d655": lambda: d655(out, Path(sys.argv[3])), "d349": lambda: d349(out)}[kind]()
