"""
R-peak detection, beat labels and HRV, one recording at a time.

- Detection: the ECG is band-passed (species band, zero-phase Butterworth via NeuroKit2's
  signal_filter) and NeuroKit2's ecg_findpeaks runs on it a window at a time (CHUNK_S plus 2 s on
  each side, peaks kept only in the window's core), so memory does not grow with the recording.
  The "neurokit" detector gets its smoothing, averaging and refractory windows from the species
  preset; the other NeuroKit2 detectors keep their published (human) constants.
- Second detector: wfdb's XQRS with the preset's heart-rate range, QRS width and refractory
  period, on the same windows (high-passed at 0.5 Hz); agreement = 2 x matched / (n1 + n2) within a tolerance.
- Polarity: NeuroKit2's ecg_invert rule on the first 120 s decides whether the lead is inverted.
- Labels: Lipponen & Tarvainen 2019 artefact classes as NeuroKit2's signal_fixpeaks ("Kubios")
  computes them on the detected beats (ectopic; missed, extra, long/short -> artifact), plus RR
  intervals outside the preset's RRmin-RRmax. An NN interval is one between two N beats of the
  same contiguous segment.
- HRV: NeuroKit2's hrv_time and hrv_frequency on the NN intervals of a window; SD1/SD2 (Poincare),
  DFA alpha1/alpha2 (NeuroKit2 fractal_dfa) and sample entropy (NeuroKit2 entropy_sample) as
  NeuroKit2's hrv_nonlinear computes them, each only when the window is long enough.
"""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import neurokit2 as nk
from scipy import signal as sps
from scipy.ndimage import median_filter
from wfdb import processing as wp

from processor import inputs as I
from processor.presets import Preset, min_window_s

log = logging.getLogger("ecg-hrv")

DETECTORS = ("neurokit", "pantompkins1985", "hamilton2002", "elgendi2010", "engzeemod2012", "kalidas2017",
             "rodrigues2021")
PAD_S = 2.0
POLARITY_S = 120.0
MIN_BEATS_MEAN = 2         # mean NN and mean heart rate need at least this many NN intervals
MIN_BEATS_TIME = 10        # variability metrics need at least this many
MIN_BEATS_POINCARE = 20
MIN_BEATS_DFA1 = 60        # 4x the largest alpha1 box (15 beats)
MIN_BEATS_DFA2 = 256       # 4x 64 beats; boxes 16..min(128, n/4)
MIN_BEATS_SAMPEN = 200
MIN_COVERAGE = 0.8         # frequency metrics need NN intervals covering 80 % of the window
TIME_METRICS = ["mean_nn", "mean_hr", "sdnn", "rmssd", "sdsd", "pnn50", "pnnx", "cvnn"]
FREQ_METRICS = ["vlf", "lf", "hf", "total_power", "lf_hf", "lf_nu", "hf_nu", "lf_peak", "hf_peak"]
NONLIN_METRICS = ["sd1", "sd2", "sd1_sd2", "dfa_alpha1", "dfa_alpha2", "sampen"]
UNITS = {"mean_nn": "ms", "mean_hr": "bpm", "sdnn": "ms", "rmssd": "ms", "sdsd": "ms", "pnn50": "%", "pnnx": "%",
         "cvnn": "ratio", "vlf": "ms2", "lf": "ms2", "hf": "ms2", "total_power": "ms2", "lf_hf": "ratio",
         "lf_nu": "n.u. (%)", "hf_nu": "n.u. (%)", "lf_peak": "Hz", "hf_peak": "Hz", "sd1": "ms", "sd2": "ms",
         "sd1_sd2": "ratio", "dfa_alpha1": "", "dfa_alpha2": "", "sampen": ""}


@dataclass
class Settings:
    preset: Preset
    detector: str = "neurokit"
    filter_band: tuple = (3.0, 100.0)
    refractory_s: float = 0.25
    rr_min_s: float = 0.28
    rr_max_s: float = 2.4
    bands: dict = field(default_factory=dict)        # vlf/lf/hf
    pnn_ms: float = 50.0
    invert: str = "auto"                             # auto | true | false
    second_detector: bool = True
    chunk_s: float = 300.0
    powerline_hz: float = -1.0                       # -1 = detect 50/60 Hz, 0 = off
    artifact_method: str = "kubios"                  # kubios | range | none
    artifact_min_pct: float = 10.0                   # Kubios flags kept only above this RR deviation

    @property
    def tolerance_s(self) -> float:
        """Matching tolerance between detectors and against reference annotations: 50 ms (the
        usual +-50 ms of beat-detector benchmarks), shrunk to 30 % of RRmin for fast hearts."""
        return min(0.05, 0.3 * self.rr_min_s)


@dataclass
class Detection:
    times: np.ndarray                   # R-peak times (s from recording start), all segments
    seg_index: np.ndarray               # segment of each beat
    xqrs_times: np.ndarray | None
    inverted: bool
    quality: float                      # mean NeuroKit2 averageQRS template correlation (0-1)
    fs: float
    segments: list
    amplitude: float                    # median R amplitude after cleaning (signal units)
    strip: dict | None = None           # a 10 s excerpt for the figure
    mains_hz: float = 0.0               # mains frequency notched (0 = none)


def _bandpass(x: np.ndarray, fs: float, band: tuple, powerline: float) -> np.ndarray:
    lo, hi = band
    hi = min(hi, 0.45 * fs)
    y = nk.signal_filter(x, sampling_rate=fs, lowcut=lo, highcut=hi, method="butterworth", order=2)
    if powerline:
        f0 = powerline
        while f0 < 0.45 * fs:  # mains and its harmonics: zero-phase IIR notches (Q 30)
            b, a = sps.iirnotch(f0, 30.0, fs)
            y = sps.filtfilt(b, a, y)
            f0 += powerline
    return y


def detect_mains(x: np.ndarray, fs: float) -> float:
    """50 or 60 Hz when that line stands out of the ECG spectrum (more than 20 times the median
    power density of 35-45 Hz), else 0."""
    if fs < 130 or x.size < 4 * fs:
        return 0.0
    f, p = sps.welch(x - np.mean(x), fs=fs, nperseg=int(min(x.size, 4 * fs)))
    ref = np.median(p[(f >= 35) & (f <= 45)])
    best, ratio = 0.0, 20.0
    for f0 in (50.0, 60.0):
        m = (f >= f0 - 1) & (f <= f0 + 1)
        if m.any() and ref > 0 and p[m].max() / ref > ratio:
            best, ratio = f0, p[m].max() / ref
    return best


def _find_peaks(clean: np.ndarray, fs: float, s: Settings) -> np.ndarray:
    kw = {}
    if s.detector == "neurokit":
        kw = {"smoothwindow": s.preset.smooth_s, "avgwindow": s.preset.avg_s, "mindelay": s.refractory_s}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = nk.ecg_findpeaks(clean, sampling_rate=int(round(fs)) if s.detector != "neurokit" else fs,
                                method=s.detector, **kw)
    pk = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    if pk.size and s.detector != "neurokit":
        # other detectors keep their own refractory constants: enforce the preset's one
        keep = [pk[0]]
        for p in pk[1:]:
            if (p - keep[-1]) / fs >= s.refractory_s:
                keep.append(p)
        pk = np.asarray(keep, dtype=np.int64)
    return pk


def _xqrs(raw: np.ndarray, fs: float, s: Settings) -> np.ndarray:
    scale = np.subtract(*np.percentile(raw, [99.5, 0.5]))
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros(0, dtype=np.int64)
    x = 2.0 * raw / scale  # ~2 "mV" peak-to-peak: XQRS's learning thresholds are in mV
    p = s.preset
    hr_min = max(1.0, 60.0 / s.rr_max_s)
    hr_max = 60.0 / s.rr_min_s
    conf = wp.XQRS.Conf(hr_init=min(max(p.typical_hr_bpm, hr_min + 1), hr_max - 1), hr_max=hr_max, hr_min=hr_min,
                        qrs_width=max(p.qrs_s, 2.0 / fs), qrs_thr_init=0.13, ref_period=s.refractory_s)
    try:
        xq = wp.XQRS(sig=x, fs=fs, conf=conf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xq.detect(verbose=False)
        return np.asarray(xq.qrs_inds, dtype=np.int64)
    except Exception as e:  # XQRS raises on very short or flat windows
        log.debug("XQRS failed on a window: %s", e)
        return np.zeros(0, dtype=np.int64)


def match(a: np.ndarray, b: np.ndarray, tol: float) -> int:
    """Number of one-to-one matches between two sorted time arrays within tol."""
    i = j = n = 0
    while i < len(a) and j < len(b):
        d = a[i] - b[j]
        if abs(d) <= tol:
            n, i, j = n + 1, i + 1, j + 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return n


def agreement(a: np.ndarray, b: np.ndarray, tol: float) -> float:
    if len(a) + len(b) == 0:
        return float("nan")
    return 2.0 * match(a, b, tol) / (len(a) + len(b))


def decide_polarity(reader: I.SeriesReader, s: Settings) -> tuple[bool, float]:
    """(inverted, mains frequency to notch) from the first 120 s of the longest segment."""
    seg = max(reader.segments, key=lambda g: g.stop - g.start)
    n = min(seg.stop - seg.start, int(POLARITY_S * seg.fs))
    x = reader.read(seg.start, seg.start + n)
    if n < 2 * seg.fs or not np.any(np.isfinite(x)):
        return s.invert == "true", max(s.powerline_hz, 0.0)
    x = np.nan_to_num(x - np.nanmedian(x))
    mains = detect_mains(x, seg.fs) if s.powerline_hz < 0 else s.powerline_hz
    if s.invert in ("true", "false"):
        return s.invert == "true", mains
    # NeuroKit2's ecg_invert rule (in each 2 s window, the sign of the sample with the largest
    # squared amplitude of the cleaned ECG; inverted when their median is negative), applied to
    # non-overlapping windows: ecg_invert's rolling window needs ~1.8 GB for 120 s at 1 kHz.
    clean = _bandpass(x, seg.fs, s.filter_band, mains)
    w = int(2 * seg.fs)
    k = clean.size // w
    blocks = clean[: k * w].reshape(k, w)
    peaks = blocks[np.arange(k), np.argmax(blocks ** 2, axis=1)]
    inv = float(np.median(peaks)) < 0
    return bool(inv), mains


def detect(reader: I.SeriesReader, s: Settings, progress=None) -> Detection:
    inverted, mains = decide_polarity(reader, s)
    times, segix, xq_times, qual, amps = [], [], [], [], []
    strip = None
    fs0 = reader.segments[0].fs if reader.segments else float("nan")
    for k, seg in enumerate(reader.segments):
        fs = seg.fs
        if seg.duration < max(2.0, 3 * s.rr_max_s):
            continue
        chunk = max(int(s.chunk_s * fs), int(10 * fs))
        pad = int(PAD_S * fs)
        for a in range(seg.start, seg.stop, chunk):
            b = min(a + chunk, seg.stop)
            ra, rb = max(seg.start, a - pad), min(seg.stop, b + pad)
            raw = reader.read(ra, rb)
            if not np.any(np.isfinite(raw)):
                continue
            raw = np.nan_to_num(raw - np.nanmedian(raw))
            if inverted:
                raw = -raw
            if rb - ra < 3 * max(s.preset.avg_s, 0.2) * fs:
                continue
            clean = _bandpass(raw, fs, s.filter_band, mains)
            pk = _find_peaks(clean, fs, s)
            core = pk[(pk + ra >= a) & (pk + ra < b)]
            t = reader.times(seg, ra, rb)
            times.append(t[core])
            segix.append(np.full(core.size, k))
            if core.size:
                amps.append(np.median(clean[core]))
            if pk.size >= 4:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        q = nk.ecg_quality(clean, rpeaks=pk, sampling_rate=int(round(fs)), method="averageQRS")
                        qual.append((float(np.nanmean(np.asarray(q)[a - ra:b - ra])), b - a))
                    except Exception as e:
                        log.debug("ecg_quality failed: %s", e)
            if s.second_detector:
                # XQRS gets the ECG only high-passed at 0.5 Hz (and notched): with baseline
                # wander left in, its learning phase can fail for a whole window
                xp = _xqrs(_bandpass(raw, fs, (0.5, s.filter_band[1]), mains), fs, s)
                xc = xp[(xp + ra >= a) & (xp + ra < b)]
                xq_times.append(t[xc])
            if strip is None and core.size >= 3:
                w0 = max(0, int(core[0] - 0.5 * fs))
                w1 = min(raw.size, w0 + int(10 * fs))
                sel = core[(core >= w0) & (core < w1)]
                strip = {"t": t[w0:w1], "x": raw[w0:w1], "peaks_t": t[sel], "peaks_x": raw[sel],
                         "fs": fs}
            if progress:
                progress(b, seg)
    times_a = np.concatenate(times) if times else np.zeros(0)
    seg_a = np.concatenate(segix).astype(int) if segix else np.zeros(0, dtype=int)
    order = np.argsort(times_a, kind="stable")
    xq = np.sort(np.concatenate(xq_times)) if (s.second_detector and xq_times) else None
    qv = (sum(q * w for q, w in qual) / sum(w for _, w in qual)) if qual else float("nan")
    return Detection(times_a[order], seg_a[order], xq, inverted, qv, fs0, reader.segments,
                     float(np.median(amps)) if amps else float("nan"), strip, mains)


# --------------------------------------------------------------------------- labels and NN intervals

@dataclass
class Beats:
    times: np.ndarray
    rr_ms: np.ndarray          # interval ending at each beat (nan for the first beat of a segment)
    labels: np.ndarray         # "N" | "ectopic" | "artifact"
    seg: np.ndarray
    nn_ms: np.ndarray          # NN intervals
    nn_t: np.ndarray           # time of the beat ending each NN interval
    nn_start: np.ndarray       # time of the beat starting it
    counts: dict


def label_beats(det: Detection, s: Settings) -> Beats:
    t, seg = det.times, det.seg_index
    n = t.size
    labels = np.array(["N"] * n, dtype=object)
    rr = np.full(n, np.nan)
    counts = {"ectopic": 0, "missed": 0, "extra": 0, "longshort": 0, "out_of_range": 0}
    for k in np.unique(seg):
        idx = np.nonzero(seg == k)[0]
        if idx.size < 2:
            continue
        tk = t[idx]
        rr[idx[1:]] = np.diff(tk) * 1000.0
        if s.artifact_method == "kubios" and idx.size >= 10:
            # signal_fixpeaks works on sample indices: use a 1 kHz grid so the classes are
            # computed on the exact detected times whatever the sampling rate.
            pk = np.round((tk - tk[0]) * 1000.0).astype(np.int64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                art, _ = nk.signal_fixpeaks(pk, sampling_rate=1000, iterative=False, method="Kubios")
            rk = rr[idx]
            # Lipponen & Tarvainen's thresholds scale with the beat-to-beat variability over 91
            # beats; in the very regular rhythm of anaesthetised animals (RR changes of 1-2 ms) an
            # ordinary 5 % change crosses them. A flag is kept only where the interval differs from
            # the median of the 11 around it by more than ARTIFACT_MIN_PCT.
            fill = np.where(np.isfinite(rk), rk, np.nanmedian(rk))
            dev = np.abs(fill / median_filter(fill, size=11, mode="nearest") - 1.0) * 100.0

            def big(j: int) -> bool:
                return any(dev[k] > s.artifact_min_pct for k in (j, j + 1) if 0 <= k < idx.size)

            ect = set()
            for i in np.asarray(art.get("ectopic", []), dtype=int):
                # NeuroKit2 reports an ectopic pattern at the beat after the premature one (whose
                # interval is the long, compensatory one): label the premature beat of the pair.
                cand = [j for j in (i - 1, i) if 1 <= j < idx.size and np.isfinite(rk[j])]
                if cand:
                    j = min(cand, key=lambda j: rk[j])
                    if big(j):
                        ect.add(j)
            counts["ectopic"] += len(ect)
            for j in ect:
                labels[idx[j]] = "ectopic"
            for key in ("missed", "extra", "longshort"):
                ii = np.asarray(art.get(key, []), dtype=int)
                ii = [j for j in ii if 0 <= j < idx.size and j not in ect and big(j)]
                counts[key] += len(ii)
                for j in ii:
                    labels[idx[j]] = "artifact"
        if s.artifact_method == "none":
            continue
        rk = rr[idx]
        bad = np.isfinite(rk) & ((rk < s.rr_min_s * 1000) | (rk > s.rr_max_s * 1000))
        counts["out_of_range"] += int(bad.sum())
        labels[idx[bad & (labels[idx] == "N")]] = "artifact"
    # NN: consecutive N beats within one segment
    ok = np.zeros(n, dtype=bool)
    if n > 1:
        ok[1:] = (labels[1:] == "N") & (labels[:-1] == "N") & (seg[1:] == seg[:-1]) & np.isfinite(rr[1:])
    if s.artifact_method != "none":
        ok &= ~(np.isfinite(rr) & ((rr < s.rr_min_s * 1000) | (rr > s.rr_max_s * 1000)))
    nn_idx = np.nonzero(ok)[0]
    return Beats(t, rr, labels, seg, rr[nn_idx], t[nn_idx], t[nn_idx - 1] if nn_idx.size else np.zeros(0), counts)


# --------------------------------------------------------------------------- HRV in a window

def _welch_peaks(nn_ms, nn_t, s: Settings, interp_hz: float) -> dict:
    out = {"lf_peak": None, "hf_peak": None}
    if nn_ms.size < 4:
        return out
    g = np.arange(nn_t[0], nn_t[-1], 1.0 / interp_hz)
    if g.size < 16:
        return out
    y = np.interp(g, nn_t, nn_ms)
    y = sps.detrend(y)
    lf = s.bands["lf"]
    nper = min(g.size, int(interp_hz * min_window_s(lf, "lf")))
    f, p = sps.welch(y, fs=interp_hz, nperseg=max(16, nper))
    for key in ("lf", "hf"):
        lo, hi = s.bands[key]
        m = (f >= lo) & (f < hi)
        if m.any() and np.any(p[m] > 0):
            out[f"{key}_peak"] = float(f[m][np.argmax(p[m])])
    return out


def interp_rate(s: Settings) -> float:
    """Resampling rate of the RR series for spectra: at least 4 Hz, and 2.25 x the upper edge of
    the band above HF (PhysioZoo's rule, Behar et al. 2018)."""
    return float(max(4.0, math.ceil(2.25 * s.bands["hf"][1] * 1.1)))


def hrv_window(nn_ms: np.ndarray, nn_t: np.ndarray, nn_start: np.ndarray, a: float, b: float,
               s: Settings) -> tuple[dict, dict, int]:
    """HRV metrics of the NN intervals that lie entirely within [a, b]. Returns (values, reasons
    for empty metrics, n_beats)."""
    sel = (nn_start >= a) & (nn_t <= b)
    x, t = nn_ms[sel], nn_t[sel]
    n = int(x.size)
    win = b - a
    vals: dict = {m: None for m in TIME_METRICS + FREQ_METRICS + NONLIN_METRICS}
    why: dict = {}
    if n < MIN_BEATS_TIME:
        if n >= MIN_BEATS_MEAN:  # a mean heart rate is still meaningful in a short window
            vals["mean_nn"] = float(np.mean(x))
            vals["mean_hr"] = float(np.mean(60000.0 / x))
        for m in vals:
            if vals[m] is None:
                why[m] = f"fewer than {MIN_BEATS_TIME} NN intervals"
        return vals, why, n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        td = nk.hrv_time({"RRI": x, "RRI_Time": t}).iloc[0]
    vals["mean_nn"] = float(td["HRV_MeanNN"])
    vals["mean_hr"] = float(np.mean(60000.0 / x))
    vals["sdnn"] = float(td["HRV_SDNN"])
    vals["rmssd"] = float(td["HRV_RMSSD"])
    vals["sdsd"] = float(td["HRV_SDSD"])
    vals["pnn50"] = float(td["HRV_pNN50"])
    vals["cvnn"] = float(td["HRV_CVNN"])
    # pNNx with the species threshold, over successive NN pairs (both intervals adjacent in time)
    succ = np.abs(t[1:] - x[1:] / 1000.0 - t[:-1]) < 1e-3
    d = np.abs(np.diff(x))[succ]
    vals["pnnx"] = float(100.0 * np.mean(d > s.pnn_ms)) if d.size else None

    # frequency domain
    coverage = float(np.sum(x) / 1000.0 / win) if win > 0 else 0.0
    need = {"vlf": min_window_s(s.bands["vlf"], "vlf"), "lf": min_window_s(s.bands["lf"], "lf"),
            "hf": min_window_s(s.bands["hf"], "hf")}
    if coverage < MIN_COVERAGE:
        for m in FREQ_METRICS:
            why[m] = f"NN intervals cover {100 * coverage:.0f} % of the window (< {100 * MIN_COVERAGE:.0f} %)"
    elif win < need["hf"]:
        for m in FREQ_METRICS:
            why[m] = f"window {win:.0f} s shorter than the {need['hf']:.0f} s HF needs"
    else:
        ir = interp_rate(s)
        vlf, lf, hf = s.bands["vlf"], s.bands["lf"], s.bands["hf"]
        ulf = (0.0, vlf[0]) if vlf[0] > 0 else (0.0, 1e-6)
        vhf = (hf[1], min(hf[1] * 1.1, ir / 2 - 1e-3))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fd = nk.hrv_frequency({"RRI": x, "RRI_Time": t}, ulf=ulf, vlf=vlf, lf=lf, hf=hf, vhf=vhf,
                                      psd_method="welch", normalize=False, interpolation_rate=ir).iloc[0]
            pw = {"vlf": float(fd["HRV_VLF"]), "lf": float(fd["HRV_LF"]), "hf": float(fd["HRV_HF"])}
        except Exception as e:
            pw = {}
            for m in FREQ_METRICS:
                why[m] = f"spectrum failed: {e}"
        if pw:
            vals["hf"] = pw["hf"]
            if win >= need["lf"]:
                vals["lf"] = pw["lf"]
                vals["lf_hf"] = pw["lf"] / pw["hf"] if pw["hf"] > 0 else None
                tot = pw["lf"] + pw["hf"]
                vals["lf_nu"] = 100 * pw["lf"] / tot if tot > 0 else None
                vals["hf_nu"] = 100 * pw["hf"] / tot if tot > 0 else None
            else:
                for m in ("lf", "lf_hf", "lf_nu", "hf_nu", "lf_peak"):
                    why[m] = f"window {win:.0f} s shorter than the {need['lf']:.0f} s LF needs"
            if win >= need["vlf"]:
                vals["vlf"] = pw["vlf"]
                vals["total_power"] = pw["vlf"] + pw["lf"] + pw["hf"]
            else:
                for m in ("vlf", "total_power"):
                    why[m] = f"window {win:.0f} s shorter than the {need['vlf']:.0f} s VLF needs"
            pk = _welch_peaks(x, t, s, ir)
            vals["hf_peak"] = pk["hf_peak"]
            if vals["lf"] is not None:
                vals["lf_peak"] = pk["lf_peak"]
    # non-linear
    if n >= MIN_BEATS_POINCARE and d.size >= 2:
        dd = np.diff(x)[succ]
        sd1 = float(np.sqrt(0.5) * np.std(dd, ddof=1))
        sdnn = float(np.std(x, ddof=1))
        sd2 = float(np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 0.0)))
        vals["sd1"], vals["sd2"] = sd1, sd2
        vals["sd1_sd2"] = sd1 / sd2 if sd2 > 0 else None
    else:
        for m in ("sd1", "sd2", "sd1_sd2"):
            why[m] = f"fewer than {MIN_BEATS_POINCARE} NN intervals"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n >= MIN_BEATS_DFA1:
            try:
                vals["dfa_alpha1"] = float(nk.fractal_dfa(x, scale=np.arange(4, 16))[0])
            except Exception as e:
                why["dfa_alpha1"] = f"DFA failed: {e}"
        else:
            why["dfa_alpha1"] = f"fewer than {MIN_BEATS_DFA1} NN intervals"
        if n >= MIN_BEATS_DFA2:
            try:
                sc = np.unique(np.geomspace(16, min(128, n // 4), 12).astype(int))
                vals["dfa_alpha2"] = float(nk.fractal_dfa(x, scale=sc)[0])
            except Exception as e:
                why["dfa_alpha2"] = f"DFA failed: {e}"
        else:
            why["dfa_alpha2"] = f"fewer than {MIN_BEATS_DFA2} NN intervals"
        if n >= MIN_BEATS_SAMPEN:
            try:
                v = float(nk.entropy_sample(x, delay=1, dimension=2, tolerance=0.2 * np.std(x, ddof=1))[0])
                vals["sampen"] = v if math.isfinite(v) else None
                if vals["sampen"] is None:
                    why["sampen"] = "no template matches (undefined)"
            except Exception as e:
                why["sampen"] = f"sample entropy failed: {e}"
        else:
            why["sampen"] = f"fewer than {MIN_BEATS_SAMPEN} NN intervals"
    for m, v in vals.items():
        if v is not None and not (isinstance(v, float) and math.isfinite(v)):
            vals[m] = None
            why.setdefault(m, "not finite")
        elif v is None:
            why.setdefault(m, "not computed")
    return vals, why, n


# --------------------------------------------------------------------------- series on a grid

def grid_series(beats: Beats, segments: list, hz: float, max_gap_s: float) -> list[tuple]:
    """(time_s, variable, value, valid) rows: hr_bpm and rr_ms from NN intervals, linearly
    interpolated onto a regular grid inside each segment."""
    rows = []
    for k, seg in enumerate(segments):
        t0 = seg.t0
        t1 = seg.t0 + (seg.stop - seg.start - 1) / seg.fs
        g = np.arange(math.ceil(t0 * hz) / hz, t1 + 1e-9, 1.0 / hz)
        if g.size == 0:
            continue
        m = (beats.nn_t >= t0) & (beats.nn_t <= t1 + 1e-9)
        nt, nv = beats.nn_t[m], beats.nn_ms[m]
        if nt.size < 2:
            for gt in g:
                rows.append((gt, "hr_bpm", None, False))
                rows.append((gt, "rr_ms", None, False))
            continue
        rr = np.interp(g, nt, nv)
        j = np.searchsorted(nt, g, side="left")
        inside = (g >= nt[0]) & (g <= nt[-1])
        jj = np.clip(j, 1, nt.size - 1)
        gap = nt[jj] - nt[jj - 1]
        valid = inside & (gap <= max_gap_s)
        # also invalid when the interval itself is longer than max_gap (a pause)
        for gt, r, v in zip(g, rr, valid & inside):
            if gt < nt[0] or gt > nt[-1]:
                rows.append((gt, "hr_bpm", None, False))
                rows.append((gt, "rr_ms", None, False))
            else:
                rows.append((gt, "hr_bpm", 60000.0 / r, bool(v)))
                rows.append((gt, "rr_ms", r, bool(v)))
    return rows


def device_hr(reader: I.SeriesReader, hz: float, chunk_s: float = 600.0) -> tuple[np.ndarray, np.ndarray]:
    """A device heart-rate channel averaged in 1/hz bins centred on the grid times."""
    ts, vs = [], []
    for seg in reader.segments:
        step = max(int(chunk_s * seg.fs), 1)
        for a in range(seg.start, seg.stop, step):
            b = min(a + step, seg.stop)
            t = reader.times(seg, a, b)
            x = reader.read(a, b)
            gi = np.round(t * hz).astype(np.int64)
            ok = np.isfinite(x)
            if not ok.any():
                continue
            u, inv = np.unique(gi[ok], return_inverse=True)
            sums = np.bincount(inv, weights=x[ok])
            cnt = np.bincount(inv)
            ts.append(u / hz)
            vs.append(sums / cnt)
    if not ts:
        return np.zeros(0), np.zeros(0)
    t = np.concatenate(ts)
    v = np.concatenate(vs)
    # chunk edges may split a bin: merge duplicates
    u, inv = np.unique(np.round(t * hz).astype(np.int64), return_inverse=True)
    return u / hz, np.bincount(inv, weights=v) / np.bincount(inv)
