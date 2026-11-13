"""Figures: PNG, 150 dpi, recording id in the title, axes with units."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal as sps  # noqa: E402

DPI = 150
BAND_COLOURS = {"vlf": "#d9d9d9", "lf": "#9ecae1", "hf": "#fdd0a2"}


def tachogram(path: Path, rid: str, det, beats, events, device, settings, species: str, calibrated: bool) -> None:
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    if beats.nn_t.size:
        hr = 60000.0 / beats.nn_ms
        t = beats.nn_t
        # break the line across gaps so pauses and segment gaps are not bridged
        brk = np.nonzero(np.diff(t) > 3 * settings.rr_max_s)[0] + 1
        for part_t, part_h in zip(np.split(t, brk), np.split(hr, brk)):
            ax.plot(part_t, part_h, color="#08519c", lw=0.8)
        ax.plot([], [], color="#08519c", lw=0.8, label="heart rate from ECG (NN beats)")
    bad = beats.labels != "N"
    if bad.any():
        rr = beats.rr_ms[bad]
        ok = np.isfinite(rr) & (rr > 0)
        ax.scatter(beats.times[bad][ok], 60000.0 / rr[ok], s=10, marker="x", color="#cb181d",
                   label=f"ectopic or artifact beats ({int(bad.sum())})", zorder=3)
    if device is not None and device[0].size:
        dt, dv = device
        brk = np.nonzero(np.diff(dt) > 2.0)[0] + 1      # do not bridge gaps between trials
        for k, (pt, pv) in enumerate(zip(np.split(dt, brk), np.split(dv, brk))):
            ax.plot(pt, pv, color="#737373", lw=0.8, alpha=0.8,
                    label="device heart rate (hr channel)" if k == 0 else None)
    for i, ev in enumerate(events):
        ax.axvspan(ev.onset, ev.onset + max(ev.duration, 0.0), color="#fdae6b", alpha=0.35, lw=0,
                   label="stimulation" if i == 0 else None)
    ax.set_xlabel("time (s from recording start)")
    ax.set_ylabel("heart rate (bpm)")
    if beats.nn_t.size:
        lo, hi = np.percentile(60000.0 / beats.nn_ms, [0.5, 99.5])
        span = max(hi - lo, 5)
        ax.set_ylim(lo - 0.3 * span, hi + 0.3 * span)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(f"{rid}: heart rate ({species} preset, detector {settings.detector})", fontsize=11)

    ax2 = fig.add_subplot(gs[1])
    st = det.strip
    if st is not None:
        x = st["x"]
        y = x * 1000.0 if calibrated else x
        ax2.plot(st["t"], y, color="black", lw=0.7)
        py = st["peaks_x"] * (1000.0 if calibrated else 1.0)
        ax2.plot(st["peaks_t"], py, "o", ms=4, mfc="none", color="#cb181d", label="detected R-peaks")
        if det.xqrs_times is not None:
            xs = det.xqrs_times[(det.xqrs_times >= st["t"][0]) & (det.xqrs_times <= st["t"][-1])]
            if xs.size:
                ax2.vlines(xs, *ax2.get_ylim(), color="#3182bd", lw=0.6, alpha=0.6, label="second detector (XQRS)")
        ax2.legend(loc="upper right", fontsize=8, frameon=False)
        ax2.set_xlim(st["t"][0], st["t"][-1])
    ax2.set_xlabel("time (s from recording start)")
    ax2.set_ylabel("ECG (mV)" if calibrated else "ECG (uncalibrated)")
    ax2.set_title("first 10 s with beats" + (" (lead inverted before detection)" if det.inverted else ""), fontsize=9)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def hrv_figure(path: Path, rid: str, beats, settings, interp_hz: float) -> bool:
    x, t = beats.nn_ms, beats.nn_t
    if x.size < 20:
        return False
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    succ = np.abs(t[1:] - x[1:] / 1000.0 - t[:-1]) < 1e-3
    a1.scatter(x[:-1][succ], x[1:][succ], s=3, alpha=0.4, color="#08519c")
    lo, hi = np.percentile(x, [0.5, 99.5])
    pad = 0.1 * (hi - lo + 1)
    a1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#969696", lw=0.7)
    a1.set_xlim(lo - pad, hi + pad)
    a1.set_ylim(lo - pad, hi + pad)
    a1.set_aspect("equal")
    a1.set_xlabel("NN interval n (ms)")
    a1.set_ylabel("NN interval n+1 (ms)")
    a1.set_title("Poincare plot", fontsize=10)
    # spectrum of the longest stretch of NN intervals without a long gap
    brk = np.nonzero(np.diff(t) > 3 * settings.rr_max_s)[0] + 1
    parts = [(pt, px) for pt, px in zip(np.split(t, brk), np.split(x, brk))]
    pt, px = max(parts, key=lambda p: p[0][-1] - p[0][0] if p[0].size else 0)
    g = np.arange(pt[0], pt[-1], 1.0 / interp_hz) if pt.size > 4 else np.zeros(0)
    if g.size >= 64:
        y = sps.detrend(np.interp(g, pt, px))
        nper = min(g.size, int(interp_hz * 10.0 / max(settings.bands["lf"][0], 1e-3)))
        f, p = sps.welch(y, fs=interp_hz, nperseg=max(64, nper))
        top = settings.bands["hf"][1] * 1.3
        m = f <= top
        for k in ("vlf", "lf", "hf"):
            b0, b1 = settings.bands[k]
            a2.axvspan(b0, b1, color=BAND_COLOURS[k], alpha=0.6, lw=0, label=f"{k.upper()} {b0:g}-{b1:g} Hz")
        a2.plot(f[m], p[m], color="black", lw=0.9)
        a2.set_xlim(0, top)
        a2.legend(fontsize=8, frameon=False)
    a2.set_xlabel("frequency (Hz)")
    a2.set_ylabel("power (ms²/Hz)")
    a2.set_title(f"RR spectrum (Welch, longest stretch {pt[-1] - pt[0]:.0f} s)", fontsize=10)
    fig.suptitle(f"{rid}: heart rate variability", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return True
