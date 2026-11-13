"""
Species presets: the numbers that make R-peak detection and HRV work on a given species, each with
its source. Every value can be overridden by its own parameter (see main.read_params).

Sources (verified DOIs, full references in CITATION.cff):
  TF     Task Force of the ESC and NASPE 1996, Circulation 93:1043, doi:10.1161/01.CIR.93.5.1043
         (human frequency bands; "about 1 min is needed to assess the HF components while
         approximately 2 min are needed to address the LF component").
  PZ     Behar et al. 2018, PhysioZoo, Front Physiol 9:1390, doi:10.3389/fphys.2018.01390:
         Table 2 (RRmin, RRmax, typical HR and QRS duration), Table 3 (bands, pNNxx), and the
         PhysioZoo configuration files Config/qrs.<species>-ecg.yml (jqrs band-pass lcf/hcf and
         refractory period rp). Numbers only; no PhysioZoo code is used (it is GPL-3.0).
         PZ's mouse bands are taken from Thireau et al. 2008 (doi:10.1113/expphysiol.2007.040733).
  CARN   Carnevali et al. 2023, Front Physiol 14:1170320, doi:10.3389/fphys.2023.1170320:
         rat "LF, 0.2-0.75 Hz" and "HF, 0.75-2.5 Hz".
  POL    Poletto et al. 2011, Physiol Behav 103:188, doi:10.1016/j.physbeh.2011.01.019: pig bands
         from pharmacological blockade, "LF was 0.0-0.09 Hz, and HF was 0.09-2.0 Hz";
  SAL    Salomao et al. 2015, PLOS ONE 10:e0134387, doi:10.1371/journal.pone.0134387: the same
         pig bands split as "VLF:0.0-0.02Hz", "LF:0.02-0.09Hz", "HF:0.09-2.0Hz" (anaesthetised pigs).
  NK     NeuroKit2's own defaults for its "neurokit" R-peak detector (smoothwindow 0.1 s,
         avgwindow 0.75 s, mindelay 0.3 s), tuned on human ECG.
  PENN   Pennsieve choice where no published preset exists, stated as such (rat and pig detector
         timing, rat VLF lower edge, pig and rat pNNx threshold, series rate, gap and window
         lengths). Checked on the demo recordings (README "Validation").
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SPECIES = ("human", "pig", "dog", "rabbit", "rat", "mouse")

# Latin binomials (NWB Subject.species, SDS subjects.xlsx) and common synonyms -> preset name.
SPECIES_ALIASES = {
    "homo sapiens": "human", "human": "human", "man": "human", "woman": "human",
    "sus scrofa": "pig", "sus scrofa domesticus": "pig", "sus domesticus": "pig", "pig": "pig",
    "swine": "pig", "minipig": "pig", "yucatan minipig": "pig", "porcine": "pig",
    "canis familiaris": "dog", "canis lupus familiaris": "dog", "dog": "dog", "canine": "dog",
    "oryctolagus cuniculus": "rabbit", "rabbit": "rabbit",
    "rattus norvegicus": "rat", "rattus": "rat", "rat": "rat",
    "mus musculus": "mouse", "mouse": "mouse", "mice": "mouse",
}


@dataclass(frozen=True)
class Preset:
    name: str
    typical_hr_bpm: float          # expected resting heart rate (detector scaling, XQRS start value)
    qrs_s: float                   # typical QRS duration
    rr_min_s: float                # shortest plausible RR interval (range filter)
    rr_max_s: float                # longest plausible RR interval (range filter)
    filter_band_hz: tuple          # band-pass before detection (upper edge capped at 0.45 fs)
    refractory_s: float            # detector refractory period (NeuroKit2 "mindelay")
    smooth_s: float                # NeuroKit2 detector: gradient smoothing window
    avg_s: float                   # NeuroKit2 detector: moving-average window
    vlf_hz: tuple
    lf_hz: tuple
    hf_hz: tuple
    pnn_ms: float                  # pNNx threshold
    series_hz: float               # series.csv grid (conventions: 4 Hz large animals, 10 Hz rodents)
    max_gap_s: float               # series.csv valid=false across gaps between NN beats longer than this
    baseline_s: float              # per-stimulation windows
    after_s: float
    sources: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


def _scaled_nk(hr: float, qrs: float) -> tuple[float, float]:
    """NeuroKit2's detector windows scaled from its human values (0.1 s smoothing for a ~0.07 s
    QRS, 0.75 s moving average for ~75 bpm) to the species' QRS width and heart rate."""
    return round(0.1 * qrs / 0.07, 4), round(0.75 * 75.0 / hr, 4)


_PZ_DET = "PZ Table 2 (HR, QRS, RRmin, RRmax) and Config/qrs.<species>-ecg.yml (jqrs lcf/hcf, rp)"
_PZ_HRV = "PZ Table 3 (VLF/LF/HF bands, pNNxx)"
_PENN_COMMON = "PENN: series_hz per conventions.md; max_gap_s, baseline_s, after_s chosen for typical protocols"

PRESETS: dict[str, Preset] = {
    "human": Preset(
        "human", 75, 0.07, 0.28, 2.4, (3.0, 100.0), 0.25, 0.1, 0.75,
        (0.003, 0.04), (0.04, 0.15), (0.15, 0.4), 50, 4, 3.0, 30, 30,
        {"detector": _PZ_DET + "; smoothing/averaging windows: NK defaults",
         "bands": "TF (as tabulated in PZ Table 3, 'Human (Task force)')", "pnn": "TF (pNN50)",
         "other": _PENN_COMMON}),
    "dog": Preset(
        "dog", 109.5, 0.04, 0.25, 1.2, (3.0, 125.0), 0.17, *_scaled_nk(109.5, 0.04),
        (0.0033, 0.067), (0.067, 0.235), (0.235, 0.877), 25, 4, 2.0, 10, 10,
        {"detector": _PZ_DET + "; NK windows scaled to HR and QRS", "bands": _PZ_HRV, "pnn": _PZ_HRV,
         "other": _PENN_COMMON}),
    "rabbit": Preset(
        "rabbit", 264, 0.02, 0.14, 0.58, (3.0, 150.0), 0.088, *_scaled_nk(264, 0.02),
        (0.0033, 0.088), (0.088, 0.341), (0.341, 1.155), 17, 10, 1.0, 10, 10,
        {"detector": _PZ_DET + "; NK windows scaled to HR and QRS", "bands": _PZ_HRV, "pnn": _PZ_HRV,
         "other": _PENN_COMMON}),
    "mouse": Preset(
        "mouse", 608, 0.00718, 0.05, 0.24, (3.0, 300.0), 0.030, *_scaled_nk(608, 0.00718),
        (0.0056, 0.152), (0.152, 1.24), (1.24, 5.0), 5, 10, 0.5, 10, 10,
        {"detector": _PZ_DET + "; NK windows scaled to HR and QRS",
         "bands": _PZ_HRV + " (from Thireau et al. 2008)", "pnn": _PZ_HRV, "other": _PENN_COMMON}),
    "rat": Preset(
        # No published PhysioZoo rat preset: detector numbers are a Pennsieve choice between the
        # rabbit and mouse presets, wide enough for anaesthetised rats and for vagal bradycardia
        # (RR up to 1 s); checked on Discover dataset 655.
        "rat", 350, 0.015, 0.10, 1.0, (3.0, 200.0), 0.06, *_scaled_nk(350, 0.015),
        (0.0033, 0.2), (0.2, 0.75), (0.75, 2.5), 10, 10, 1.0, 10, 10,
        {"detector": "PENN (between PZ rabbit and mouse); NK windows scaled to HR and QRS",
         "bands": "CARN (LF 0.2-0.75 Hz, HF 0.75-2.5 Hz); VLF lower edge 0.0033 Hz as PZ dog/rabbit",
         "pnn": "PENN (between PZ rabbit 17 ms and mouse 5 ms)", "other": _PENN_COMMON}),
    "pig": Preset(
        # No published PhysioZoo pig preset: detector numbers are a Pennsieve choice close to the
        # dog preset, with RR up to 2 s for vagal bradycardia; checked on Discover dataset 349.
        "pig", 100, 0.06, 0.25, 2.0, (3.0, 100.0), 0.2, *_scaled_nk(100, 0.06),
        (0.0, 0.02), (0.02, 0.09), (0.09, 2.0), 50, 4, 3.0, 10, 10,
        {"detector": "PENN (close to PZ dog); NK windows scaled to HR and QRS",
         "bands": "POL (LF 0.0-0.09, HF 0.09-2.0 Hz) split into VLF/LF at 0.02 Hz as SAL",
         "pnn": "PENN (TF's 50 ms: pig heart rates are close to human)", "other": _PENN_COMMON}),
}


def resolve(name: str | None) -> str | None:
    """Preset name for a species string (preset name, Latin binomial or synonym), else None."""
    if not name:
        return None
    s = " ".join(str(name).strip().lower().replace("_", " ").split())
    if s in PRESETS:
        return s
    if s in SPECIES_ALIASES:
        return SPECIES_ALIASES[s]
    for k, v in SPECIES_ALIASES.items():
        if s.startswith(k):
            return v
    return None


def min_window_s(band: tuple, kind: str) -> float:
    """Shortest window for a band's power, from the Task Force rule for humans (about 1 min for HF
    with its 0.15 Hz lower edge = 10 cycles; about 2 min for LF with its 0.04 Hz lower edge = 4.8
    cycles), applied to each species' band edges. VLF needs 10 cycles of its upper edge (250 s in
    humans), so VLF is only reported from long windows."""
    lo, hi = band
    if kind == "hf":
        return 10.0 / lo
    if kind == "lf":
        return 4.8 / lo if lo > 0 else 4.8 / (hi / 4.5)
    return 10.0 / hi
