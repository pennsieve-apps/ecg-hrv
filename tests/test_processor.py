"""The synthetic fixture through the processor (no Docker): planted beats, ectopy, polarity,
stimulation bradycardia, gapped trial-by-trial data, the missing-ECG path and the contract."""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from processor import ecg as E
from processor import presets as P
from tests import make_fixture

ROOT = Path(__file__).resolve().parent.parent


def run(inp: Path, out: Path, **env) -> subprocess.CompletedProcess:
    e = {**os.environ, "INPUT_DIR": str(inp), "OUTPUT_DIR": str(out), "THREADS": "1", "PYTHONPATH": str(ROOT),
         "TMPDIR": str(out.parent / "tmp"), **{k: str(v) for k, v in env.items()}}
    return subprocess.run([sys.executable, "-m", "processor.main"], env=e, cwd=ROOT, capture_output=True, text=True,
                          timeout=600)


def rows(path: Path, delimiter=",") -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("fx")
    truth = make_fixture.write(d / "in")
    r = run(d / "in", d / "out")
    assert r.returncode == 0, r.stdout + r.stderr
    return d / "out" / "ecg-hrv", truth, r


def hrv_value(hrv, rid, scope, metric):
    v = [r["value"] for r in hrv if r["recording_id"] == rid and r["scope"] == scope and r["metric"] == metric]
    assert len(v) == 1
    return float(v[0]) if v[0] else None


def test_outputs_exist(result):
    out, _, _ = result
    for f in ("beats.csv", "series.csv", "hrv.csv", "qc.csv", "features.csv", "summary.json", "provenance.json"):
        assert (out / f).is_file(), f
    assert rows(out / "beats.csv")[0].keys() == {"recording_id", "beat", "time_s", "rr_ms", "label", "detector"}
    assert list(rows(out / "series.csv")[0]) == ["recording_id", "time_s", "variable", "value", "valid"]
    assert list(rows(out / "hrv.csv")[0]) == ["recording_id", "scope", "metric", "value", "n_beats", "window_s"]
    assert list(rows(out / "qc.csv")[0])[:7] == ["recording_id", "detector", "n_beats", "pct_ectopic", "pct_artifact",
                                                  "agreement_with_second_detector", "signal_quality"]


@pytest.mark.parametrize("rid", ["sub-h01_rec-rest", "sub-r01_rec-vns", "sub-p01_rec-grid"])
def test_r_peaks_match_truth(result, rid):
    out, truth, _ = result
    det = np.array([float(r["time_s"]) for r in rows(out / "beats.csv") if r["recording_id"] == rid])
    ref = np.array(truth[rid]["beats_s"])
    tol = 0.02
    tp = E.match(ref, det, tol)
    sens, ppv = tp / ref.size, tp / det.size
    # the pig trials are 25 s long: a beat within the first ~0.5 s of a trial may be missed
    assert sens >= (0.97 if rid.startswith("sub-p01") else 0.995), (rid, sens)
    assert ppv >= 0.995, (rid, ppv)


def test_ectopic_beat_labelled(result):
    out, truth, _ = result
    beats = [r for r in rows(out / "beats.csv") if r["recording_id"] == "sub-h01_rec-rest"]
    ect = [int(r["beat"]) - 1 for r in beats if r["label"] == "ectopic"]
    assert ect == [truth["sub-h01_rec-rest"]["ectopic_index"]]
    assert all(r["label"] == "N" for r in beats if r["label"] != "ectopic")


def test_inverted_lead_found(result):
    out, _, _ = result
    qc = {r["recording_id"]: r for r in rows(out / "qc.csv")}
    assert qc["sub-r01_rec-vns"]["inverted"] == "true"
    assert qc["sub-h01_rec-rest"]["inverted"] == "false"
    for rid in qc:
        assert float(qc[rid]["agreement_with_second_detector"]) > 0.97, rid


def test_species_presets_applied(result):
    out, _, _ = result
    prov = json.loads((out / "provenance.json").read_text())
    sp = {rid: r.get("species") for rid, r in prov["recordings"].items()}
    assert sp == {"sub-h01_rec-rest": "human", "sub-r01_rec-vns": "rat", "sub-p01_rec-grid": "pig",
                  "sub-h02_rec-noecg": "human"}
    assert prov["recordings"]["sub-r01_rec-vns"]["preset_used"]["hf_hz"] == [0.75, 2.5]
    assert set(prov["parameters"]["SPECIES_PRESETS"]) == {"human", "rat", "pig"}


def test_hrv_spectrum_finds_planted_rhythms(result):
    out, _, _ = result
    hrv = rows(out / "hrv.csv")
    rid = "sub-h01_rec-rest"
    assert hrv_value(hrv, rid, "recording", "hf_peak") == pytest.approx(0.25, abs=0.02)
    assert hrv_value(hrv, rid, "recording", "lf_peak") == pytest.approx(0.10, abs=0.02)
    assert hrv_value(hrv, rid, "recording", "mean_hr") == pytest.approx(65, abs=1.5)
    assert hrv_value(hrv, rid, "recording", "rmssd") > 10


def test_stimulation_bradycardia_grows_with_amplitude(result):
    out, truth, _ = result
    hrv = rows(out / "hrv.csv")
    rid = "sub-r01_rec-vns"
    drops = []
    for i, tr in enumerate(truth[rid]["trains"]):
        eid = f"stim-{i + 1:04d}"
        base = hrv_value(hrv, rid, f"{eid}:baseline", "mean_nn")
        dur = hrv_value(hrv, rid, f"{eid}:during", "mean_nn")
        drops.append(dur / base - 1)
        # the effect ramps in over 2 s of a 10 s train: mean increase is ~90 % of the planted one
        assert dur / base - 1 == pytest.approx(0.9 * tr["rr_increase"], rel=0.25)
    assert drops == sorted(drops)
    # 10 s windows: no LF/HF (needs >= 24 s for the rat LF band), with the reason in qc.csv
    assert hrv_value(hrv, rid, "stim-0001:during", "lf_hf") is None
    qc = {r["recording_id"]: r for r in rows(out / "qc.csv")}
    assert "shorter than" in qc[rid]["notes"]


def test_gapped_recording_and_device_hr(result):
    out, _, _ = result
    qc = {r["recording_id"]: r for r in rows(out / "qc.csv")}
    assert float(qc["sub-p01_rec-grid"]["hr_device_mae_bpm"]) < 3
    series = [r for r in rows(out / "series.csv") if r["recording_id"] == "sub-p01_rec-grid"]
    t = np.array([float(r["time_s"]) for r in series if r["variable"] == "hr_bpm"])
    assert not np.any((t > 26) & (t < 54))          # nothing written in the gap between trials
    assert {r["variable"] for r in series} == {"hr_bpm", "rr_ms", "hr_device_bpm"}
    hrv = rows(out / "hrv.csv")
    assert hrv_value(hrv, "sub-p01_rec-grid", "stim-0003:during", "mean_hr") is not None


def test_series_grid(result):
    out, _, _ = result
    s = [r for r in rows(out / "series.csv") if r["recording_id"] == "sub-r01_rec-vns" and r["variable"] == "hr_bpm"]
    t = np.array([float(r["time_s"]) for r in s])
    assert np.allclose(np.diff(t), 0.1)                 # rat preset: 10 Hz
    valid = [r for r in s if r["valid"] == "true"]
    assert len(valid) > 0.95 * len(s)
    assert all(r["value"] == "" for r in s if r["valid"] == "false" and float(r["time_s"]) < 0.1)


def test_missing_ecg_is_skipped_not_failed(result):
    out, _, r = result
    prov = json.loads((out / "provenance.json").read_text())
    rec = prov["recordings"]["sub-h02_rec-noecg"]
    assert rec["status"] == "skipped" and "no ECG" in rec["reason"]
    summ = json.loads((out / "summary.json").read_text())
    assert any("sub-h02_rec-noecg" in w for w in summ["warnings"])


def test_provenance_and_summary_contract(result):
    out, _, _ = result
    prov = json.loads((out / "provenance.json").read_text())
    assert list(prov) == ["app", "version", "variant", "tools", "parameters", "inputs", "started", "finished",
                          "wall_seconds", "peak_rss_mb", "threads", "recordings", "citations", "licence_notice"]
    assert prov["app"] == "ecg-hrv" and "neurokit2" in prov["tools"] and "wfdb" in prov["tools"]
    assert "10.3758/s13428-020-01516-y" in prov["citations"]
    assert "10.3389/fphys.2023.1170320" in prov["citations"]           # rat bands used
    summ = json.loads((out / "summary.json").read_text())
    assert set(summ) >= {"app", "version", "recordings", "figures", "tables", "warnings"}
    for f in summ["figures"]:
        assert (out / f["path"]).is_file() and len(f["caption"]) > 40
    assert isinstance(summ["recordings"]["sub-h01_rec-rest"]["mean_hr_bpm"], float)


def test_features(result):
    out, _, _ = result
    f = rows(out / "features.csv")
    names = {r["feature"] for r in f if r["recording_id"] == "sub-h01_rec-rest"}
    assert {"hrv_rmssd", "hrv_sdnn", "hrv_lf_hf", "ecg_n_beats"} <= names


def test_rmssd_matches_definition(result):
    out, _, _ = result
    beats = [r for r in rows(out / "beats.csv") if r["recording_id"] == "sub-r01_rec-vns"]
    rr = np.array([float(r["rr_ms"]) for r in beats[1:]])
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    hrv = rows(out / "hrv.csv")
    assert hrv_value(hrv, "sub-r01_rec-vns", "recording", "rmssd") == pytest.approx(rmssd, rel=0.01)


def test_only_missing_signals_exits_zero(tmp_path):
    make_fixture.write(tmp_path / "fx")
    inp = tmp_path / "in"
    (inp / "sub-h02").mkdir(parents=True)
    shutil.copy(tmp_path / "fx/physio/sub-h02/sub-h02_rec-noecg.nwb", inp / "sub-h02")
    r = run(inp, tmp_path / "out")
    assert r.returncode == 0, r.stdout
    out = tmp_path / "out/ecg-hrv"
    assert len(rows(out / "beats.csv")) == 0 and len(rows(out / "hrv.csv")) == 0
    summ = json.loads((out / "summary.json").read_text())
    assert any("no recording has an ECG channel" in w for w in summ["warnings"])


def test_bare_nwb_and_merged_layout(tmp_path):
    """NWB files without physio-import's tables, events.tsv merged at the root, species by parameter."""
    make_fixture.write(tmp_path / "fx")
    inp = tmp_path / "in"
    inp.mkdir()
    shutil.copy(tmp_path / "fx/physio/sub-r01/sub-r01_rec-vns.nwb", inp)
    shutil.copy(tmp_path / "fx/events/stim-events/events.tsv", inp)
    r = run(inp, tmp_path / "out", SPECIES="rat", FIGURES="false", SECOND_DETECTOR="false")
    assert r.returncode == 0, r.stdout
    out = tmp_path / "out/ecg-hrv"
    qc = rows(out / "qc.csv")
    assert qc[0]["recording_id"] == "sub-r01_rec-vns" and qc[0]["agreement_with_second_detector"] == ""
    scopes = {x["scope"] for x in rows(out / "hrv.csv")}
    assert "stim-0003:during" in scopes
    assert not (out / "figures").exists()


def test_no_input_exits_one(tmp_path):
    (tmp_path / "in").mkdir()
    r = run(tmp_path / "in", tmp_path / "out")
    assert r.returncode == 1


@pytest.mark.parametrize("env", [{"SPECIES": "cat"}, {"DETECTOR": "magic"}, {"LF_BAND": "0.2-0.1"},
                                 {"CHUNK_S": "1"}, {"FIGURES": "maybe"}])
def test_bad_parameters_exit_two(tmp_path, env):
    (tmp_path / "in").mkdir()
    assert run(tmp_path / "in", tmp_path / "out", **env).returncode == 2


def test_presets_well_formed():
    for name, p in P.PRESETS.items():
        assert p.vlf_hz[1] == p.lf_hz[0] and p.lf_hz[1] == p.hf_hz[0], name
        assert p.rr_min_s < 60 / p.typical_hr_bpm < p.rr_max_s, name
        assert p.refractory_s < p.rr_min_s, name
        assert {"detector", "bands", "pnn", "other"} <= set(p.sources), name
    assert P.resolve("Rattus norvegicus") == "rat" and P.resolve("Sus scrofa domesticus") == "pig"
    assert P.resolve("Homo sapiens") == "human" and P.resolve("zebrafish") is None
    # Task Force: about 1 min for HF and 2 min for LF in humans
    assert P.min_window_s(P.PRESETS["human"].hf_hz, "hf") == pytest.approx(66.7, abs=0.1)
    assert P.min_window_s(P.PRESETS["human"].lf_hz, "lf") == pytest.approx(120, abs=0.1)


def test_fixture_is_valid_nwb(tmp_path):
    pynwb = pytest.importorskip("pynwb")
    make_fixture.write(tmp_path)
    for p in (tmp_path / "physio").rglob("*.nwb"):
        with pynwb.NWBHDF5IO(str(p), "r") as io:
            nwb = io.read()
            assert nwb.identifier == p.stem


def test_app_yml_matches_code():
    yaml = pytest.importorskip("yaml")
    from processor import main as M
    app = yaml.safe_load((ROOT / "app.yml").read_text())
    params = {p["name"]: p for p in app["parameters"]}
    assert set(params) == set(M.DEFAULTS)
    for name, p in params.items():
        assert str(p["defaultValue"]).lower() == M.DEFAULTS[name].lower(), name
    assert set(params["DETECTOR"]["validValues"]) == set(E.DETECTORS)
    assert set(params["SPECIES"]["validValues"]) == {"auto", *P.SPECIES}
    assert (app["runtime"]["cpu"], app["runtime"]["memory"]) == (1024, 2048)
