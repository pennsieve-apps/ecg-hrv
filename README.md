# ecg-hrv: heart rate and heart rate variability from the ECG, for humans and animals

ecg-hrv is built on **NeuroKit2**, developed by Dominique Makowski and colleagues
([repository](https://github.com/neuropsychology/NeuroKit),
[paper](https://doi.org/10.3758/s13428-020-01516-y)): its ECG filters, R-peak detectors, HRV
functions and its implementation of the beat-classification method of **Jukka A. Lipponen and Mika
P. Tarvainen** ([paper](https://doi.org/10.1080/03091902.2019.1640306)). The second R-peak detector
is XQRS from **WFDB for Python**, from the MIT Laboratory for Computational Physiology
([repository](https://github.com/MIT-LCP/wfdb-python), [software DOI](https://doi.org/10.13026/9njx-6322)).
The HRV measures follow the **Task Force of the European Society of Cardiology and the North
American Society of Pacing and Electrophysiology** ([standard](https://doi.org/10.1161/01.CIR.93.5.1043)),
and the animal presets take their numbers from **PhysioZoo** by Joachim A. Behar and colleagues
([paper](https://doi.org/10.3389/fphys.2018.01390); numbers only, no code), from **Luca Carnevali and
colleagues** for rats ([paper](https://doi.org/10.3389/fphys.2023.1170320)) and from **Rosangela Poletto
and colleagues** and **Edgard Salomão and colleagues** for pigs
([paper](https://doi.org/10.1016/j.physbeh.2011.01.019), [paper](https://doi.org/10.1371/journal.pone.0134387)).
Pennsieve packages these methods as an app; it did not develop them. The demo data are the
**Fantasia Database** of Nikhil Iyengar and colleagues and the **Autonomic Aging** database of Andy
Schumann and Karl-Jürgen Bär (PhysioNet), and Pennsieve Discover datasets from **Oliver Armitage and
colleagues** (pig vagus nerve stimulation, 349), **Shane A. Bender and colleagues** (rat heart-rate
control, 655) and **M. Khawar Ali and colleagues** (human ECG, 406).

One step of two workflows:

- **Cardiovascular Response to Nerve Stimulation:** `physio-import → stim-events → {ecg-hrv, bp-resp} → stim-dose-response → physio-report`
- **Heart Rate Variability and Baroreflex:** `physio-import → {ecg-hrv, bp-resp} → physio-report`

> **Research output.** Heart rate, beat labels and HRV are computed automatically from the ECG for
> research. They are not a clinical ECG or rhythm interpretation, and an "ectopic" label is an
> RR-interval pattern, not a diagnosis.

## What it does

**For the physiologist.** For every recording with an ECG, the app finds each heartbeat (R-peak),
labels beats that do not fit the rhythm (premature "ectopic" beats, missed or spurious detections),
and from the normal-to-normal (NN) intervals computes the heart rate and the standard heart rate
variability (HRV) measures: SDNN, RMSSD and pNN50 in the time domain; VLF, LF and HF power, LF/HF
and the peak frequencies in the frequency domain; and the Poincaré SD1/SD2, detrended fluctuation
analysis (DFA α1, α2) and sample entropy. It does this for the whole recording and, when the
stimulation trains of stim-events are linked, for a baseline window before each train, during it,
and a window after it. The numbers that make this work in a given species (how fast the heart can
beat, which frequencies count as "respiratory" HF and "sympathetic/baroreflex" LF) come from a
species preset: human, pig, dog, rabbit, rat or mouse. A device heart-rate channel (e.g. the
monitor's own HR trace) is passed through and compared with the ECG-derived heart rate. Two figures
per recording show the heart rate over time with the labelled beats and an ECG excerpt with the
detected beats, so you can see at a glance whether detection worked.

**For the data scientist.** Per recording, one ECG channel (`channels.csv` type `ecg`, else name
heuristics) is read from NWB with h5py in `CHUNK_S` windows (plus 2 s on each side; peaks are kept
in each window's core), band-passed (zero-phase Butterworth, species band) and, when a 50/60 Hz
line stands out of the spectrum, notched. NeuroKit2 `ecg_findpeaks` finds R-peaks; wfdb XQRS runs
on the same windows (high-passed at 0.5 Hz) and the two are matched one to one within
min(50 ms, 0.3 × RRmin). Lead polarity follows NeuroKit2's `ecg_invert` rule. Beats are labelled
with NeuroKit2's `signal_fixpeaks(method="Kubios")` classes on the detected beats (not corrected
beats), plus a range filter; NN intervals are those between two `N` beats of the same contiguous
segment (gapped NWB `timestamps` make segments). HRV uses NeuroKit2 `hrv_time` and `hrv_frequency`
(Welch, `normalize=False`, so powers are in ms²) on `{"RRI", "RRI_Time"}` input, and NeuroKit2
`fractal_dfa` and `entropy_sample` with the settings `hrv_nonlinear` uses. Heart rate and RR are
also resampled to a regular `SERIES_HZ` grid by linear interpolation between NN beats
(`series.csv`), the shared shape that stim-dose-response reads.

### Method details

- **Filter and detector.** Band-pass and refractory period from the preset (PhysioZoo's jqrs
  band-pass and refractory values for human, dog, rabbit, mouse). The `neurokit` detector's
  smoothing and averaging windows are NeuroKit2's human values (0.1 s, 0.75 s) scaled by the
  species' QRS width and heart rate. The other detectors (`pantompkins1985`, `hamilton2002`,
  `elgendi2010`, `engzeemod2012`, `kalidas2017`, `rodrigues2021`, as NeuroKit2 names them) keep
  their published human constants; only the refractory period is enforced on their output. Use them
  for human ECG.
- **Mains.** `POWERLINE_HZ=auto` notches 50 or 60 Hz and harmonics (IIR notch, Q 30) when that line
  is more than 20 times the median power density of 35-45 Hz in the first 120 s. This rescued
  Autonomic Aging record 0002, whose ECG carries 50 Hz interference as large as the QRS.
- **Beat labels.** `ectopic`: NeuroKit2 reports an ectopic pattern at the beat after the premature
  one; the app labels the premature beat of the pair (the shorter interval). `artifact`: NeuroKit2's
  missed, extra and long/short classes, and RR intervals outside the preset's RR range. A flag is
  kept only when the RR interval differs from the median of the 11 around it by more than
  `ARTIFACT_MIN_PCT` (10 %): Lipponen and Tarvainen's thresholds scale with the beat-to-beat
  variability, and in the very regular rhythm of anaesthetised animals (RR changes of 1-2 ms) they
  labelled 2-5 % of ordinary rat beats as ectopic (dataset 655) before this rule.
- **Minimum windows.** An HRV metric is left empty, with the reason in `qc.csv` `notes`, when its
  window is too short. Time domain (SDNN, RMSSD, pNN50, pNNx): at least 10 NN intervals (mean NN and
  mean heart rate: 2). Frequency domain: NN intervals covering at least 80 % of the window, and the
  Task Force rule applied to each species' band edges: HF needs 10 cycles of its lower edge (1 min
  for the human 0.15 Hz), LF 4.8 cycles of its lower edge (2 min for 0.04 Hz), VLF 10 cycles of its
  upper edge (250 s for 0.04 Hz). Poincaré: 20 NN intervals; DFA α1 (boxes 4-15 beats): 60; DFA α2
  (boxes 16 to min(128, n/4)): 256; sample entropy (m = 2, r = 0.2 SD): 200.
- **Spectra.** NN intervals resampled at max(4 Hz, 2.25 × 1.1 × the HF upper edge) (PhysioZoo's
  rule), Welch. LF/HF normalised units are LF/(LF+HF) and HF/(LF+HF) × 100, as the Task Force
  defines them (not NeuroKit2's LF/total power).

### Species presets

Every value can be overridden by its own parameter. `provenance.json` lists the preset used for each
recording. Sources: **TF** Task Force 1996; **PZ** Behar et al. 2018 (PhysioZoo) Table 2, Table 3
and the numbers in its `Config/qrs.<species>-ecg.yml`; PZ's mouse bands come from Thireau et al.
2008; **CARN** Carnevali et al. 2023; **POL/SAL** Poletto et al. 2011 (pig LF 0.0-0.09 Hz, HF
0.09-2.0 Hz from pharmacological blockade in conscious pigs) as split into VLF/LF at 0.02 Hz by
Salomão et al. 2015 in anaesthetised pigs; **NK** NeuroKit2 defaults; **PENN** a Pennsieve choice
where nothing is published, checked on the demo data.

| | human | dog | rabbit | mouse | rat | pig |
|---|---|---|---|---|---|---|
| typical HR (bpm) | 75 (PZ) | 109.5 (PZ) | 264 (PZ) | 608 (PZ) | 350 (PENN) | 100 (PENN) |
| RR range (s) | 0.28-2.4 (PZ) | 0.25-1.2 (PZ) | 0.14-0.58 (PZ) | 0.05-0.24 (PZ) | 0.10-1.0 (PENN) | 0.25-2.0 (PENN) |
| band-pass (Hz) | 3-100 (PZ) | 3-125 (PZ) | 3-150 (PZ) | 3-300 (PZ) | 3-200 (PENN) | 3-100 (PENN) |
| refractory (s) | 0.25 (PZ) | 0.17 (PZ) | 0.088 (PZ) | 0.030 (PZ) | 0.06 (PENN) | 0.2 (PENN) |
| detector smoothing / averaging (s) | 0.1 / 0.75 (NK) | 0.057 / 0.514 | 0.029 / 0.213 | 0.010 / 0.093 | 0.021 / 0.161 | 0.086 / 0.563 |
| VLF (Hz) | 0.003-0.04 (TF) | 0.0033-0.067 (PZ) | 0.0033-0.088 (PZ) | 0.0056-0.152 (PZ) | 0.0033-0.2 (CARN; lower edge as PZ) | 0.0-0.02 (SAL) |
| LF (Hz) | 0.04-0.15 (TF) | 0.067-0.235 (PZ) | 0.088-0.341 (PZ) | 0.152-1.24 (PZ) | 0.2-0.75 (CARN) | 0.02-0.09 (POL, SAL) |
| HF (Hz) | 0.15-0.4 (TF) | 0.235-0.877 (PZ) | 0.341-1.155 (PZ) | 1.24-5 (PZ) | 0.75-2.5 (CARN) | 0.09-2.0 (POL) |
| shortest window HF / LF / VLF (s) | 67 / 120 / 250 | 43 / 72 / 149 | 29 / 55 / 114 | 8 / 32 / 66 | 13 / 24 / 50 | 111 / 240 / 500 |
| pNNx threshold (ms) | 50 (TF) | 25 (PZ) | 17 (PZ) | 5 (PZ) | 10 (PENN) | 50 (PENN) |
| series grid (Hz) | 4 | 4 | 10 | 10 | 10 | 4 |
| max gap in series (s) | 3 | 2 | 1 | 0.5 | 1 | 3 |
| baseline / after window (s) | 30 / 30 | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 |

Detector smoothing and averaging windows for animals are NK's human values scaled by QRS width
(PZ's typical QRS duration; rat 0.015 s and pig 0.06 s are PENN) and by heart rate. Series grid,
gap and window lengths are PENN (series rates as conventions.md asks: 4 Hz for humans and large
animals, 10 Hz for rodents).

## Ports

| Port | Direction | What |
|---|---|---|
| `physio` | input (required) | physio-import's `physio` output: `recordings.csv`, `channels.csv`, NWB per recording; or any NWB 2.x files with ECG |
| `events` | input (optional) | stim-events' `events.tsv` (rows with `trial_type` `stimulation` or `block`); physio-import's own `events.tsv` markers are ignored |
| `hrv` | output | `ecg-hrv/` with the tables, figures, `summary.json` and `provenance.json` below |

Files are found by name and content anywhere under `INPUT_DIR` (following symlinks), so the ports
also work merged side by side.

## Parameters

| Name | Default | What it does |
|---|---|---|
| `SPECIES` | `auto` | `auto` (from `recordings.csv`, else the NWB `Subject.species`, else human), `human`, `pig`, `dog`, `rabbit`, `rat`, `mouse` |
| `DETECTOR` | `neurokit` | NeuroKit2 R-peak detector (list above) |
| `ECG_CHANNEL` | `auto` | channel to analyse; `auto` = first ECG not marked flat/saturated. A second ECG lead is used for `agreement_other_lead` |
| `INVERT` | `auto` | inverted lead: `auto`, `true`, `false` |
| `ARTIFACT_CORRECTION` | `kubios` | beat labelling: `kubios` (Lipponen and Tarvainen classes + range), `range`, `none` |
| `ARTIFACT_MIN_PCT` | `10` | keep a Kubios flag only above this RR deviation from the local median (%) |
| `SECOND_DETECTOR` | `true` | run XQRS for the agreement in `qc.csv` |
| `FILTER_BAND` | `auto` | detector band-pass, e.g. `3-100` |
| `REFRACTORY_S` | `0` (preset) | shortest R-R time for the detector |
| `RR_RANGE_S` | `auto` | plausible RR range, e.g. `0.28-2.4` |
| `VLF_BAND`, `LF_BAND`, `HF_BAND` | `auto` | frequency bands, e.g. `0.04-0.15` |
| `PNN_THRESHOLD_MS` | `0` (preset) | threshold of `pnnx` |
| `SERIES_HZ` | `0` (preset) | `series.csv` grid rate |
| `MAX_GAP_S` | `0` (preset) | `series.csv` `valid=false` across longer gaps between NN beats |
| `BASELINE_S`, `AFTER_S` | `0` (preset) | per-stimulation windows (cut at neighbouring trains) |
| `POWERLINE_HZ` | `auto` | mains notch: `auto`, `0`, `50`, `60` |
| `FIGURES` | `true` | two figures per recording |
| `CHUNK_S` | `300` | seconds of ECG processed at a time |
| `THREADS` | `0` | BLAS threads (0 = from the cgroup CPU quota) |

## Outputs (under `OUTPUT_DIR/ecg-hrv/`)

| File | Columns / content |
|---|---|
| `beats.csv` | `recording_id, beat, time_s, rr_ms` (interval ending at the beat; empty for the first beat of a segment), `label` (`N`, `ectopic`, `artifact`), `detector` |
| `series.csv` | `recording_id, time_s, variable, value, valid`; variables `hr_bpm`, `rr_ms` and, with a device heart-rate channel, `hr_device_bpm` (bin means). Grid points only inside recorded segments |
| `hrv.csv` | `recording_id, scope` (`recording` or `<event_id>:baseline/during/after`), `metric, value, n_beats, window_s`. Metrics and units: `mean_nn` ms, `mean_hr` bpm, `sdnn` ms, `rmssd` ms, `sdsd` ms, `pnn50` %, `pnnx` % (species threshold), `cvnn`, `vlf`/`lf`/`hf`/`total_power` ms², `lf_hf`, `lf_nu`/`hf_nu` %, `lf_peak`/`hf_peak` Hz, `sd1`/`sd2` ms, `sd1_sd2`, `dfa_alpha1`, `dfa_alpha2`, `sampen`. Empty value = window too short (reason in `qc.csv` notes) |
| `qc.csv` | `recording_id, detector, n_beats, pct_ectopic, pct_artifact, agreement_with_second_detector, signal_quality` (mean NeuroKit2 `averageQRS` template match, 0-1), then `channel, sampling_rate_hz, species, inverted, mean_hr_bpm, agreement_other_lead, hr_device_mae_bpm, hr_device_r, notes` |
| `features.csv` | `recording_id, channel, feature, value`: `hrv_<metric>` (whole recording), `ecg_n_beats`, `ecg_pct_ectopic`, `ecg_pct_artifact`, `ecg_detector_agreement` |
| `figures/<recording_id>_heart-rate.png` | heart rate over the recording (labelled beats, device HR, stimulation trains) and the first 10 s of ECG with the detected beats |
| `figures/<recording_id>_hrv.png` | Poincaré plot and RR spectrum with the species bands |
| `summary.json`, `provenance.json` | headline numbers per recording, captioned figures and tables, warnings, a research-output notice; parameters, presets, tool versions, timings and citations |

A recording without ECG is listed in `provenance.json` as `skipped` with the reason and a warning;
the step still exits 0. Exit 1 only when there is no input or every recording with ECG failed;
exit 2 for bad parameters.

## Runtime and results on the demo data

Measured with the final image in Docker (**amd64 emulation on an M-series Mac**; `--cpus 1
--memory 1g`, read-only file system, uid 65534, HOME/TMPDIR under `/tmp`, as on Lambda). Peak
memory is the container's cgroup `memory.peak`. Inputs: NWB files written to the physio-import
contract by a stand-in converter, because physio-import was being built at the same time (see
"Validation").

| Demo | Recordings | Wall time | Peak memory |
|---|---|---|---|
| Fantasia (human, 250 Hz, ~2 h each) | 6 | 58 s (~9 s per recording) | 321 MB |
| Autonomic Aging (human, 1 kHz, 15-32 min, 2 leads) | 3 | 40 s | 334 MB |
| Discover 655 (rat, 1 kHz, 20-28 min) | 3 | 26 s | 330 MB |
| Discover 349 (pig, 1 kHz, 20 trials, trial by trial) | 1 | 8 s | 279 MB |
| Discover 406 (human, uncalibrated WAV, 500 Hz, 5-17 min) | 20 | 46 s | 297 MB |

Memory does not grow with recording length (the ECG is read in `CHUNK_S` windows); beats and the
series grid are kept in memory (about 100 MB per day of rodent ECG). `runtime` asks for 1 vCPU and
2048 MiB (the smallest Fargate pair with one vCPU); this fits Lambda. Image: ~430 MB uncompressed,
~140 MB compressed.

## Validation

All numbers from the image above. Scripts: `validation/validate.py`.

- **R-peaks vs reference annotations (PhysioNet Fantasia, 6 records, 44,164 reference beats).**
  Sensitivity 0.9995-1.0000 and positive predictivity 0.9947-0.9999 within ±50 ms
  (`wfdb.processing.compare_annotations`), per record: f1y01 0.9999/0.9999, f1o01 0.9997/0.9947,
  f2y01 0.9998/0.9995, f2o01 1.0000/0.9999, f2y02 1.0000/0.9989, f2o02 0.9995/0.9984. The beat
  labels found 21 of 21 annotated non-normal beats in f2o02 and 75 of 125 in f2o01, but only 7 of
  64 and 3 of 64 in f2y01 and f2y02: labelling uses RR intervals only, so a beat that is
  morphologically abnormal but not premature stays `N`.
- **Two detectors.** NeuroKit2 and XQRS agree on 99.7-100 % of beats on every Fantasia, Aging and
  rat recording and on 98.1-100 % of the 406 recordings.
- **Two leads (Autonomic Aging 0001-0003).** Beats from ECG1 and ECG2 agree on 99.89-99.95 %.
  Record 0002 needed the automatic 50 Hz notch (without it, 433 instead of 2148 beats).
- **Rat (Discover 655, rat preset).** 6,052-8,258 beats per trial, 0.4-0.7 % artifact, detector
  agreement 0.997-0.998. The app's heart rate correlates with the authors' heart rate
  (`derivative/HR`, normalised, 0.5 s steps) with Pearson r = 0.917, 0.845 and 0.876 for the gain
  2, 5 and 10 trials at the best time offset (123.5, 259.5, 245 s; the authors' traces start at the
  controller start). The HF peak sits at 1.0 Hz in all three trials (the ventilation rate).
- **Pig (Discover 349, pig preset).** 430 beats over 20 trials, 0.5 % artifact; mean absolute
  difference from the acquisition device's heart-rate channel 0.82 bpm (r = 0.92 on the 4 Hz grid).
  Per-train changes in mean heart rate (during minus baseline) range from -7.2 bpm (2.5 mA, 30 Hz)
  to +6.5 bpm (0.5 mA, 20 Hz), the sign change with amplitude that stim-dose-response's neural
  fulcrum analysis looks for. Stimulation onsets were taken as 10 s into each trial for this check
  (the dataset's description); the workflow takes them from stim-events.
- **HRV against published values.** Discover 406's own HRV results are not in the demo data (the
  worksheets are scanned forms and the analysis files are in a vendor format), so no comparison was
  possible. Plausibility: Fantasia RMSSD 46-73 ms in the three young subjects and 15-29 ms in the
  three elderly (the age effect the database was built to show); 406 resting RMSSD 35-91 ms with
  HF peaks at 0.15-0.31 Hz.
- **Synthetic fixture (tests).** Every planted beat found within 20 ms in the human and rat
  recordings (≥ 97 % in 25 s pig trials), the one planted premature beat labelled `ectopic` and no
  other beat labelled, the inverted rat lead detected, HF and LF peaks at the planted 0.25 and
  0.10 Hz, the planted 10/20/35 % bradycardia recovered per train in order, device heart rate within
  3 bpm, and the recording without ECG skipped with exit 0.

## Limitations

- **Not a rhythm analysis.** Beat labels come from RR intervals only (no QRS morphology): bundle
  branch block, paced beats or non-premature ventricular beats are labelled `N`; atrial
  fibrillation makes most HRV measures meaningless and the app does not detect it.
- **Species.** Validated on human (Fantasia, Aging, 406), rat (655) and pig (349) ECG. The dog,
  rabbit and mouse presets use PhysioZoo's published numbers but were not run on real recordings
  here. Rat and pig detector settings are a Pennsieve choice. For other species set the bands and
  ranges by parameter.
- **Anaesthesia and ventilation.** In anaesthetised, mechanically ventilated animals the HF peak is
  the ventilator rate and LF/VLF power is low and often not autonomic in origin; vagal blockade,
  anaesthetic depth and stimulation all change HRV. LF/HF as a "sympathovagal balance" is contested
  even in humans.
- **Short windows.** Per-stimulation windows of 5-30 s give mean heart rate and at most time-domain
  variability; spectral measures need the minimum windows above and are left empty otherwise.
  Metrics from windows of different length are not comparable.
- **Gapped recordings.** For trial-by-trial data the recording-level spectrum interpolates across
  gaps between trials (noted in `qc.csv`); use the per-stimulation windows instead.
- **Uncalibrated ECG** (e.g. 406's WAV files): timing measures only; amplitudes are not reported.
- **Detectors other than `neurokit`** keep NeuroKit2's human constants and are not suited to
  rodent heart rates.

## Licences

Apache-2.0 (this app). Bundled components and their licences: `THIRD_PARTY.md` (NeuroKit2 and WFDB
for Python are MIT; NumPy, SciPy, pandas, h5py, scikit-learn are BSD-3-Clause). PhysioZoo's preset
values are used as published numbers; its GPL-3.0 code is not used.

Demo data licences: Fantasia ODC-By 1.0; Autonomic Aging ODbL 1.0; Discover 349, 655 and 406 CC BY 4.0.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author
lists in CITATION.cff.

- Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., Schölzel, C., &
  Chen, S. H. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing.
  *Behavior Research Methods* 53, 1689-1696. doi:10.3758/s13428-020-01516-y, PMID 33528817.
- Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability time
  series artefact correction using novel beat classification. *Journal of Medical Engineering &
  Technology* 43, 173-181. doi:10.1080/03091902.2019.1640306, PMID 31314618.
- Task Force of the European Society of Cardiology and the North American Society of Pacing and
  Electrophysiology (1996). Heart Rate Variability. *Circulation* 93, 1043-1065.
  doi:10.1161/01.CIR.93.5.1043.
- Xie, C., McCullum, L., Johnson, A., Pollard, T., Gow, B., & Moody, B. (2023). Waveform Database
  Software Package (WFDB) for Python (version 4.1.0). PhysioNet. doi:10.13026/9njx-6322; with
  Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101,
  e215. doi:10.1161/01.CIR.101.23.e215, PMID 10851218.
- Species presets: Behar, J. A. et al. (2018). PhysioZoo: A Novel Open Access Platform for Heart
  Rate Variability Analysis of Mammalian Electrocardiographic Data. *Frontiers in Physiology* 9,
  1390. doi:10.3389/fphys.2018.01390, PMID 30337883. Thireau, J. et al. Heart rate variability in
  mice: a theoretical and practical guide. *Experimental Physiology* 93, 83-94.
  doi:10.1113/expphysiol.2007.040733, PMID 17911354. Carnevali, L. et al. (2023). Sex differences in
  heart rate and heart rate variability in rats: Implications for translational research.
  *Frontiers in Physiology* 14, 1170320. doi:10.3389/fphys.2023.1170320, PMID 37035663. Poletto, R.
  et al. (2011). Identification of low and high frequency ranges for heart rate variability and
  blood pressure variability analyses using pharmacological autonomic blockade with atropine and
  propranolol in swine. *Physiology & Behavior* 103, 188-196. doi:10.1016/j.physbeh.2011.01.019,
  PMID 21281655. Salomão, E. et al. (2015). Heart Rate Variability Analysis in an Experimental Model
  of Hemorrhagic Shock and Resuscitation in Pigs. *PLOS ONE* 10, e0134387.
  doi:10.1371/journal.pone.0134387, PMID 26247476.
- Optional detector: Pan, J., & Tompkins, W. J. (1985). A Real-Time QRS Detection Algorithm. *IEEE
  Transactions on Biomedical Engineering* BME-32, 230-236. doi:10.1109/TBME.1985.325532, PMID
  3997178. The other detectors are cited in NeuroKit2's `ecg_peaks` documentation.
- HRV metrics and norms: Shaffer, F., & Ginsberg, J. P. (2017). An Overview of Heart Rate
  Variability Metrics and Norms. *Frontiers in Public Health* 5, 258. doi:10.3389/fpubh.2017.00258,
  PMID 29034226.
- SciPy: Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in
  Python. *Nature Methods* 17, 261-272. doi:10.1038/s41592-019-0686-2, PMID 32015543.
- NWB: Rübel, O. et al. (2022). The Neurodata Without Borders ecosystem for neurophysiological data
  science. *eLife* 11, e78362. doi:10.7554/eLife.78362, PMID 36193886.
- Demo data: Iyengar, N., Peng, C-K., Morin, R., Goldberger, A. L., & Lipsitz, L. A. Fantasia
  Database. PhysioNet. doi:10.13026/C2RG61; Iyengar, N. et al. (1996). Age-related alterations in
  the fractal scaling of cardiac interbeat interval dynamics. *American Journal of
  Physiology-Regulatory, Integrative and Comparative Physiology* 271, R1078-R1084.
  doi:10.1152/ajpregu.1996.271.4.R1078, PMID 8898003. Schumann, A., & Bär, K. (2021). Autonomic
  Aging: A dataset to quantify changes of cardiovascular autonomic function during healthy aging.
  PhysioNet. doi:10.13026/2hsy-t491; Schumann, A., & Bär, K.-J. (2022). *Scientific Data* 9, 95.
  doi:10.1038/s41597-022-01202-y, PMID 35322044. Armitage, O. et al. (2023). Influence of vagus nerve
  stimulation on cardiac activity in pigs. Pennsieve Discover. doi:10.26275/dm5l-tspb. Bender, S. A.
  et al. (2026). Single-electrode bidirectional control of heart rate via vagus nerve modulation in
  rat model. Pennsieve Discover. doi:10.26275/5cwo-mkpz. Ali, M. K. et al. (2024). Best parameters of
  heart rate variability for assessing autonomic responses to brief rectal distention in patients
  with irritable bowel syndrome. Pennsieve Discover. doi:10.26275/nung-b9bm.

## Acknowledgements

NeuroKit2 is developed by Dominique Makowski and the NeuroKit2 contributors; WFDB for Python by the
MIT Laboratory for Computational Physiology, and PhysioNet asks users of its software and data to
cite Goldberger et al. 2000 (above). The species presets owe their numbers to the PhysioZoo team
(Behar et al. 2018). The demo datasets are published on Pennsieve Discover (349, 655, 406) and
PhysioNet (Fantasia, Autonomic Aging) by their authors, credited above.

## Development

    python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
    make test PYTHON=.venv/bin/python     # pytest on the synthetic fixture (no Docker)
    make fixture PYTHON=.venv/bin/python  # data/fixture/{physio,events}, truth.json
    make run                              # Docker over the fixture into data/output/
    make lambda-check                     # read-only, any uid, HOME/TMPDIR under /tmp
    make run-demo DEMO=<physio-import output>

The review block in `app.yml` generates the `physio` and `events` ports with
`tests/make_fixture.py` (h5py and numpy only).
