# Sample input

ecg-hrv reads, recursively under `INPUT_DIR`:

1. the `physio` output of **physio-import** (`recordings.csv`, `channels.csv`, `events.tsv` and one
   NWB file per recording under `sub-*/`): the normal case in both physiology workflows;
2. optionally the `events` output of **stim-events** (`stim-events/events.tsv`): HRV is then also
   computed before, during and after each stimulation train;
3. any **NWB 2.x files** with ECG `TimeSeries` in `/acquisition` (named `ecg`, `ECG ...`, `EKG`,
   `Lead II`, ...), without physio-import's tables; set `SPECIES` when the file has no
   `Subject.species`.

## Synthetic (no download)

    python -m tests.make_fixture data/fixture   # data/fixture/physio and data/fixture/events are the two ports

Four recordings: a human at rest with one premature beat, a rat with an inverted lead and three
stimulation trains of growing bradycardia, a pig recorded trial by trial (gaps) with a device
heart-rate channel, and a human recording without ECG (skipped).

## Real recordings used to test this app

Run physio-import on these first (its README says how); its `physio` output is this app's input.

- **PhysioNet Fantasia Database** (doi:10.13026/C2RG61, ODC-By 1.0): healthy young and elderly
  adults, ~2 h supine rest, ECG at 250 Hz with reference beat annotations (`.ecg`), used to check
  R-peaks. Files `f1y01`, `f1o01`, `f2y01`, `f2o01`, `f2y02`, `f2o02` (`.hea`, `.dat`, `.ecg`)
  from https://physionet.org/content/fantasia/1.0.0/.
- **PhysioNet Autonomic Aging** (doi:10.13026/2hsy-t491, ODbL 1.0): resting adults, two ECG leads
  at 1000 Hz; records `0001`-`0003` from https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/.
- **Pennsieve Discover 349** (doi:10.26275/dm5l-tspb, CC BY 4.0): pig, vagus nerve stimulation grid,
  ECG at 1 kHz and a device heart-rate channel, `files/primary/sub-2/subject-2.hdf5`.
- **Pennsieve Discover 655** (doi:10.26275/5cwo-mkpz, CC BY 4.0): rat, closed-loop heart-rate control
  by vagus stimulation, CSV exports `files/primary/sub-Bimodal01/perf-*/*.smrx.csv` (ECG at 1 kHz),
  and the authors' heart rate in `files/derivative/HR/*.xlsx`.
- **Pennsieve Discover 406** (doi:10.26275/nung-b9bm, CC BY 4.0): human, uncalibrated single-lead
  ECG as `.wav` at 500 Hz, `files/primary/sub-100*/perf-*/*.wav`.

In the workflow-physiology repository, `scripts/fetch-sample-data.sh` stages all of these under
`data/`, and physio-import's outputs go to `data/work/<demo>/physio-import/physio`.
