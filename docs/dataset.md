# Dataset: GREAT sEMG with arm translation

Where the data comes from, how to get it, how it is laid out, and how it was
recorded. Specifications below were checked against the source paper, the
`recording_parameters.txt` shipped with the data, and the authors' own repository.

## At a glance

| | |
|---|---|
| Name | EMG Dataset for Gesture Recognition with Arm Translation |
| Short name | GREAT, after the authors' repository `MoveR_AT_GREAT` |
| Domain | Surface electromyography for gesture recognition, prosthetics, rehabilitation |
| Participants | 8 healthy adult males, right hand, two consecutive days |
| Signal | 16-channel sEMG at 2 kHz, bandpassed 20 to 450 Hz at acquisition, 4th order |
| Also recorded | 18-sensor data glove, finger positions. Neither is used here |
| Gestures | 6: power, lateral, tripod, pointer, open, rest |
| Arm positions | 9, a 3 by 3 grid at 45 degree offsets around a neutral P5 |
| Trials | 4,800 grasps: 8 participants x 2 days x 2 blocks x 5 positions x 6 gestures x 5 repetitions |
| Format | HDF5 for signals, CSV for labels, one folder per participant, day and block |
| Data licence | CC0 1.0 |
| Paper licence | CC BY 4.0 |

## Source and citation

**Data, on Dryad.** DOI `10.5061/dryad.8sf7m0czv`, released 2024-11-19, CC0 1.0.

> Kyranou, I., Szymaniak, K. and Nazarpour, K. *EMG dataset for gesture recognition
> with arm translation.* Dryad (2024). https://doi.org/10.5061/dryad.8sf7m0czv

**Paper, the data descriptor.** *Scientific Data* (2025) 12:100, CC BY 4.0.

> Kyranou, I., Szymaniak, K. and Nazarpour, K. *EMG Dataset for Gesture Recognition
> with Arm Translation.* Scientific Data 12, 100 (2025).
> https://doi.org/10.1038/s41597-024-04296-8

**Authors' repository.** https://github.com/MoveR-Digital-Health-and-Care-Hub/MoveR_AT_GREAT
This is the authoritative source for the gesture code mapping, for the reason given
below.

## Getting it

1. Open https://doi.org/10.5061/dryad.8sf7m0czv, which resolves to the Dryad page.
2. Use **Download dataset**. No account is needed; the data is CC0.
3. Unpack it. One folder per participant appears.
4. Only `emg_data.hdf5` and `trials.csv` are needed here. `glove_data.hdf5` and
   `finger_data.hdf5` are not read.

## Layout

Folders are named `participant{X}_day{Y}_block{Z}`, with Y and Z each 1 or 2.

The paper calls the two recordings of a day *sessions*, while the folders and
`recording_parameters.txt` call them *blocks*. This document follows the names on
disk.

| File | Contents | Used |
|---|---|---|
| `emg_data.hdf5` | 16-channel sEMG, keyed by trial, 150 trials per block | yes |
| `trials.csv` | labels: gesture, target position, trial number, block | yes |
| `glove_data.hdf5` | 18-sensor data glove | no |
| `finger_data.hdf5` | five finger positions | no |
| `recording_parameters.txt` | acquisition settings | reference |

Each key in `emg_data.hdf5` is one trial, holding a `(16, time)` array. `trials.csv`
carries `row_number`, `target_position` (1 to 9), `grasp` (1 to 6), `trial_no` and
`block`.

**Gesture codes:** `1=power, 2=lateral, 3=tripod, 4=pointer, 5=open, 6=rest`.

This mapping is taken from the authors' repository rather than the paper. The paper
does not bind numbers to names explicitly, and its prose order differs from the order
in its Table 5, notably for pointer and tripod, so the paper alone does not settle
it. The repository README and Table 5 agree that tripod precedes pointer.

## How it was recorded

**Electrodes.** Four Delsys Trigno Quattro sensors, each carrying one reference and
four sEMG channels, giving 16 channels at 2 kHz. A 4th-order 20 to 450 Hz bandpass is
applied during acquisition. Electrodes sit in two rows of eight on the forearm, the
first aligned to extensor carpi ulnaris, on skin cleaned with 70% isopropyl alcohol.

**Kinematics.** An 18-sensor CyberGlove II, calibrated once per participant. Not used
here.

**Gestures.** power, a strong fist grasp; lateral, a key grip; tripod, a
three-finger grasp; pointer, an extended index finger; open; and rest.

**Arm positions.** Nine positions on a 3 by 3 grid, each 45 degrees from the neutral
P5 in elevation and azimuth. Only five are recorded per day, in two configurations:

- **Con+**, day 1: positions 2, 4, 5, 6, 8
- **Conx**, day 2: positions 1, 3, 5, 7, 9

Only the centre, P5, is shared, so it holds roughly twice the data of any other
position.

**Trials.** One trial is a five second hold of the cued gesture followed by three
seconds of rest, repeated five times for each gesture and position pair. That is
6 x 5 x 5 = 150 trials per block, two blocks a day over two days, so 600 trials per
participant and 4,800 in total.

## How this project uses it

- **Signals only.** The 16 sEMG channels from `emg_data.hdf5`. No extra filtering is
  applied, since the data arrives already bandpassed.
- **Windowing.** 128 ms windows, 256 samples, with a 50 ms hop of 100 samples, giving
  **470,413 windows** across the 4,800 trials.
- **Features.** Per channel and per window: the Hudgins 4, giving 64 dimensions, and
  an extended 14, giving 224.
- **Grouping keys.** Participant, arm position, day and trial are stored beside the
  features so that a split can keep every window of a trial, position or participant
  on one side.

## Placing it locally

`DATA_DIR` at the top of `semg/features.py` points at the folder holding the
participant directories, by default `../data/extracted/data`:

```
data/extracted/data/
  participant_1/
    participant1_day1_block1/    emg_data.hdf5, trials.csv, recording_parameters.txt
    participant1_day1_block2/
    participant1_day2_block1/
    participant1_day2_block2/
  participant_2/
  ...
  participant_8/
```

Change that one line if the data lives elsewhere.

## Ethics

The paper reports that participants received written and verbal explanation and gave
informed consent, that the study followed the Declaration of Helsinki, and that it
held institutional ethical approval, including separate consent for the publication
of identifiable images. The data is released under CC0 and may be reused and
redistributed freely, with attribution encouraged.
