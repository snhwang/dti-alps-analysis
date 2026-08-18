# Analysis scripts

Every script that produced a number, table or figure in *Head Position Confounds
the DTI-ALPS Index in Aging and Disease*.

These are the working scripts, not a packaged tool. They are published so that
any result in the paper can be traced to the code that made it. Expect to edit
paths before anything runs.

## What you can run, and what you cannot

There is a chain: source images, then processing, then cached per-session CSVs,
then the analyses that read those CSVs. Only the code is published here, so where
you can join the chain depends on what data you have.

- **30 scripts read imaging data.** They need the cohorts below, FSL, and the
  processed sessions on disk.
- **36 scripts read a cached CSV that an earlier script wrote.** They fail with a
  `FileNotFoundError` in a fresh clone until you have run the upstream script
  that produces their input. That is expected, not a fault.
- **24 need neither**, and run immediately. `scripts/sorting_bias_floor.py` is
  the clearest example: it simulates the acquisition and reproduces the
  eigenvalue-sorting noise floor quoted in the paper, with no data at all.

Result files are not published. HCP-A derived values are restricted by the AABC
Data Use Terms, and the final values for everything are in the manuscript.

## Data

- **HCP-A** — the BALSA repository, <https://balsa.wustl.edu>, to qualified
  registered users under the AABC Consortium Data Use Terms. Identifier-linked
  derived values may not be redistributed under those Terms, which is why none
  are here.
- **DLBS** — OpenNeuro, <https://openneuro.org/datasets/ds004856>, version 1.2.0.
- **Trigeminal neuralgia** — OpenNeuro,
  <https://openneuro.org/datasets/ds005713>, version 2.0.2, CC0.

## Install

    uv venv
    uv pip install -r requirements.txt

Install **FSL** separately, <https://fsl.fmrib.ox.ac.uk>. It is not a Python
package. Several scripts shell out to it for registration and the tract atlases.

## Data directories must be edited

Paths are hardcoded to the machine the analyses ran on: `Q:/dti_output` for the
processed sessions, `M:/` for the trigeminal release, `F:/` and `P:/` for the
HCP-A structural packages, `C:/Users/.../fsl` for FSL.

`scripts/data_paths.py` translates those between Windows and WSL, but the drive
letters are literals. Change them to your own locations first.

    grep -rn "winpath\|:/" scripts/*.py

## Running

Each script is self-contained and documented at the top of the file, which is the
best starting point for tracing a single number.

`scripts/regenerate_and_diff.py` runs the chain end to end and diffs each output
against the committed version. `scripts/verify_manuscript.py` checks the
manuscript's numbers against those outputs; it needs the manuscript source, which
is not published here.

## Licence

MIT.
