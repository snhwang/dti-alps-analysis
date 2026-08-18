# Analysis scripts

Every script that produced a number, table or figure in *Head Position Confounds
the DTI-ALPS Index in Aging and Disease*.

These are the working scripts, not a packaged tool. They are published so that
any result in the paper can be traced to the code that made it. Expect to edit
paths before anything runs.

## Data

None of it is included here. Get it from the sources below.

- **HCP-A** — through the BALSA repository, <https://balsa.wustl.edu>, to
  qualified registered users under the AABC Consortium Data Use Terms. Derived
  values are not redistributed here, and identifier-linked derived values may not
  be redistributed at all under those Terms.
- **DLBS** — OpenNeuro, <https://openneuro.org/datasets/ds004856>, version 1.2.0.
- **Trigeminal neuralgia** — OpenNeuro,
  <https://openneuro.org/datasets/ds005713>, version 2.0.2, CC0.

Result files are not included either. The final values are in the manuscript.

## Requirements

    uv venv
    uv pip install -r requirements.txt

Install **FSL** separately (<https://fsl.fmrib.ox.ac.uk>). It is used for
registration and for the tract atlases, and several scripts shell out to it.

`scripts/axis_error_sensitivity.py` imports `dti_alps_refined`, the companion
package at <https://github.com/snhwang/dti-alps-refined>.

## Data directories must be edited

Paths are hardcoded to the machine the analyses ran on. Drive letters appear as
`Q:/dti_output` for the processed sessions, `M:/` for the trigeminal release,
`F:/` and `P:/` for the HCP structural packages, and `C:/Users/.../fsl` for FSL.

`scripts/data_paths.py` translates those between Windows and WSL, but the letters
themselves are literals. Change them to your own locations before running
anything. `grep -rn 'winpath\|:/' scripts/*.py` finds them.

## Running

`scripts/regenerate_and_diff.py` runs the chain that regenerates the derived
values and diffs each output against the committed version.
`scripts/verify_manuscript.py` checks the manuscript's numbers against those
outputs; it needs the manuscript source, which is not published here.

Individual scripts are self-contained and documented at the top of each file,
which is the better starting point for tracing one number.

## Licence

MIT.
