# Archived scripts

Nothing here produces a number in the current manuscript. These are kept
because they ran, and because a reader tracing the project's history is better
served by finding them than by finding a gap.

## Trigeminal neuralgia (OpenNeuro ds005713)

A patient cohort analyzed during revision and then removed before submission.
It was dropped for three reasons, none of them a problem with the code:

- The group effect it showed was bilateral, with no difference between the side
  ipsilateral and contralateral to the pain, so it was not a consequence of the
  nerve lesion and could not be interpreted.
- At 168 participants it was the smallest cohort in the paper by a factor of
  five, so it did not answer the reviewer request for larger data either.
- Aging is the clinical domain this index is used in, and both remaining
  cohorts are aging cohorts with clinical and cognitive phenotyping, so the
  clinical question was already covered.

`tn_alps.py` is **not** here. Despite its name it is no longer
trigeminal-specific: `aging_cohort_comparators.py` and `ds001907_alps.py` both
import `alps_variants` from it to compute ALPS-PAS and the per-voxel comparator
in the aging cohorts. It should probably be renamed, which is why it has not
been.
