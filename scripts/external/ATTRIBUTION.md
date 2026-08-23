# Vendored third-party code

## LD-ALPS

Both files here are Ford Burles's, redistributed unmodified under the MIT
license stated at <https://fordburles.com/ld-alps.html>. They are vendored
rather than downloaded at runtime so that a reader gets the same code the
reported numbers were produced with, even if the author's page changes.

| file | source | what it is |
| --- | --- | --- |
| `ld-alps.py` | <https://fordburles.com/r/ld-alps.py> | the author's current release |
| `ld-alps-original.py` | <https://fordburles.com/r/ld-alps-original.py> | the version behind the author's manuscript |

**The paper uses `ld-alps.py`, the current release.** The two are not
equivalent, and the author changed the method between them:

- `ld-alps-original.py` raises the DBSCAN `min_samples` adaptively from 5 to 20
  while any retained vector's mean great-circle distance exceeds 1 radian,
  takes cluster label 0, and then censors retained vectors whose mean distance
  is more than 3.5 standard deviations from the cluster mean.
- `ld-alps.py` fits DBSCAN once at `min_samples=5`, takes the largest non-noise
  cluster, and has no censoring step. Its ADC interpolation also deduplicates
  the projected points and falls back to linear, then nearest, when
  Clough-Tocher fails, where the original leaves such a voxel undefined.

Neither is a modification by us. Do not "fix" one to match the other: the
difference is the author's, and the manuscript reports the current release.

At the time of writing the author states "Manuscript currently under review",
so there is no citation to give beyond the URL above. Replace this note with
the formal citation once it appears.
