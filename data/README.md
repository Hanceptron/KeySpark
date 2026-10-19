# keyspark_windows.csv

Content-free per-window feature table released with the paper. One row per
scored one-minute window (windows with fewer than five events are dropped).

| Column | Meaning |
| --- | --- |
| `set` | `genuine` (the author's real input), `synthetic` (generator output, `keyspark/botgen.py`), or `injected` (seeded demonstration days and the development test user) |
| `label` | 0 = human, 1 = input automation |
| `kind` | generator kind for synthetic rows: `jiggler`, `typer`, `keep_awake`, `clicker` |
| `user` | `U1` = the genuine user, `U2` = the test user; empty for synthetic rows |
| `day` | day index, 1 = first genuine day; injected days may carry values <= 0 |
| `session` | session id (300 s idle gap), unique within the file |
| `window` | window index within its session, 0-based |
| `n_events` | events in the window |
| `key_diversity` ... `mouse_fraction` | the seven features of `keyspark.ml.FEATURES`, defined in the paper (Section 3.3) |

Counts: 3,649 genuine, 645 synthetic, 1,301 injected windows; 22 genuine days.

**What is not here.** No key identity, typed text, pointer coordinate, or clock
time. Dates are replaced by day indices and names by `U1`/`U2`. The raw event
archive is not released because it contains the author's plaintext keystrokes.

**Reproducing the paper.** `paper/scripts/reproduce_from_csv.py` recomputes the
session-wise hold-out (F1 0.982 +- 0.009), the leave-one-kind-out folds (mean
F1 0.459), and the day-level policy (2/22 false alarms, 3/4 detections at
tau = 0.8, m = 2) from this file alone. The throughput benchmarks need a raw
archive and are not reproducible from it.

Regenerate from a local archive with `paper/scripts/export_features.py`.
