# Data directory

**No real patient data is stored or tracked in this repository.**

In line with the project data-governance policy, raw, input, interim,
processed, and output data files are excluded by the top-level
[`.gitignore`](../.gitignore) and must never be committed or pushed to any
public remote. Only source code and non-sensitive schema metadata are public.

## What lives here

| Item | Tracked? | Notes |
|------|----------|-------|
| `codebook_crosswalk.csv` | yes | Variable schema only (names, types, allowed values). Contains **no** patient records. |
| `example/generate_example.py` | yes | Generates an example cohort that reproduces the schema and marginal distributions used by the pipeline. |
| `example/cohort.csv` | **no** | Generated locally on demand; git-ignored. |
| `raw/`, `input/`, `interim/`, `processed/`, `external/` | **no** | Reserved for private local data; git-ignored. |

## Running on private data

Place the private cohort file at a local path (inside `data/raw/` or anywhere
outside the tracked tree) and set `data.path` in `config/default.yaml`, or pass
`--data <path>` to the pipeline runner. Verify with `git status` that no data
file is staged before any commit or push.
