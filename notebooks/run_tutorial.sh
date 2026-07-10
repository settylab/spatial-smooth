#!/usr/bin/env bash
# Execute the tutorial and scrub it for publication. Reproducible from a clean clone.
#
#   ./notebooks/run_tutorial.sh              # uses the python on PATH
#   PYTHON=/path/to/venv/bin/python ./notebooks/run_tutorial.sh
#
# Runs on the full section (~36k cells); the Gaussian-process steps take several minutes. On a
# cluster, wrap this in your scheduler rather than embedding host paths here.
#
# Never export MPLBACKEND=Agg for this: inside the kernel it makes `plt.show()` a no-op, so the
# notebook executes cleanly and embeds no figures at all.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
python_bin="${PYTHON:-python}"

# Keep thread pools polite; these are read by the kernel, not by matplotlib.
export PYTHONPATH="${root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-4}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

cd "$here"

"$python_bin" build_tutorial.py           # refuses to write a notebook that does not compile

"$python_bin" -m jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.startup_timeout=300 \
    --ExecutePreprocessor.timeout=3600 \
    tutorial.ipynb

# Redact host paths; keep this package's warnings. Exits non-zero if a leak survives.
"$python_bin" scrub_notebook.py tutorial.ipynb
