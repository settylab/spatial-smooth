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

# Run everything against *this* interpreter's environment, not whatever the shared machine puts
# first. Two independent hazards on a busy multi-user box, both fixed by pointing at $python_bin's
# own bin / prefix (discovered dynamically -- no host paths baked in):
#
#   1. ``$python_bin -m jupyter nbconvert`` execs ``jupyter-nbconvert`` found on PATH. If some other
#      environment's console script is ahead on PATH, its shebang launches a different (possibly
#      broken) Python. Prepend $python_bin's bin so the matching console scripts win.
#   2. ``--execute`` resolves the "python3" kernelspec off the Jupyter search path, which may point
#      at an unrelated environment. Register an ipykernel into $python_bin's prefix (idempotent) and
#      put that prefix first on JUPYTER_PATH so the notebook runs under the same Python.
python_dir="$(cd "$(dirname "$python_bin")" && pwd)"
export PATH="${python_dir}:${PATH}"
env_prefix="$("$python_bin" -c 'import sys; print(sys.prefix)')"
"$python_bin" -m ipykernel install --sys-prefix --name python3 >/dev/null 2>&1 || true
export JUPYTER_PATH="${env_prefix}/share/jupyter${JUPYTER_PATH:+:${JUPYTER_PATH}}"

cd "$here"

"$python_bin" build_tutorial.py           # refuses to write a notebook that does not compile

"$python_bin" -m jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.startup_timeout=300 \
    --ExecutePreprocessor.timeout=3600 \
    tutorial.ipynb

# Redact host paths; keep this package's warnings. Exits non-zero if a leak survives.
"$python_bin" scrub_notebook.py tutorial.ipynb
