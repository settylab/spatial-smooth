#!/bin/bash
#SBATCH --job-name=ss-tutorial
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -uo pipefail

ROOT=/fh/fast/setty_m/user/dotto/nexus/work/spatial-smooth
VENV=/fh/fast/setty_m/user/dotto/nexus/locals/venvs/livermet-kompot08

export PYTHONPATH="$ROOT/src"
# No MPLBACKEND here: forcing Agg makes the kernel's plt.show() a no-op, so the notebook
# executes cleanly but embeds no figures. The notebook selects `%matplotlib inline` itself.
export LOKY_MAX_CPU_COUNT=4
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

cd "$ROOT/notebooks" || exit 1

"$VENV/bin/jupyter-nbconvert" \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.startup_timeout=300 \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=ss-venv \
    tutorial.ipynb
rc=$?
echo "nbconvert rc=$rc"

# Strip stderr streams: they embed absolute paths and sandbox thread-pool noise, and this
# notebook is rendered into public documentation.
"$VENV/bin/python" - <<'PY'
import nbformat
nb = nbformat.read("tutorial.ipynb", as_version=4)
dropped = 0
for cell in nb.cells:
    if cell.get("cell_type") != "code":
        continue
    keep = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream" and out.get("name") == "stderr":
            dropped += 1
            continue
        keep.append(out)
    cell["outputs"] = keep
nbformat.write(nb, "tutorial.ipynb")
print(f"scrubbed {dropped} stderr outputs")
PY

exit $rc
