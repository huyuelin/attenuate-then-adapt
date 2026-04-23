#!/usr/bin/env bash
# Download the 8-domain and 16-domain continual-LM sources.
#
# The repository intentionally does not bundle any data. Run this script
# to populate ``./data/8domain`` with eight sub-directories. Each
# sub-directory must contain a single .txt or .bin file with the
# tokenized stream; the exact formats expected by the external trainer
# are out of scope here.

set -e

target="${DATA_ROOT:-./data/8domain}"
mkdir -p "${target}"

echo "[download] target directory: ${target}"
echo
echo "The 8-domain stream consists of the following subsets:"
python - <<'PY'
from benchmarks.datasets import manifest_lines
for line in manifest_lines():
    print("  " + line)
PY
echo
echo "Populate each subdirectory manually (this script is a checklist,"
echo "not an automated downloader)."
