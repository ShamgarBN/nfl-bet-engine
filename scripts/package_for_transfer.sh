#!/usr/bin/env bash
# Package the project into a single ``.tar.gz`` you can copy to another
# Mac (AirDrop, iCloud Drive, external SSD, scp, whatever).
#
# What goes in:
#   - all source code, pyproject.toml, uv.lock, README, tests
#   - data/warehouse.duckdb        (DuckDB store of every game/odds/snap we've pulled)
#   - data/raw/                    (raw upstream caches)
#   - data/cache/picks_*.csv       (per-week prediction snapshots)
#   - data/cache/last_*_sync.txt   (so the destination knows when jobs last ran)
#   - data/journal/                (the prediction journal)
#   - models/                      (trained ML artefacts + archive)
#   - reports/                     (end-of-season reports)
#   - NFL Forecast.app/            (launcher bundle, re-signed on dest)
#   - .git/                        (commit history)
#
# What's left out (auto-regenerated on the destination):
#   - .venv/                       (~600 MB, platform-specific wheels)
#   - logs/                        (rebuilt on first launch)
#   - __pycache__/                 (bytecode)
#   - data/cache/http_cache.sqlite (cached HTTP responses; rebuilds itself)
#
# Use ``--full`` to also include the HTTP cache. Larger archive but the
# destination Mac never has to re-call any nflverse / Open-Meteo endpoint
# we've already seen.

set -euo pipefail

ROOT="$( cd "$( dirname "$0" )/.." && pwd )"
PROJECT_NAME="$( basename "${ROOT}" )"

INCLUDE_HTTP_CACHE=0
OUTPUT_DIR="${HOME}/Desktop"
for arg in "$@"; do
  case "${arg}" in
    --full)        INCLUDE_HTTP_CACHE=1 ;;
    --lean)        INCLUDE_HTTP_CACHE=0 ;;
    --output=*)    OUTPUT_DIR="${arg#--output=}" ;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [--full|--lean] [--output=DIR]

  --lean (default)  Skip data/cache/http_cache.sqlite. Resulting tarball is
                    ~80-100 MB. The destination Mac re-caches HTTP responses
                    on demand; first morning-sync runs a little slower.

  --full            Include the HTTP cache. Tarball balloons but the
                    destination needs zero re-fetching.

  --output=DIR      Where to write the tarball. Default: ~/Desktop
EOF
      exit 0 ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

# Refuse to package while the app is running -- DuckDB would copy in an
# inconsistent state and the new Mac would see corruption.
if pgrep -f 'nfl-model app' >/dev/null 2>&1 || pgrep -f 'nfl-model serve' >/dev/null 2>&1; then
  cat >&2 <<EOF
The NFL Forecast app appears to be running.

Close it (Cmd-Q the window or kill the process) before packaging, otherwise
the DuckDB warehouse will be copied mid-transaction.

  pkill -f 'nfl-model'

then re-run this script.
EOF
  exit 1
fi

TIMESTAMP="$( date +%Y%m%d_%H%M%S )"
ARCHIVE="${OUTPUT_DIR}/${PROJECT_NAME}-transfer-${TIMESTAMP}.tar.gz"

EXCLUDES=(
  "--exclude=${PROJECT_NAME}/.venv"
  "--exclude=${PROJECT_NAME}/logs"
  "--exclude=${PROJECT_NAME}/__pycache__"
  "--exclude=*.pyc"
  "--exclude=.DS_Store"
  "--exclude=${PROJECT_NAME}/.mypy_cache"
  "--exclude=${PROJECT_NAME}/.pytest_cache"
  "--exclude=${PROJECT_NAME}/.ruff_cache"
  "--exclude=${PROJECT_NAME}/.ipynb_checkpoints"
)
if [[ "${INCLUDE_HTTP_CACHE}" -eq 0 ]]; then
  EXCLUDES+=( "--exclude=${PROJECT_NAME}/data/cache/http_cache.sqlite" )
fi

echo "Packaging from: ${ROOT}"
echo "Output archive: ${ARCHIVE}"
echo "HTTP cache included: $([[ "${INCLUDE_HTTP_CACHE}" -eq 1 ]] && echo yes || echo "no (default)")"
echo ""

cd "$( dirname "${ROOT}" )"
tar -czf "${ARCHIVE}" \
    "${EXCLUDES[@]}" \
    "${PROJECT_NAME}"

SIZE="$( du -h "${ARCHIVE}" | awk '{print $1}' )"
SHA="$( shasum -a 256 "${ARCHIVE}" | awk '{print $1}' )"

cat <<EOF

Built ${ARCHIVE}
Size: ${SIZE}
SHA-256: ${SHA}

Next steps (on the destination Mac):

  1. Install uv if it's not already there:
        curl -LsSf https://astral.sh/uv/install.sh | sh

  2. Copy the tarball over. Recommended landing spot: ~/Projects/ or ~/Applications/.

  3. Extract it:
        cd ~/Projects
        tar -xzf ~/Downloads/${PROJECT_NAME}-transfer-${TIMESTAMP}.tar.gz

  4. Strip any Gatekeeper quarantine flags the transfer added:
        xattr -dr com.apple.quarantine ~/Projects/${PROJECT_NAME}

  5. Re-sign the .app on this Mac (ad-hoc, no Apple Developer account needed):
        cd ~/Projects/${PROJECT_NAME}
        ./scripts/build_app_bundle.sh

  6. Double-click "NFL Forecast.app".

Done. Both Macs now have identical models, warehouse, and journal.
EOF
