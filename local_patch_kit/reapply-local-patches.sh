#!/usr/bin/env bash
# One registered patch set, rehearsed before any source changes. Does not restart Hermes.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOME_ROOT="$(cd -- "$HERE/.." && pwd)"
# Native Windows Python does not translate MSYS /c/... arguments.
if command -v cygpath >/dev/null 2>&1; then
  HERE="$(cygpath -m "$HERE")"
  HOME_ROOT="$(cygpath -m "$HOME_ROOT")"
fi
ROOT="${HERMES_AGENT_ROOT:-$HOME_ROOT/hermes-agent}"
KIT="$HOME_ROOT/patches/replay-kit"

# Windows Store installs put execution aliases on PATH that open the Store GUI when run.
# Reject them before anything is executed, including HERMES_PATCH_PYTHON and PATH entries.
is_store_alias() {
  local value="${1//\\//}"
  case "${value,,}" in
    */windowsapps | */windowsapps/*) return 0 ;;
  esac
  return 1
}

# Resolve a bare command name through PATH. Absolute/relative paths are used as given,
# never split into a command line and never evaluated.
resolve_interpreter() {
  case "$1" in
    */* | *\\* | *:*) printf '%s' "$1" ;;
    *) command -v -- "$1" 2>/dev/null || true ;;
  esac
}

# A candidate is usable only if a bounded, non-interactive Python probe exits 0.
probe_interpreter() {
  local candidate="$1" status
  if is_store_alias "$candidate"; then
    printf 'reapply-local-patches: refusing Windows Store execution alias: %s\n' "$candidate" >&2
    return 1
  fi
  if [ ! -f "$candidate" ]; then
    printf 'reapply-local-patches: interpreter not found: %s\n' "$candidate" >&2
    return 1
  fi
  if command -v timeout >/dev/null 2>&1; then
    if timeout 20 "$candidate" -I -B -c \
      'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' </dev/null >/dev/null 2>&1; then
      return 0
    else
      status=$?
    fi
  else
    if "$candidate" -I -B -c \
      'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' </dev/null >/dev/null 2>&1; then
      return 0
    else
      status=$?
    fi
  fi
  printf 'reapply-local-patches: interpreter failed the Python 3.11 probe (exit %s): %s\n' \
    "$status" "$candidate" >&2
  return 1
}

consider_interpreter() {
  local resolved
  resolved="$(resolve_interpreter "$1")"
  if [ -z "$resolved" ]; then
    printf 'reapply-local-patches: interpreter not found: %s\n' "$1" >&2
    return 1
  fi
  probe_interpreter "$resolved"
}

find_path_interpreter() {
  local directory name
  local IFS=':'
  for directory in ${PATH:-}; do
    [ -n "$directory" ] || continue
    for name in python python3 python.exe python3.exe; do
      if [ -f "$directory/$name" ] && probe_interpreter "$directory/$name"; then
        PY="$directory/$name"
        return 0
      fi
    done
  done
  return 1
}

PY=""
if [ -f "$ROOT/.venv/Scripts/python.exe" ] && probe_interpreter "$ROOT/.venv/Scripts/python.exe"; then
  PY="$ROOT/.venv/Scripts/python.exe"
elif [ -f "$ROOT/.venv/bin/python" ] && probe_interpreter "$ROOT/.venv/bin/python"; then
  PY="$ROOT/.venv/bin/python"
elif [ -n "${HERMES_PATCH_PYTHON:-}" ] && consider_interpreter "$HERMES_PATCH_PYTHON"; then
  PY="$(resolve_interpreter "$HERMES_PATCH_PYTHON")"
elif find_path_interpreter; then
  :
else
  printf 'reapply-local-patches: no usable Python interpreter; refusing to run the replay driver\n' >&2
  exit 1
fi

# Reuse the installed guard's selection rule instead of duplicating it here.
select_manifest() {
  "$1" -I -B - "$ROOT" "$HERE" <<'PY'
import importlib.util
import os
import sys
from pathlib import Path


def fail(message):
    sys.stderr.write('reapply-local-patches: ' + message + '\n')
    raise SystemExit(1)


def main():
    root = Path(sys.argv[1])
    guard_path = Path(sys.argv[2]) / 'native_patch_guard.py'
    try:
        spec = importlib.util.spec_from_file_location('_native_patch_guard', guard_path)
        if spec is None or spec.loader is None:
            fail('cannot load the native patch guard at ' + str(guard_path))
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        selected = Path(guard.manifest_for(root, guard.driver_module(root)))
    except SystemExit:
        raise
    except Exception as exc:  # no traceback: paths and exception text are enough to recover
        fail('cannot select a patch manifest for ' + str(root) + ': ' + str(exc))
    # Binary write keeps CRLF translation out of the path passed back to Bash.
    sys.stdout.buffer.write(os.fsencode(str(selected)) + b'\n')
    sys.stdout.buffer.flush()


main()
PY
}

DRIVER_ARGS=("$HERE/reapply_local_patches.py" --root "$ROOT")
if [[ $# -gt 0 && "$1" != -* ]]; then
  # Preserve the old optional explicit-patch argument, with the safety checks.
  DRIVER_ARGS+=(--patch "$1")
  shift
else
  explicit=0
  for argument in "$@"; do
    case "$argument" in
      --manifest | --manifest=* | --patch | --patch=*) explicit=1 ;;
    esac
  done
  if [ "$explicit" -eq 0 ]; then
    if ! SELECTED="$(select_manifest "$PY")"; then
      printf 'reapply-local-patches: no patch manifest selected; refusing to run the replay driver\n' >&2
      exit 1
    fi
    SELECTED="${SELECTED%$'\r'}"
    if [ -z "$SELECTED" ]; then
      printf 'reapply-local-patches: the manifest selection returned an empty path; refusing to run the replay driver\n' >&2
      exit 1
    fi
    DRIVER_ARGS+=(--manifest "$SELECTED")
  fi
fi
DRIVER_ARGS+=(--workspace "$HOME_ROOT/cache/scratch/local-patch-replay" --report "$KIT/latest-report.json")
exec "$PY" "${DRIVER_ARGS[@]}" "$@"
