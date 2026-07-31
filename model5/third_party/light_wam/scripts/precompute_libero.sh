#!/usr/bin/env bash
set -euo pipefail
LIBERO_SUITE="${LIBERO_SUITE:-spatial}"
case "${LIBERO_SUITE}" in
  spatial|object|goal|10) ;;
  *)
    echo "LIBERO_SUITE must be one of: spatial, object, goal, 10" >&2
    exit 1
    ;;
esac
TARGET="libero_${LIBERO_SUITE}" bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/precompute.sh" "$@"
