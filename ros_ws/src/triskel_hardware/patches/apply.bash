#!/usr/bin/env bash
set -Eeuo pipefail

patch_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
driver_directory="${1:-${patch_directory}/../../sts_hardware_interface}"
patch_file="${patch_directory}/sts-proportional-velocity-units.patch"

[[ -d "${driver_directory}/.git" ]] || {
    printf 'ERROR: STS dependency is missing at %s\n' "${driver_directory}" >&2
    exit 1
}

if git -C "${driver_directory}" apply --reverse --check "${patch_file}" 2>/dev/null; then
    printf 'STS proportional velocity patch is already applied.\n'
elif git -C "${driver_directory}" apply --check "${patch_file}"; then
    git -C "${driver_directory}" apply "${patch_file}"
    printf 'Applied STS proportional velocity patch.\n'
else
    printf 'ERROR: STS proportional velocity patch does not apply cleanly.\n' >&2
    exit 1
fi
