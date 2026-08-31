#!/usr/bin/env bash
set -e

source "/opt/ros/${ROS_DISTRO}/setup.bash"

if [[ -f "${TRISKEL_WS}/install/setup.bash" ]]; then
    source "${TRISKEL_WS}/install/setup.bash"
fi

exec "$@"

