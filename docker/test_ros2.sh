#!/usr/bin/env bash
set -Ee -o pipefail

source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
source "${TRISKEL_WS:-/workspace/triskel/ros_ws}/install/setup.bash"
set -u

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/tests/helpers.bash"
for test_module in "${script_directory}"/tests/test_*.bash; do
    source "${test_module}"
done

test_directory="$(mktemp -d /tmp/triskel-ros2-smoke.XXXXXX)"
launch_log="${test_directory}/bringup.log"
bringup_pid=""

cleanup() {
    local status=$?
    trap - EXIT

    if [[ -n "${bringup_pid}" ]] && kill -0 "${bringup_pid}" 2>/dev/null; then
        kill -INT "${bringup_pid}" 2>/dev/null || true
        for _ in {1..10}; do
            if ! kill -0 "${bringup_pid}" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        kill -TERM "${bringup_pid}" 2>/dev/null || true
        wait "${bringup_pid}" 2>/dev/null || true
    fi

    if ((status != 0)); then
        printf 'ROS 2 smoke test failed. Bringup log follows:\n' >&2
        tail -n 200 "${launch_log}" >&2 || true
    fi

    rm -rf -- "${test_directory}"
    exit "${status}"
}
trap cleanup EXIT

start_bringup() {
    ros2 launch triskel_bringup operator.launch.py \
        use_mock_hardware:=true \
        start_rviz:=false \
        recording_root:="${test_directory}/recordings" \
        >"${launch_log}" 2>&1 &
    bringup_pid=$!
}

run_test "Robot description variants" test_robot_descriptions
start_bringup
run_test "ROS controllers and dashboard discovery" test_runtime_startup
run_test "Feedback, actions, odometry, and TF" test_runtime_interfaces
run_test "Validated Home and Rest poses" test_named_poses
run_test "Keyboard dead-man and Cartesian teleoperation" test_keyboard_teleop
run_test "Meta Quest teleoperation and homing" test_meta_quest_teleop
run_test "rosbag2 recording and replay" test_recording_and_replay

printf '\nROS 2 dashboard parity smoke suite passed.\n'
