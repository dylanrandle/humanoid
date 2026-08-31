test_keyboard_teleop() {
    local before_teleop
    local base_command_file="${test_directory}/dashboard-base-command.txt"
    local tool_command_file="${test_directory}/dashboard-tool-command.txt"
    local servo_status_file="${test_directory}/servo-status.txt"
    local servo_output_file="${test_directory}/servo-output.txt"
    local servo_motion=false
    local deadline

    before_teleop="$(api_get /api/status)"
    api_post /api/mode '{"mode":"keyboard"}' \
        | jq -e '.status.mode == "keyboard"' >/dev/null

    timeout 10 ros2 topic echo --once /cmd_vel >"${base_command_file}" &
    local base_echo_pid=$!
    timeout 10 ros2 topic echo --once /servo_node/delta_twist_cmds >"${tool_command_file}" &
    local tool_echo_pid=$!
    timeout 20 ros2 topic echo --once /servo_node/status >"${servo_status_file}" &
    local servo_status_pid=$!
    timeout 20 ros2 topic echo --once /arm_controller/joint_trajectory >"${servo_output_file}" &
    local servo_output_pid=$!

    sleep 2
    for _ in {1..10}; do
        api_post /api/teleop '{"commands":["base_forward","tool_up"]}' >/dev/null
        sleep 0.1
    done

    wait "${base_echo_pid}" || fail "Dashboard did not publish a base teleop command."
    wait "${tool_echo_pid}" || fail "Dashboard did not publish a tool teleop command."
    grep -Fq 'x: 0.1' "${base_command_file}"
    grep -Fq 'z: 0.05' "${tool_command_file}"

    deadline=$((SECONDS + 10))
    while ((SECONDS < deadline)); do
        local status
        status="$(api_get /api/status)"
        if jq -e --argjson before "${before_teleop}" '
            [range(1; 8) as $index
              | ((.status.joints["arm_\($index)"] - $before.status.joints["arm_\($index)"]) | fabs)]
            | max > 0.002
        ' <<<"${status}" >/dev/null; then
            servo_motion=true
            break
        fi
        api_post /api/teleop '{"commands":["tool_up"]}' >/dev/null
        sleep 0.1
    done

    if [[ "${servo_motion}" != true ]]; then
        kill "${servo_status_pid}" "${servo_output_pid}" 2>/dev/null || true
        wait "${servo_status_pid}" "${servo_output_pid}" 2>/dev/null || true
        printf 'Last observed Servo status:\n' >&2
        cat "${servo_status_file}" >&2 || true
        printf 'Last observed Servo controller output:\n' >&2
        cat "${servo_output_file}" >&2 || true
        printf 'Servo node graph:\n' >&2
        ros2 node info /servo_node >&2 || true
        printf 'Servo command topic graph:\n' >&2
        ros2 topic info --verbose /servo_node/delta_twist_cmds >&2 || true
        printf 'Servo configured topics:\n' >&2
        ros2 param get /servo_node moveit_servo.cartesian_command_in_topic >&2 || true
        ros2 param get /servo_node moveit_servo.command_out_topic >&2 || true
        fail "MoveIt Servo did not move an arm joint."
    fi

    kill "${servo_status_pid}" "${servo_output_pid}" 2>/dev/null || true
    wait "${servo_status_pid}" "${servo_output_pid}" 2>/dev/null || true
    api_get /api/status | jq -e '
        [
            .status.topic_rates[]
            | select(
                .topic == "/servo_node/delta_twist_cmds"
                or .topic == "/arm_controller/joint_trajectory"
            )
            | .state
        ] == ["healthy", "healthy"]
    ' >/dev/null || fail "Active task-space command rates were not healthy."

    api_post /api/teleop '{"commands":[]}' >/dev/null
    api_post /api/mode '{"mode":"idle"}' >/dev/null
    local visualization_after_teleop
    visualization_after_teleop="$(
        timeout 5 ros2 topic echo --once /triskel/visualization/ready
    )" || fail "Visualization stopped after task-space teleoperation."
    assert_contains "${visualization_after_teleop}" "data: true" \
        "post-teleop visualization readiness"
}
