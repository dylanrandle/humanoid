test_runtime_startup() {
    local expected_controllers=(
        joint_state_broadcaster
        omni_base_controller
        arm_controller
        gripper_controller
    )
    local controller_state=""
    local all_active=false
    local deadline=$((SECONDS + 90))

    while ((SECONDS < deadline)); do
        if ! kill -0 "${bringup_pid}" 2>/dev/null; then
            wait "${bringup_pid}" || true
            fail "Bringup exited before all controllers became active."
        fi

        controller_state="$(
            timeout 5 ros2 control list_controllers -c /controller_manager 2>/dev/null || true
        )"
        all_active=true
        for controller in "${expected_controllers[@]}"; do
            if ! grep -Eq "^${controller}[[:space:]].*[[:space:]]active$" \
                <<<"${controller_state}"; then
                all_active=false
                break
            fi
        done

        if [[ "${all_active}" == true ]]; then
            break
        fi
        sleep 1
    done

    [[ "${all_active}" == true ]] || fail "Controllers did not become active within 90 seconds."
    printf '%s\n' "${controller_state}"

    wait_for_status 60 "Dashboard did not report a ready Triskel graph." '
        .status.ready == true
        and .status.robot == "triskel"
        and .status.visualization.ready == true
        and .status.visualization.port == 8080
    '
    curl --fail --silent --show-error http://127.0.0.1:8080/ >/dev/null \
        || fail "Viser browser visualization is unavailable."

    local visualization_ready
    visualization_ready="$(timeout 5 ros2 topic echo --once /triskel/visualization/ready)" \
        || fail "No visualization readiness heartbeat arrived."
    assert_contains "${visualization_ready}" "data: true" "visualization readiness"

    api_post /api/robot '{"robot":"triskel"}' \
        | jq -e '.status.robot == "triskel"' >/dev/null
    api_get /api/status | jq -e '
        .status.teleop_devices == ["keyboard", "meta_quest"]
    ' >/dev/null

    wait_for_status 10 "Dashboard did not report healthy continuous ROS topic rates." '
        [
            .status.topic_rates[]
            | select(
                .topic == "/joint_states"
                or .topic == "/omni_base_controller/odom"
                or .topic == "/triskel/visualization/ready"
            )
            | .state
        ] == ["healthy", "healthy", "healthy"]
    '
}

test_runtime_interfaces() {
    local joint_states
    local actions
    local topic_info
    local odometry
    local base_transform=""
    local base_transform_ready=false
    local deadline

    joint_states="$(timeout 20 ros2 topic echo --once /joint_states)" \
        || fail "No joint state arrived within 20 seconds."
    for joint in wheel_1 wheel_2 wheel_3 arm_1 arm_2 arm_3 arm_4 arm_5 arm_6 arm_7 gripper_1; do
        assert_contains "${joint_states}" "${joint}" "joint-state entry"
    done

    actions="$(ros2 action list)"
    assert_contains "${actions}" "/arm_controller/follow_joint_trajectory" "arm action"
    assert_contains "${actions}" "/gripper_controller/follow_joint_trajectory" "gripper action"

    topic_info="$(ros2 topic info /cmd_vel)"
    grep -Eq 'Subscription count: [1-9][0-9]*' <<<"${topic_info}" \
        || fail "The base command topic has no subscribers."
    ros2 topic pub --once /cmd_vel geometry_msgs/msg/TwistStamped \
        "{header: {frame_id: base_link}, twist: {linear: {x: 0.05}, angular: {z: 0.1}}}" \
        >/dev/null

    odometry="$(timeout 20 ros2 topic echo --once /omni_base_controller/odom)" \
        || fail "No odometry arrived within 20 seconds."
    assert_contains "${odometry}" "frame_id: odom" "odometry frame"
    assert_contains "${odometry}" "child_frame_id: base_link" "odometry child frame"

    deadline=$((SECONDS + 20))
    while ((SECONDS < deadline)); do
        base_transform="$(timeout 3 ros2 topic echo --once /tf 2>/dev/null || true)"
        if grep -Fq 'frame_id: odom' <<<"${base_transform}" \
            && grep -Fq 'child_frame_id: base_link' <<<"${base_transform}"; then
            base_transform_ready=true
            break
        fi
    done
    [[ "${base_transform_ready}" == true ]] \
        || fail "No canonical odom-to-base_link TF arrived within 20 seconds."
}
