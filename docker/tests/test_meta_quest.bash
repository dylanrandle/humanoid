test_meta_quest_teleop() {
    local before_quest
    local quest_base_file="${test_directory}/quest-base-command.txt"
    local quest_tool_file="${test_directory}/quest-tool-command.txt"
    local quest_gripper_moved=false
    local quest_status=""
    local deadline

    before_quest="$(api_get /api/status)"
    api_post /api/mode '{"mode":"meta_quest"}' \
        | jq -e '.status.mode == "meta_quest" and .status.teleop_device == "meta_quest"' \
            >/dev/null

    timeout 15 ros2 topic echo --once /cmd_vel >"${quest_base_file}" &
    local quest_base_echo_pid=$!
    PYTHONUNBUFFERED=1 timeout 15 ros2 topic echo /servo_node/delta_twist_cmds \
        | awk '
            $1 == "x:" && ($2 + 0 > 0.001 || $2 + 0 < -0.001) {
                print
                found = 1
                exit
            }
            END { exit !found }
        ' >"${quest_tool_file}" &
    local quest_tool_echo_pid=$!
    ros2 topic pub --rate 20 --times 80 /triskel/teleop/meta_quest/joy sensor_msgs/msg/Joy \
        '{axes: [0.25, 0.5, 0.0, 0.2], buttons: [0, 1, 0, 0, 1, 0]}' >/dev/null &
    local quest_joy_pid=$!
    ros2 topic pub --rate 10 --times 40 /triskel/teleop/meta_quest/right_controller_pose \
        geometry_msgs/msg/PoseStamped \
        '{header: {frame_id: openxr}, pose: {orientation: {w: 1.0}}}' >/dev/null &
    local quest_reference_pose_pid=$!

    sleep 1
    ros2 topic pub --rate 50 --times 100 /triskel/teleop/meta_quest/right_controller_pose \
        geometry_msgs/msg/PoseStamped \
        '{header: {frame_id: openxr}, pose: {position: {x: 0.03}, orientation: {w: 1.0}}}' \
        >/dev/null

    wait "${quest_base_echo_pid}" \
        || fail "Meta Quest joystick input did not publish a base command."
    wait "${quest_tool_echo_pid}" 2>/dev/null || true
    wait "${quest_joy_pid}" || fail "Meta Quest Joy publisher exited unexpectedly."
    wait "${quest_reference_pose_pid}" \
        || fail "Meta Quest reference-pose publisher exited unexpectedly."

    if ! grep -Fq 'x: 0.05' "${quest_base_file}"; then
        printf 'Observed Meta Quest base command:\n' >&2
        cat "${quest_base_file}" >&2 || true
        fail "Meta Quest joystick input used the wrong base mapping."
    fi
    if [[ ! -s "${quest_tool_file}" ]]; then
        printf 'Observed Meta Quest tool command:\n' >&2
        cat "${quest_tool_file}" >&2 || true
        fail "Meta Quest pose input did not publish tool motion."
    fi

    deadline=$((SECONDS + 10))
    while ((SECONDS < deadline)); do
        quest_status="$(api_get /api/status)"
        if jq -e --argjson before "${before_quest}" '
            .status.mode == "meta_quest"
            and .status.joints.gripper_1 < $before.status.joints.gripper_1 - 0.01
        ' <<<"${quest_status}" >/dev/null; then
            quest_gripper_moved=true
            break
        fi
        sleep 0.25
    done
    if [[ "${quest_gripper_moved}" != true ]]; then
        printf 'Meta Quest gripper status:\n%s\n' "${quest_status}" >&2
        fail "Meta Quest B-button input did not move the gripper."
    fi

    sleep 0.5
    local quest_gripper
    quest_gripper="$(api_get /api/status | jq -r '.status.joints.gripper_1')"
    ros2 topic pub --rate 20 --times 5 /triskel/teleop/meta_quest/joy sensor_msgs/msg/Joy \
        '{axes: [0.0, 0.0, 0.0, 0.0], buttons: [0, 0, 0, 0, 0, 0]}' >/dev/null
    ros2 topic pub --rate 20 --times 5 /triskel/teleop/meta_quest/joy sensor_msgs/msg/Joy \
        '{axes: [0.0, 0.0, 0.0, 0.0], buttons: [0, 0, 1, 0, 0, 0]}' >/dev/null
    wait_for_home "${quest_gripper}"
}
