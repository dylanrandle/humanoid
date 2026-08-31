test_recording_and_replay() {
    local record_response
    local recording_id
    local recording_path

    record_response="$(api_post /api/recording/start '{}')"
    recording_id="$(jq -r '.status.recording.id' <<<"${record_response}")"
    [[ -n "${recording_id}" && "${recording_id}" != null ]] \
        || fail "Dashboard did not return a recording identifier."

    sleep 1
    api_post /api/mode '{"mode":"keyboard"}' >/dev/null
    for _ in {1..5}; do
        api_post /api/teleop '{"commands":["base_left"]}' >/dev/null
        sleep 0.1
    done
    api_post /api/teleop '{"commands":[]}' >/dev/null
    api_post /api/mode '{"mode":"idle"}' >/dev/null
    api_post /api/recording/stop '{}' >/dev/null

    recording_path="${test_directory}/recordings/${recording_id}/bag"
    [[ -f "${recording_path}/metadata.yaml" ]] \
        || fail "rosbag2 did not finalize the dashboard recording."
    api_get /api/status | jq -e --arg id "${recording_id}" \
        '.status.recordings | any(.id == $id)' >/dev/null

    api_post /api/replay/start "{\"recording\":\"${recording_id}\"}" >/dev/null
    wait_for_status 30 "Dashboard rosbag2 replay did not complete." '
        .status.replay.running == false and .status.replay.outcome == "completed"
    '
}
