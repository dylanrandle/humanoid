test_named_poses() {
    local home_status

    home_status="$(api_post /api/mode '{"mode":"home"}')"
    jq -e '.status.mode == "homing" or .status.mode == "idle"' \
        <<<"${home_status}" >/dev/null
    wait_for_home 0.0

    api_post /api/mode '{"mode":"rest"}' >/dev/null
    wait_for_rest

    api_post /api/mode '{"mode":"home"}' >/dev/null
    wait_for_home 0.0
}
