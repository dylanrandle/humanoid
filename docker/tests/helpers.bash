fail() {
    printf 'ERROR: %s\n' "$*" >&2
    return 1
}

run_test() {
    local description="$1"
    shift
    printf '\n==> %s\n' "${description}"
    "$@"
}

assert_contains() {
    local haystack="$1"
    local needle="$2"
    local description="$3"
    grep -Fq -- "${needle}" <<<"${haystack}" || fail "Missing ${description}: ${needle}"
}

api_get() {
    curl --fail --silent --show-error "http://127.0.0.1:8765${1}"
}

api_post() {
    curl --fail --silent --show-error \
        --header 'Content-Type: application/json' \
        --data "${2}" \
        "http://127.0.0.1:8765${1}"
}

wait_for_status() {
    local timeout_seconds="$1"
    local error_message="$2"
    local query="$3"
    shift 3
    local deadline=$((SECONDS + timeout_seconds))
    local status=""

    while ((SECONDS < deadline)); do
        status="$(api_get /api/status 2>/dev/null || true)"
        if jq -e "$@" "${query}" <<<"${status}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.25
    done

    if [[ -n "${status}" ]]; then
        printf 'Last dashboard status:\n%s\n' "${status}" >&2
    fi
    fail "${error_message}"
}

wait_for_home() {
    local expected_gripper="$1"
    wait_for_status 20 "Dashboard did not reach the Home state." '
        .status.mode == "idle"
        and ((.status.joints.arm_2 + 0.75) | fabs) < 0.04
        and ((.status.joints.arm_3 - 0.5) | fabs) < 0.04
        and ((.status.joints.arm_6 - 1.0) | fabs) < 0.04
        and ((.status.joints.gripper_1 - $gripper) | fabs) < 0.04
    ' --argjson gripper "${expected_gripper}"
}

wait_for_rest() {
    wait_for_status 20 "Dashboard did not reach the Rest state." '
        .status.mode == "idle"
        and ((.status.joints.arm_2 + 1.6) | fabs) < 0.04
        and ((.status.joints.arm_3 + 0.1) | fabs) < 0.04
        and ((.status.joints.arm_4 - 1.65) | fabs) < 0.04
        and ((.status.joints.arm_6 - 0.21) | fabs) < 0.04
        and ((.status.joints.gripper_1 + 2.2) | fabs) < 0.04
    '
}
