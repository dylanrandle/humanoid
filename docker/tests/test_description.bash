test_robot_descriptions() {
    local bringup_share
    local xacro_file

    bringup_share="$(ros2 pkg prefix --share triskel_bringup)"
    xacro_file="${bringup_share}/urdf/triskel.urdf.xacro"
    xacro "${xacro_file}" use_mock_hardware:=true >"${test_directory}/mock.urdf"
    xacro "${xacro_file}" use_mock_hardware:=false >"${test_directory}/physical.urdf"

    grep -Fq '<plugin>sts_hardware_interface/STSHardwareInterface</plugin>' \
        "${test_directory}/mock.urdf"
    grep -Fq '<param name="enable_mock_mode">true</param>' "${test_directory}/mock.urdf"
    grep -Fq '<plugin>sts_hardware_interface/STSHardwareInterface</plugin>' \
        "${test_directory}/physical.urdf"
    grep -Fq '<param name="enable_mock_mode">false</param>' "${test_directory}/physical.urdf"

    # Only wheels expose signed velocity commands. Arm/gripper trajectories use
    # position commands, with both hard position and URDF velocity limits enabled.
    local description
    for description in mock physical; do
        local urdf="${test_directory}/${description}.urdf"
        [[ "$(grep -Fc '<command_interface name="position">' "${urdf}")" -eq 8 ]]
        [[ "$(grep -Fc '<command_interface name="velocity">' "${urdf}")" -eq 3 ]]
        if grep -Eq '<command_interface name="acceleration"|<limits enable="false"' "${urdf}"; then
            fail "Unexpected servo-profile command or disabled ROS limit in ${description}."
        fi
    done
}
