test_robot_descriptions() {
    local bringup_share
    local xacro_file

    bringup_share="$(ros2 pkg prefix --share triskel_bringup)"
    xacro_file="${bringup_share}/urdf/triskel.urdf.xacro"
    xacro "${xacro_file}" use_mock_hardware:=true >"${test_directory}/mock.urdf"
    xacro "${xacro_file}" use_mock_hardware:=false >"${test_directory}/physical.urdf"

    grep -Fq '<plugin>mock_components/GenericSystem</plugin>' "${test_directory}/mock.urdf"
    grep -Fq '<plugin>sts_hardware_interface/STSHardwareInterface</plugin>' \
        "${test_directory}/physical.urdf"
}
