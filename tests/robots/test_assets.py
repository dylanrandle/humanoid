"""Structural checks for checked-in robot description assets."""

import re
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

ASSETS_DIR = Path(__file__).resolve().parents[2] / "src" / "humanoid" / "robots" / "assets"
URDF_PATHS = sorted(ASSETS_DIR.glob("*/urdf/*.urdf"))
SRDF_PATHS = sorted(ASSETS_DIR.glob("*/srdf/*.srdf"))

SIMPLE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
PACKAGE_URI_PATTERN = re.compile(r"^package://([^/]+)/(.+)$")
VERSION_PATTERN = re.compile(r"(?:^|_)v\d+(?:_|$)")


def test_asset_file_names_are_simple_and_version_free():
    for path in ASSETS_DIR.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        assert SIMPLE_NAME_PATTERN.fullmatch(path.stem), f"Generated asset name: {path}"
        assert not VERSION_PATTERN.search(path.stem), f"Versioned asset name: {path}"
        assert path.suffix == path.suffix.lower()


def test_obj_material_references_exist():
    for obj_path in ASSETS_DIR.rglob("*.obj"):
        for line in obj_path.read_text().splitlines():
            if line.startswith("mtllib "):
                material_path = obj_path.parent / line.removeprefix("mtllib ")
                assert material_path.exists(), f"Missing material referenced by {obj_path}"


@pytest.mark.parametrize("urdf_path", URDF_PATHS, ids=lambda path: path.name)
def test_urdf_names_and_mesh_references_are_consistent(urdf_path: Path):
    root = ET.parse(urdf_path).getroot()
    robot_name = urdf_path.stem.removesuffix("_collision")
    assert root.attrib["name"] == robot_name

    links = [element.attrib["name"] for element in root.findall("link")]
    joints = [element.attrib["name"] for element in root.findall("joint")]
    assert len(links) == len(set(links))
    assert len(joints) == len(set(joints))
    assert all(SIMPLE_NAME_PATTERN.fullmatch(name) for name in [*links, *joints])
    assert not any(VERSION_PATTERN.search(name) for name in [*links, *joints])

    link_names = set(links)
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        assert parent is not None
        assert child is not None
        assert parent.attrib["link"] in link_names
        assert child.attrib["link"] in link_names

    for mesh in root.iter("mesh"):
        uri = mesh.attrib["filename"]
        match = PACKAGE_URI_PATTERN.fullmatch(uri)
        assert match is not None, f"Unsupported mesh URI in {urdf_path}: {uri}"
        package_name, relative_path = match.groups()
        mesh_path = ASSETS_DIR / package_name / relative_path
        assert mesh_path.exists(), f"Missing mesh referenced by {urdf_path}: {mesh_path}"
        assert SIMPLE_NAME_PATTERN.fullmatch(mesh_path.stem)
        assert not VERSION_PATTERN.search(mesh_path.stem)


@pytest.mark.parametrize("srdf_path", SRDF_PATHS, ids=lambda path: path.name)
def test_srdf_references_primary_urdf_names(srdf_path: Path):
    srdf = ET.parse(srdf_path).getroot()
    urdf_path = srdf_path.parents[1] / "urdf" / f"{srdf_path.stem}.urdf"
    urdf = ET.parse(urdf_path).getroot()
    link_names = {element.attrib["name"] for element in urdf.findall("link")}
    joint_names = {element.attrib["name"] for element in urdf.findall("joint")}

    for element in srdf.iter():
        for attribute in ("link1", "link2", "parent_link"):
            if name := element.attrib.get(attribute):
                assert name in link_names, f"Unknown {attribute} in {srdf_path}: {name}"
        if element.tag == "joint":
            name = element.attrib["name"]
            assert name in joint_names, f"Unknown joint in {srdf_path}: {name}"
