"""Resolve canonical ROS description resources for browser visualization."""

from pathlib import Path
from xml.etree import ElementTree as ET

PACKAGE_URI_PREFIX = "package://triskel_description/"


def write_resolved_urdf(source: Path, package_share: Path, destination: Path) -> Path:
    """Write a temporary URDF whose mesh URIs are absolute local paths."""

    package_root = package_share.resolve()
    tree = ET.parse(source)
    for mesh in tree.getroot().iter("mesh"):
        uri = mesh.attrib.get("filename", "")
        if not uri.startswith(PACKAGE_URI_PREFIX):
            raise ValueError(f"Unsupported Triskel mesh URI: {uri}")

        relative_path = Path(uri.removeprefix(PACKAGE_URI_PREFIX))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe Triskel mesh URI: {uri}")

        # A colcon --symlink-install keeps individual resource files in the
        # source tree. Validate the URI lexically before resolving that valid
        # installed symlink instead of requiring its target to remain under
        # the package share directory.
        resolved = (package_root / relative_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Triskel visualization mesh is unavailable: {resolved}")
        mesh.set("filename", str(resolved))

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination
