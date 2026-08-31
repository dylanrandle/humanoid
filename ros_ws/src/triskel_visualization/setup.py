from pathlib import Path
from xml.etree import ElementTree as ET

from setuptools import find_packages, setup

PACKAGE_DIRECTORY = Path(__file__).resolve().parent
MANIFEST = ET.parse(PACKAGE_DIRECTORY / "package.xml").getroot()


def manifest_text(tag: str) -> str:
    value = MANIFEST.findtext(tag)
    if value is None:
        raise RuntimeError(f"package.xml is missing {tag}")
    return value


PACKAGE_NAME = manifest_text("name")
MAINTAINER = MANIFEST.find("maintainer")
if MAINTAINER is None or MAINTAINER.text is None or "email" not in MAINTAINER.attrib:
    raise RuntimeError("package.xml must define a maintainer name and email")
INSTALL_REQUIRES = [
    "setuptools",
    *(PACKAGE_DIRECTORY / "requirements.txt").read_text().splitlines(),
]

setup(
    name=PACKAGE_NAME,
    version=manifest_text("version"),
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
    ],
    install_requires=INSTALL_REQUIRES,
    zip_safe=True,
    maintainer=MAINTAINER.text,
    maintainer_email=MAINTAINER.attrib["email"],
    description=manifest_text("description"),
    license=manifest_text("license"),
    entry_points={
        "console_scripts": [
            "visualizer = triskel_visualization.node:main",
        ],
    },
)
