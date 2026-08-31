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

setup(
    name=PACKAGE_NAME,
    version=manifest_text("version"),
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/static", [str(path) for path in Path("static").glob("*")]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer=MAINTAINER.text,
    maintainer_email=MAINTAINER.attrib["email"],
    description=manifest_text("description"),
    license=manifest_text("license"),
    entry_points={
        "console_scripts": [
            "dashboard = triskel_operator.node:main",
            "meta_quest_bridge = triskel_operator.quest_bridge:main",
        ],
    },
)
