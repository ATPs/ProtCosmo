"""Setuptools configuration for ProtCosmo."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def load_version() -> str:
    version_path = ROOT / "VERSION"
    version = version_path.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"VERSION file is empty: {version_path}")
    return version


def load_requirements() -> list[str]:
    requirements_path = ROOT / "requirements.txt"
    if not requirements_path.exists():
        return []
    requirements: list[str] = []
    for raw in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


setup(
    name="protcosmo",
    version=load_version(),
    description="Faster way of searching dark proteins.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=load_requirements(),
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "protcosmo=protcosmo.protcosmo:main",
        ]
    },
)
