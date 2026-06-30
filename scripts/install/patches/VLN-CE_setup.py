from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="vlnce",
    version="0.1.0",
    packages=find_packages(
        include=[
            "vlnce_baselines",
            "vlnce_baselines.*",
            "habitat_extensions",
            "habitat_extensions.*",
            "vlnce_server",
            "vlnce_server.*",
        ]
    ),
)
