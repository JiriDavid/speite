"""
Setup configuration for Speite - Offline Speech-to-Text System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text(encoding="utf-8").splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="speite",
    version="0.1.0",
    author="Speite Team",
    description="Offline speech-to-text system for low-connectivity environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JiriDavid/speite",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "speite-server=main:main",
            "speite-cli=cli:main",
        ],
    },
    include_package_data=True,
    keywords="speech-to-text whisper offline transcription audio",
    project_urls={
        "Bug Reports": "https://github.com/JiriDavid/speite/issues",
        "Source": "https://github.com/JiriDavid/speite",
    },
)
