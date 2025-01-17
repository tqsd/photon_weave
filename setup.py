"""fock envelope simulator"""
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="photon_weave",
    version="0.1.4",
    author="Simon Sekavčnik, Kareem H. El-Safty, Janis Nötzel",
    author_email="simon.sekavcnik@tum.de",
    description="General Quantum Simulator, with focus on optics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    url="https://github.com/tqsd/photon_weave",
    project_urls={
        "Documentation": "https://photon-weave.readthedocs.io/en/latest/index.html",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(where="."),
    install_requires=["jax", "jaxlib"],
)
