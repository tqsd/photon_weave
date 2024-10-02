"""fock envelope simulator"""
from setuptools import find_packages, setup

setup(
    name="photon_weave",
    version="0.1.0",
    author="Simon Sekavčnik",
    author_email="simon.sekavcnik@tum.de",
    description="General Quantum Simulator, with focus on optics",
    license="Apache 2.0",
    packages=find_packages(where="."),
    install_requires=["jax", "jaxlib"],
)
