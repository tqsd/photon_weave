"""fock envelope simulator"""
from setuptools import find_packages, setup

setup(
    name="photon_weave",
    version="0.0.3",
    author="Simon Sekavčnik",
    author_email="simon.sekavcnik@tum.de",
    description="Fock Envelope Simulator",
    license="Apache 2.0",
    packages=find_packages(where="."),
    install_requires=["numpy", "scipy", "numba"],
)
