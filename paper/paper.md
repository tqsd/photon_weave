---
title: "Photon-Weave"
tags:
 - Python
 - Quantum Optics Simulation
 - Quantum Photonics
 - Quantum Physics
 - JAX
 - Quantum Computing
 
  
authors:
 - name: Simon Sekavčnik
   orcid: 0000-0002-1370-9751
   affiliation: 1
 - name: Kareem H. El-Safty
   orcid: 0000-0001-8740-0637
   affiliation: 1
 - name: Janis Nötzel
   orcid: 0000-0003-0091-3072
   affiliation: 1
affiliations:
 - name: Technical University of Munich, Theoretical Quantum System Design, Munich, Germany
   index: 1
date: 15.10.2024
bibliography: paper.bib
---
# Summary
Photon Weave is a quantum systems simulator designed to offer intuitive abstractions for simulating quantum photonic systems and their interactions in Fock spaces [@Fock1932] and any custom Hilbert spaces. The simulator focuses on simplifying complex quantum state representations, such as continuous photonic states with polarization using envelopes that can mimic pulses, making it more approachable for specific quantum simulations. While general-purpose quantum simulation libraries such as QuTiP [@johansson2012qutip] provide robust tools for quantum state manipulations, some require advanced software skills to manipulate complex system interactions. Photon Weave addresses this by abstracting such details, streamlining the simulation process, and allowing quantum systems to interact naturally as the simulation progresses.

In contracts to other frameworks such as Qiskit [@aleksandrowicz2019qiskit], which are primarily designed for qubit-based computations, Photon Weave excels at simulating continuous-variable quantum systems, mainly photons, as well as custom quantum states that can interact dynamically. Furthermore, Photon Weave offers a balance of flexibility and automation by deferring the joining of quantum spaces using State Containers until necessary, enhancing computational efficiency. The simulator supports CPU and GPU execution, ensuring scalability and performance for large-scale simulations. This is achieved by using the JAX [@jax2018github] library.

# Statement of Need
Tools like QuTiP, Qiskit, Piquasso, and StrawberryFields [@kolarovszki2024piquassophotonicquantumcomputer; @killoran2019strawberry] already exist for modeling quantum phenomena, but many of them either require extensive user control (QuTiP) or enforce rigid circuit structures (StrawberryFields). Researchers in quantum optics and related fields need a tool that simplifies photonic systems simulations and supports dynamic interactions between custom quantum systems. Photon Weave introduces such features without restricting itself to the circuit model so researchers can focus on component development. Such a tool could generate a library of components and gates that closely model real-world devices, fostering greater collaboration among scientists in those fields. In Fig. \ref{fig:1}, a complicated scenario of lossy Beam Splitters is depicted, and in Fig. \ref{fig:2}, the performance is compared to Qiskit and QuTip.


![The simulation of lossy Beam Splitters. The simulation tracks the state evolution throughout the experiment. The losses here are photon absorption. \label{fig:1}](circuit.png)


![Comparison between Photon Weave, Qiskit, and QuTip regarding simulation time and the required space to simulate the experiment in \autoref{fig:1}. The steps are the executed operations. \label{fig:2}](lossy_circuit_paper-2.png)

# Photon Weave Overview
Photon Weave is a quantum simulation library designed for simulating any system, provided that simulating hardware meets the resource requirements. This simulator allows users to easily create, manipulate, and measure quantum systems.

## Photon Weave Implementation Details
In the following sections, we will describe the main features of Photon Weave; details about implementations and usage can be found in [the documentation](https://photon-weave.readthedocs.io).

### State Containers
Photon Weave's core functionality revolves around quantum state containers. States can be represented in three forms: `Label,` `Vector,` or `Matrix,` which progressively require more memory. Photon Weave automatically manages these representations, reducing representations where applicable to save resources. The framework provides state containers such as `Fock,` `Polarization,` `Envelope,` and `CustomState.` `Fock,` `Polarization,` and `CustomState` are essential state containers that hold the quantum state in any valid representation until the state interacts with other states. When states interact, these containers store references to the `Envelope,` `CompositeEnvelope,` or both. This allows each container to understand its place within a larger product space and how it evolves mathematically.

### Envelopes
Photon Weave places a particular emphasis on the `Envelope` concept. An `Envelope` represents a pulse of light, where all photons are indistinguishable and share the same polarization, representing the $\mathcal{F}\otimes\mathcal{P}$ space where $\mathcal{F}$ represents the Fock space and $\mathcal{P}$ represents the Polarization space. Initially, when the states are separable, they are stored in the respective `Fock` and `Polarization` containers. In addition to the states, an `Envelope` holds essential metadata such as wavelength and temporal profile.

### Composite Envelopes
When envelopes interact, for example, using a beam-splitter [@xiang2002theorem], their states must be joined. The necessary state data are extracted from their respective containers, and their Hilbert spaces form a product space in these cases. A `CompositeEnvelope` can contain multiple product spaces, which can be accessed from any of the contributing state containers. Additionally, `CompositeEnvelope` instances can be merged, allowing states within both envelopes to interact. `CustomState` instances can also be included in a `CompositeEnvelope` since any custom state can, in principle, interact with any other state.

### Operations
Photon Weave provides several ways to perform operations on quantum states. All operations are created using the Operation type as well as one of the Enums: `FockOperationType,` `PolarizationOperationType,` `CustomStateOperationType,` and `CompositeOperationType` to further define what on which type of a state the operation will operate. Operations can be manually constructed or generated using expressions with a context along with the predefined ones. Photon Weave supports photonic operators such as `Squeezing,` `Displacement,` `Phase Shift,` `Beam Splitter,` and non-linear operations. It also supports Pauli operators.

Photon Weave optimizes resource usage by automatically adjusting the dimensionality of the Fock space when necessary, even within product states. This ensures that only the minimal required space is used, dynamically resizing the quantum state representation to avoid unnecessary memory consumption.

Once an operation is defined, it can be applied to an appropriate state at any level. If a state is a part of a product space, Photon Weave ensures that the operation is applied to the correct subspace. Additionally, Kraus operators can be applied to any desired state space. This allows the user to simulate losses at any level.

### Measurements
Photon Weave offers a robust measurement framework for any state. By default, Fock spaces are measured on a number basis, Polarization spaces are measured on a computational basis, and `CustomState` is measured on a respective basis. Photon Weave also supports more precise measurement definitions, such as Positive Operator Valued Measurement (POVM).

# Conclusion
Photon Weave is an open-source quantum system simulator under the Apache-2.0 license, targeting researchers and developers who need an easy-to-use yet powerful simulation tool. One of the intended outcomes is to build a library of interoperable quantum device models powered by the Photon Weave framework.

# Acknowledgments
This work was financed by the Federal Ministry of Education and Research of Germany via grants 16KIS1598K, 16KISQ039, 16KISQ077, and 16KISQ168 as well as in the program of "Souverän. Digital. Vernetzt.". Joint project 6G-life, project identification number: 16KISK002. We acknowledge further funding from the DFG via grant NO 1129/2-1 and by the Bavarian Ministry for Economic Affairs (StMWi) via the project 6GQT and the Munich Quantum Valley.

# References
