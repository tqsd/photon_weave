---
title: "Photon-Weave"
tags:
 - Python
 - Quantum Optics Simulation
 - Quantum
  
authors:
 - name: Simon Sekavčnik
    orcid: 0000-0002-1370-9751
    affiliation: 1
 - name: Kareem El-Safty
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
Photon Weave is a quantum systems simulator designed to offer intuitive abstractions for simulating photonic quantum systems and their interactions in Fock space along with arbitrary custom Hilbert space. The simulator focuses on simplifying complex quantum state representations, such as photon pulses (envelopes) with polarization, making it more approachable for specialized quantum simulations. While general-purpose quantum simulation libraries such as QuTiP [@johansson2012qutip]provide robust tools for quantum state manipulations, they often require meticulous organization of operations for larger simulations, introducing complexity that can be automated. Photon Weave addresses this by abstracting such details, streamlining the simulation process, and allowing quantum systems to interact naturally as the simulation progresses.

In contracts to frameworks such as Qiskit [@wille2019ibm], which are primarily designed for qubit-based computations, Photon Weave excels at simulating continuous-variable quantum systems, particularly photons, as well as custom quantum states that can interact dynamically. Furthermore, Photon Weave offers a balance of flexibility and automation by deferring the joining of quantum spaces until it is necessary, enhancing computational efficiency. The simulator supports both CPU and GPU execution, ensuring scalability and performance for large-scale simulations. This is achieved by using the jax [@jax2018github] library.

# Statement of Need
Tools like QuTiP, Qiskit, and Strawberry Fields [@killoran2019strawberry] already exist for modeling quantum phenomena, but many of them either require extensive user control (QuTiP) or enforce rigid circuit structures (Strawberry Fields). Researchers in quantum optics and related fields need a tool that simplifies photonic systems simulations, supports dynamic interactions between custom quantum systems, and eliminates the need for a circuit model. Such a tool could be used to generate a library of devices and gates that closely model real-world devices, fostering greater collaboration among scientists in these fields.

# Photon Weave Overview
Photon Weave is a quantum simulation library designed for simulating any system, provided simulating hardware meets the computational resource requirements. With this simulator, users can create, manipulate, and measure quantum systems with ease.

## Photon Weave Implementation Details
In the following sections, we will describe the main features of Photon Weave; details about implementations and usage can be found in [the documentation](https://photon-weave.readthedocs.io).

### State Containers
Photon Weave's core functionality revolves around quantum state containers. States can be represented in three forms: `Label,` `Vector`, or `Matrix,` which progressively require more memory. These representations are automatically managed by Photon Weave, which will shrink representations where applicable to save resources. The framework provides state containers such as `Fock,` `Polarization,` `Envelope,` and `CustomState.`
- `Fock,` `Polarization,` and `CustomState` are basic state containers that hold the quantum state in any valid representation until the state is joined with other states.
- When states are joined, these containers store references to the `Envelope,` `CompositeEnvelope,` or both. This allows each container to understand its place within a larger product space and how it is tensorized.

### Envelopes
Photon Weave places a particular emphasis on the `Envelope` concept. An `Envelope` represents a pulse of light, where all photons are indistinguishable and share the same polarization, representing the $\mathcal{F}\otimes\mathcal{P}$ space. Initially, when the spaces are separable, their states are stored in the respective `Fock` and `Polarization` containers. In addition to the states, an `Envelope` holds important metadata such as wavelength and temporal profile.


### Composite Envelopes
When envelopes interact, such as at a beam splitter, their states need to be joined. In these cases, the necessary state data is extracted from their respective containers and tensorized into a product state. A `CompositeEnvelope` can contain multiple product spaces, which can be accessed from any of the contributing state containers. Additionally, `CompositeEnvelope` instances can be merged, allowing states within both envelopes to interact. Since any basic state can, in principle interact with any other state, `CustomState` instances can also be included in a `CompositeEnvelope.`

### Operations
Photon Weave provides several ways to perform operations on quantum states. All operations are created using specialized classes (`FockOperation,` `PolarizationOperation,` `CustomStateOperation,` and `CompositeOperation`), each designed to work on a specific type of state. Operations can be predefined, manually constructed, or generated using expressions with a context.

```python
context = {
   "n": lambda dims: number_operator(dims[0])
}
op = Operation(
 FockOperationType.Expression,
    expr=("expm", ("s_mult", -1j, jnp.pi, "n")),
    context=context,
)
```
Photon Weave optimizes resource usage by automatically adjusting the dimensionality of the Fock space when necessary, even within product states. This ensures that only the minimal required space is used. This dynamic resizing of the quantum state representations avoids unnecessary memory consumption.

Once an operation is defined, it can be applied to the state at any level. If the state is part of a product state, Photon Weave ensures that the operation is applied to the correct subspace. Additionally, quantum channels defined by Kraus operators can be applied to any desired state space.

### Measuring
Photon Weave offers a robust measurement framework for any state. By default, Fock spaces are measured in number basis, Polarization spaces are measured in computational basis, and `CustomState` is measured in the respective basis. Photon Weave also supports more precise measurement definitions, such as POVM measurement.

# Conclusion
Photon Weave is offered as an open-source quantum system simulator under the Apache-2.0 license, targeting researchers and developers who need an easy-to-use yet powerful simulation tool. One of the intended outcomes is to build a library of interoperable quantum device models powered by the Photon Weave framework.

# Acknowledgments
This work was financed by the Federal Ministry of Education and Research of Germany via grants 16KIS1598K,
16KISQ039, 16KISQ077 and 16KISQ168 as well as in the programme of “Souver¨an. Digital. Vernetzt.”.
Joint project 6G-life, project identification number: 16KISK002. We acknowledge further funding from the
DFG via grant NO 1129/2-1 and by the Bavarian Ministry for Economic Affairs (StMWi) via the project
6GQT and by the Munich Quantum Valley.

# References
