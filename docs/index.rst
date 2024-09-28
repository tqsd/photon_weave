.. photon_weave documentation master file, created by
   sphinx-quickstart on Fri Sep 27 12:42:52 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Photon Weave's Documentation!
=============================================

Photon Weave is a general-purpose quantum simulator framework that focuses on optical simulations in the Fock domain. The framework also includes a `CustomState` class, which can represent arbitrary quantum systems with a user-defined number of basis states. Photon Weave aims to be an easy-to-use simulator, abstracting away the complexities of product space management and operation applications for the user.

Mathematical Introduction
===========================

The **Photon Weave** framework isi built around the representation and manipulation fo optical states in various Hilbert spaces. The primary spaces used within the framework are the **Fock Space**, the **Polarization Space**, and a general **Custom Hilbert Space**.

Fock Space
----------

The **Fock space** :math:`\mathcal{F}` is a Hilbert space that describes quantum state with discrete photon numbers. Each state in the Fock space is represented by a photon number basis :math:`|n\rangle`, where :math:`n=0,1,2,\ldots` denotes the number of photons. The Fock states are orthonormal and satifsy the following relation ship:

.. math::
   \langle n | m \rangle = \delta_{nm}

where :math:`\delta_{nm}` is the Kronecker delta. The Fock space is ideal for representing quantum light states, such as single-photon states, coherent states, and squeezed states.

In **Photon Weave**, a single Fock space can be represented in three different ways:
 - **Label**: Minimizes the memory required for representation, represented by a label (integer). Labels can represent any basis state such as :math:`|m\rangle`, where :math:`m \in \mathbb{N}`.
 - **State Vector**: Represents the state as a vector. This representation can describe any pure state, including global phases.
 - **Density Matrix**: Represents the state in a density matrix formalism. It can represent both pure and mixed states. Note that global phases are not preserved in this representation

Transition between the **Label**, **State Vector** and **Density Matrix** representatiosn can be easily achieved using `expand()` or `contract()` methods. However, contractions may not always be succesfull if the state is not representable in the target form. For instance, contracting mixed state to a vector state will fail without raising an error. It is also important to node that contracting **Density matrix** to **State vector** also disregards the global phase.

Polarization Space
-----------------

The **Polarization space** :math:`\mathcal{P}` describes the polarization state, usually of some Envelope. The polarization space is a two-dimensional hilbert space :math:`\mathcal{H}^2`. With the following basis states:

.. math::
   \begin{align}
   |H\rangle &= \begin{bmatrix} 1 \\ 0 \end{bmatrix} \\
   |V\rangle &= \begin{bmatrix} 0 \\ 1 \end{bmatrix} \\
   |R\rangle &= \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{i}{\sqrt{2} \end{bmatrix} \\
   |L\rangle &= \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{-i}{\sqrt{2} \end{bmatrix}
   \end{align}


The **Polarization Space** can also be represented in three different forms:
 - **Label**: Minimizes the memory required for representation, represented by a `PolarizationLabel` enum class. Labels can represent the follwing states :math:`\{|H\rangle, |V\rangle, |R\rangle, |L\rangle\}`.
 - **State Vector**: Represents the state as a vector. This representation can describe any pure state, including global phases.
 - **Density Matrix**: Represents the state in a density matrix formalism. It can represent both pure and mixed states. Note that global phases are not preserved in this representation

Transition between the **Label**, **State Vector** and **Density Matrix** representatiosn can be easily achieved using `expand()` or `contract()` methods. However, contractions may not always be succesfull if the state is not representable in the target form. For instance, contracting mixed state to a vector state will fail without raising an error. It is also important to node that contracting **Density matrix** to **State vector** also disregards the global phase.
 
Custom Hilbert Space
---------------------
The **Custom Hilbert Space** :math:`\mathcal{H}^d` can describe any arbitrary finite dimensional quantum system Hilbert space. When creating `CustomState(d)` the dimensionality of that space needs to be provided through :math:`m`.

The **Custom Hilbert Space** can also be represented in three different forms:
 - **Label**: Minimizes the memory required for representation, represented by an integer.
 - **State Vector**: Represents the state as a vector. This representation can describe any pure state, including global phases.
 - **Density Matrix**: Represents the state in a density matrix formalism. It can represent both pure and mixed states. Note that global phases are not preserved in this representation

Transition between the **Label**, **State Vector** and **Density Matrix** representatiosn can be easily achieved using `expand()` or `contract()` methods. However, contractions may not always be succesfull if the state is not representable in the target form. For instance, contracting mixed state to a vector state will fail without raising an error. It is also important to node that contracting **Density matrix** to **State vector** also disregards the global phase.

Envelopes and Composite Envelopes
====================================
Since **Photon Weave** focuses on quantum optics, it defines an **Envelope**, which can be interpreted as a temporal mode or pulse. An Envelope is characterized by its central wavelength, temporal profile, Fock space and polarization space. The Fock and Polarization spaces define the photon number state and the polarization state, respectively. When the states are separate, they are contained within the individual Fock and Polarization instances. These states can be combined into a product space by using `combine()` method of the **Envelope** instance. When this method is invoked, the states are extracted from the Fock and Polarization instances, combined and stored in the `Envelope` instance. The Fock and Polarization instances hold references to the `Envelope` instance which contains their states.

The product state in an **Envelope** can be represented in a **State Vector** formalism or **Density Matrix** formalism. You can use `contract()` or `expand()` method on `Envelope` instance or any participating states to change between these representations.

When dealing with larger product states, such as two **Envelopes** or an **Envelope** and **Custom Hilbert State**, **Composite Envelopes** are used. **Composite Envelopes** offer a robust product space management system, where the product spaces are correctly maintained and dynamically created during the simulation. In short, when applying operations to the state in a **Composite Envelope** the product spaces are managed automatically to support any operation. Product spaces can only be created between state instances that share the same `CompositeEnvelope`. When state is represented in a `CompositeEnvelope` its state is extracted, and its instance then holds a reference to the `CompositeEnvelope` instance. Operations can then be applied to the state instance, envelope or composite envelope. **Photon Weave** will correctly route the operations to the appropriate level.

For example, if a Fock state instance is a part of an `Envelope`, which in turn is a part of `CompositeEnvelope`, but its space has not yet been extracted, you can apply operation that acts only on the said Fock space to any of the state contianers (`CompostieEnvelope`, `Envelope`, `Fock`) and **Photon Weave** will then route the operation to the appropriate `Fock` instance.

This logic relieves the user from having to implement complex product space tracking and management, making simulations more straightforward and intuitive.

Applying Operations
=====================

**Photon Weave** ships with some predefined oprations. Operations can be defined in the following way:
.. code-block:: python
		import jax.numpy as jnp
		from photon_weave.operations import (
		    Operation, FockOperationType, PolarizationOperationType,
		    CustomStateOperationType, CompostieOperationType
		)
		fock_operation = Operation(FockOperationType.PhaseShift, phi=jnp.pi)
		polarization_operation = Operation(PolarizationOperationType.X)
		custom_state_operation = Operation(CustomStateOperationType.Expression, expr=expr)
		composite_opreation = Operation(
		    CompositeOperationType.NonPolarizingBeamSplitter,
		    eta=jnp.pi/4
		)
Each operation is defined thorugh `Operation` class. The type of operation is defined with the first parameter, additional parameters must then be defined through key word arguments. Some predefined operatrions require specific key word arguments, like for example in the case of non polarizing beam splitter operation. Beside the predefined operation, user can also create `Custom` operator for the base states (`FockOperationType`, `PolarizationOperationType`, `CustomStateOPerationType`), where the operator matrix needs to be passed as a `operator` key word paramter.

.. code-block:: python
		operator = jnp.array(
		    [[0,0],
		     [1,0]]
		)
		op = Operation(
		    FockOperationType.Custom,
		    operator=operator
		)
		 



Applying Quantum Channels
=============================


Measuring
===========


Measuring with POVM Operators
=================================






.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   API
   faq
   modules
   photon_weave
   photon_weave.state
   photon_weave.operation
   
