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
   |R\rangle &= \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{i}{\sqrt{2}} \end{bmatrix} \\
   |L\rangle &= \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{-i}{\sqrt{2}} \end{bmatrix}
   \end{align}


The **Polarization Space** can also be represented in three different forms:
 - **Label**: Minimizes the memory required for representation, represented by a `PolarizationLabel` enum class. Labels can represent the follwing states :math:`\{|H\rangle, |V\rangle, |R\rangle, |L\rangle\}`.
 - **State Vector**: Represents the state as a vector. This representation can describe any pure state, including global phases.
 - **Density Matrix**: Represents the state in a density matrix formalism. It can represent both pure and mixed states. Note that global phases are not preserved in this representation

Transition between the **Label**, **State Vector** and **Density Matrix** representatiosn can be easily achieved using `expand()` or `contract()` methods. However, contractions may not always be succesfull if the state is not representable in the target form. For instance, contracting mixed state to a vector state will fail without raising an error. It is also important to node that contracting **Density matrix** to **State vector** also disregards the global phase.
 
Custom Hilbert Space
---------------------
The **Custom Hilbert Space** :math:`\mathcal{H}^d` can describe any arbitrary finite dimensional quantum system Hilbert space. When creating `CustomState(d)` the dimensionality of that space needs to be provided through :math:`d`.

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


**Photon Weave** ships with intuitive operation definition and application. All operations are defined through `Operation` class. First argument when defining the operation is the operation type. There are four operation types `Enum` classes defined:
- `FockOperationType`: `Enum` class defining operations on Fock spaces.
- `PolarizationOperationType`: `Enum` class defining operations on polarization spaces.
- `CustomStateOperationType`: `Enum` class defining operations on Custom State spaces.
- `CompositeOperationType`: `Enum` class defining operations on multiple spaces. This operation type can be applied to a product space with multiple spaces, where the spaces can be an arbitrary type.

Defining and applying operators is as straight forward as defining an operator and applying it to a space:

.. code:: python
    
    from photon_weave.operation import (
        Operation, FockOperationType,
        PolarizationOperationType,
        CustomStateOperaitonType,
        CompositeOperationType
    )

    fock = Fock()
    fock_op = Operation(FockOperationType.Displace, alpha=0.5)
    fock.apply_operation(fock_op)

    polarization = Polarization()
    polarization_op = Operation(PolarizationOperationType.RX, theta=0.5)
    polarization.apply_operation(polarization_op)

    custom_state = CustomState(3)

    custom_operator = jnp.array(
        [[0, 0, 0],
         [1, 0, 0],
    	[0, 1, 0]]
    )
    custom_state_op= Operation(
        CustomStateOperation.Custom,
        operator = custom_operator
    )
    custom_state.apply_operation(custom_state_op)

    env=Envelope()
    # Envelope can extract the state into a product state
    env.combine()

    # Apply operation in an envelope
    env.apply_operation(fock_op, env.fock)
    env.apply_operation(polarization_op, env.polarization)
    # Or apply operation on individual spaces
    env.fock.apply_operation(fock_op)
    env.polarization.apply_operation(polarization_operation)


    ce = CompositeEnvelope(env, custom_state)
    # Composite Envelope can combine the states in any configuration
    ce.combine(env.fock, env.polarization, custom_state)

    # Operations can be applied at the state level
    env.fock.apply_operation(fock_operation)
    env.polarization.apply_operaiton(polarization_op)
    custom_state.apply_operation(custom_state_op)

    # Operations can be applied at the envelope level
    env.apply_operation(fock_op, env.fock)
    env.apply_operation(polarization_op, env.polarization)

    # Operations can also be applied at the composite operation level
    ce.apply_operation(fock_op, env.fock)
    ce.apply_operation(polarization_op, env.polarization)
    ce.apply_operation(custom_state_op, custom_state)

    # Additionaly Composite operations can be applied only in
    # Composite envelope
    bs = Operation(
        CompositeOperaiton.NonPolarizingBeamSplitter,
	theta=jnp.pi/4
    )

    # Beam Split operation requires additional Fock space
    env2 = Envelope()

    # Add the new envelope to the composite envelope
    ce = CompositeEnvelope(ce, env2)

    ce.apply_operation(bs, env.fock, env2.fock)
    


Fock Operations
^^^^^^^^^^^^^^^^


Operations on Fock spaces are defined through `FockOperationType` class. `FockOperationType` will size the defined operator to the appropriate size before applying, so user doesn't need to explicitly control the dimensions. Furthermore in some cases, post operation state requires more dimensions in order to accurately represent the state. **Photon Weave** tries to compute number of dimensions needed to correctly represent the state. **Photon Weave** implements some of the common operations: (creation, annihilation, phase shift, squeezing, displacing and identity. Addinitonally the user can define an operator using an `Expression` or manually providing an operator with the `Custom` enumeration.

Fock operations can be applied on three levels, depending on the situation. If the fock state is in some product space, either in `Envelope` or in `CompositeEnvelope`, **Photon Weave** will correctly route the operation to the appropriate space.

To see the list of implemented operations on the consult the `fock_operation.py`.

Polarization Operations
^^^^^^^^^^^^^^^^^^^^^

Operations on Polarization spaces are defined through `PolarizationOperationType` class. Since `Polarization` is always two dimensional Hilber space, the operations need to be of same dimensions. Beside the implemented operators, Polarization operation also implements a `Custom` operation, but not `Expression` operation types.

To see the list of implemented operations and required parameters for each of them, consult `polarization_operation.py`.

Custom State Operations
^^^^^^^^^^^^^^^^^^^^^^

Custom State Operations define operations which operate on `CustomState`. Custom state operations only defines two types of operations: `Custom` and `Expression`, requiring the user to explicitly define the operators either through expression or by providing the operator manually. Keep in mind that the dimensionalty of the operator must match the dimensionality of the space, on which the operator will act.

Further documentation can be found at `custom_state_operation.py`.

Composite Operations 
^^^^^^^^^^^^^^^^^^^^

Composite Operations define an operatrions on multiple spaces. Prior to operation the states must be members of the same composite envelope. These operations must be applied to the composite envelope and correct order of the spaces it acts on must be given. The order of operator tensoring must reflect the order of given spaces in the `apply_operation` method call.

Custom Operators
^^^^^^^^^^^^^^^^

Custom operation is a simple way of manually providing an operation. The user must make sure that the dimensionalty of the operator matches the dimensionality of the target space. In case of `FockOperationType.Custom`, **Photon Weave** will resize the state to the dimensionality of the operator. If the operator has smaller dimension than the underlying state, the **Photon Weave** will try to shirnk the state, but the shrinking process may fail if part of the state would fall outside of the new dimension cutoff.


Expression defined operators
^^^^^^^^^^^^^^^^^^^^^^^^^^

Some operations types offer `Expression` defined operators. When defining `Expression` type of operation, the user must give at least two key word arguments: `expr` and `context`.

`context` must be a dictionary, with keys of `str` type and values must be of a `Callable` type. Each callable must consume one argument. Each callable must compute a matrix operator, with dimensionality given as a parameter:

.. code:: python
   
    context = {
       "a_dag": lambda dims: creation_operator(dims[0])
       "a":     lambda dims: annihilation_operator(dims[0])
       "n":     lambda dims: number_operator(dims[0])
    }

In some cases the operation dimensions are not necessary:

.. code:: python

    context = {
    "a": lambda dims: jnp.array([[0,0,0],[1,0,0],[0,0,0]]),
    "b": lambda dims: jnp.array([[0,0,0],[0,0,0],[0,1,0]])
    }

In those cases the `Callable` must still consume one dimension parameter, even if it doesn't use it.

The `expr` expression is then constructed in a Lisp inspired way with tuples. Tuples are evaluated from the inner most tuple to the outer most one. The first element in every single tuple is a string, which defines the operation. Following arguments are operands, on which the operation is evaluated.

To find out more and examples see `expression_interpreter.py`.

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
   
