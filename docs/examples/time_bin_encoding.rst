Time Bin Encoding
=================

Time bin encoding is a fundamental concept in quantum optics. This tutorial demonstrates how to simulate time bin encoding using the `photon_weave` package. The Simulation makes use of beam splitters and phase shifters in a structured setup.

Overview
--------

Time Bin Encoding Setup
^^^^^^^^
The time bin encoding process involves:
    * Four Beam Splitters
    * Two Phase Shifters

 A high level diagram is as follows:
 
.. code:: python

    """ 
        ┍━[PS:a]━┑    ┍━[PS:b]━┑
        │        │    │        │
    ━━━[/]━━━━━━[\]━━[/]━━━━━━[\]━━◽
                               ┃
                              ◽
   """
The goal is to simulate the interaction of photons through these components and measure the outcomes at various points in time.

Function Description
^^^^^^^^
.. code: python
   def time_bin_encoding(alpha:float, beta:float) -> List[List[int]]

Parameters:
    * `alpha` (float): Phase Shift for the first arm
    * `beta`  (float): Phase Shift for the second arm

Returns:
* `List[List[int]]`: Measurement outcomes at three time intervals. Each list contains two integers representing the outcomes at the top and bottom detectors


Implementation Steps
-----------

1. Imports
^^^^^^^^^^
First import all of the needed libraries and objects in the top of the file.

.. code:: python

   from typing import List
   import wax.numpy as jnp

   from photon_weave.operation import (
       CompositeOperationType, FockOperationType, Operation
   )
   from photon_weave.state.composite_envelope import CompositeEnvelope
   from photon_weave.state.envelope import Envelope


2. Initialize Envelopes
^^^^^^^^^^
Create an envelope with a photon state and an empty envelope.

.. code:: python

    env1 = Envelope()
    env1.fock.state = 3
    env2 = Envelope()


3. Define the Operations
^^^^^^^^^^
Define the beam splitter and phase shifter. Notice that `alpha` and `beta` parameters are received from the function arguments.

.. code:: python

    bs = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi/4)
    s1 = Operation(FockOperationType.PhaseShift, phi=alpha)
    s2 = Operation(FockOperationType.PhaseShift, phi=beta)

  
4. First Beam Splitter Application
^^^^^^^^^^

Combine the two envelopes and apply the first beam splitter. After the beam splitter one part of the pulse goes through phase shifter.

.. code:: python
    ce = CompositeEnvelope(env1, env2)
    ce.apply_operation(bs, env1.fock, env2.fock)

    ce.apply_operation(ps, env1.fock)

    
4. Second Beam Splitter Application
^^^^^^^^^^

Since the two arms have different length, the envelopes enter the beam splitter at two different times, so we consider vacuum state in the other input. To simulate this, we create two empty envelopes.

.. code:: python

    tmp_env_0_1 = Envelope()
    tmp_env_0_2 = Envelope()

    ce = CompositeEnvelope(ce, tmp_env_0_1, tmp_env_0_2)
    ce.apply_operation(bs, env2.fock, tmp_env_0_1.fock)
    ce.apply_operation(bs, tmp_env_0_2.fock, env1.fock)


Part of the pulses exits the experiment through the one open port of the second beam splitter. In our case the two systems `tmp_env_0_1` and `tmp_env_0_2` are escaping the simulation. We do not need to do anything, just be aware that the states are still included in the product state.

5. Third Beam Splitter
^^^^^^^

Due to the different arm lengths in the first part of the experiment the two envelopes go through the third beam splitter at two different times. Thus we need to create new vacuum state envelopes as in the previous case.
Part of the two (now split) pulses travel again through the phase shifter.

.. code:: python

    env_1_1 = Envelope()
    env_1_2 = Envelope()

    ce = CompositeEnvelope(ce, env_1_1, env_1_2)
    ce.apply_operation(bs, env1.fock, env_1_1.fock)
    ce.apply_operation(bs, env2.fock, env_1_2.fock)

    ce.apply(ps2, env1.fock)
    ce.apply(ps2, env2.fock)
    

5. Fourth Beam Splitter
^^^^^^^
Now we need to correctly combine the envelopes in the last beam splitter. We can consider three times:
    * at :math:`t_0`: an envelope that traveled the shortest path is entering the beam splitter alone `env_1_2`
    * at :math:`t_1`: an envelope which traveled through the short path after the first beam splitter and longer path after the third beam splitter enters the last beam splitter together with the envelope which traveled the longer path after the first beam splitter and longer path after the third beam splitter (`env2, env_1_1`).
    * at :math:`t_2`: an envelope that traveled the longest path is entering the beam splitter alone `env1`

.. code:: python

    env_2_1 = Envelope()
    env_2_2 = Envelope()
    ce = CompositeEnvelope(ce, env_2_1, env_2_2)
    ce.combine(env2.fock, env_2_1.fock)
    ce.combine(env2.fock, env_1_1.fock)
    ce.combine(env1.fock, env_2_2.fock)
    # t_1
    ce.apply_operation(bs, env_1_2.fock, env_2_1.fock)
    # t_2
    ce.apply_operation(bs, env2.fock, env_1_1.fock)
    # t_3
    ce.apply_operation(bs, env1.fock, env_2_2.fock)


6. Measuring
^^^^^^^
Now we can measure the outcomes at the two detectors at the three different times.
At the end we return the measurement results in the list. Where each element in the list is indicating the outcome separately at the two detectors at the three different times.

.. code:: python

    # t_0 At the top detector
    m_t_0_0 = env_1_2.measure()[env_1_2.fock]
    # t_0 At the bottom detector
    m_t_0_1 = env_2_1.measure()[env_2_1.fock]

    m_t0 = (m_t_0_0, m_t_0_1)

    # t_1 At the top detector
    m_t_1_0 = env2.measure()[env2.fock]
    # t_1 At the bottom detector
    m_t_1_1 = env_1_1.measure()[env_1_1.fock]

    m_t1 = (m_t_1_0, m_t_1_1)

    # t_2 At the top detector
    m_t_2_0 = env1.measure()[env1.fock]
    # t_2 At the bottom detector
    m_t_2_1 = env_2_2.measure()[env_2_2.fock]
    m_t2 = (m_t_2_0, m_t_2_1)

    return (m_t0, m_t1, m_t2)


Execution
---------

Now we can execute our function, which simulates the time bin encoding:

.. code:: python

    if __name__ == "__main__:
        outcomes = time_bin_encodin(0,0.5)
