Usage Guide
===========

Here's how you can use Photon Weave in your quantum optics project.

Basic Example:
--------------

.. code-block:: python

   from photon_weave import QuantumCircuit

   # Create a basic quantum circuit
   circuit = QuantumCircuit()

   # Add components
   circuit.add_beam_splitter(...)
   circuit.add_phase_shift(...)

   circuit.simulate()