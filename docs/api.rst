API Overview
============

This page highlights the primary user-facing entry points and where to find details.

Core Concepts
-------------
- Envelope and CompositeEnvelope orchestration (`photon_weave.state.envelope`, `photon_weave.state.composite_envelope`).
- Operations (`photon_weave.operation`): Fock, Polarization, Custom, and Composite operations.
- Configuration (`photon_weave.photon_weave.Config` and `Session`) for seeds, contractions, and JIT flags.

Measurement
-----------
- Current APIs:
  - `Envelope.measure`, `Envelope.measure_POVM`, `Envelope.measure_expectation`
  - `CompositeEnvelope` measurement methods
  - Utility functions in `photon_weave.state.utils.measurements`
- Near-term roadmap:
  - Unified `MeasurementSpec`/`MeasurementResult` and shot support (see `measurement_api_todo.md`).

Operations
----------
- High-level: `Envelope.apply_operation`, `Envelope.apply_kraus`, `CompositeEnvelope.apply_operation`
- Low-level helpers: `photon_weave.state.utils.operations` (documented NumPy-style) and adapters in `photon_weave.core.adapters`.

Reference
---------
.. toctree::
   :maxdepth: 2

   Full API (autodoc) <photon_weave>
   Core Internals <photon_weave.core>
   State API (envelopes & composites) <photon_weave.state>
   State Utilities <photon_weave.state.utils>
   Operation API <photon_weave.operation>
   Operation Helpers <photon_weave.operation.helpers>
   Constants <photon_weave.constants>
   Extras & Interop <photon_weave.extra>
