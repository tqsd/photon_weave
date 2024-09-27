.. photon_weave documentation master file, created by
   sphinx-quickstart on Fri Sep 27 12:42:52 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

photon_weave documentation
==========================

Welcome to Photon Weave's Documentation!
========================================

Photon Weave is a general-purpose quantum simulator framework that focuses on optical simulations in the Fock domain. The framework also includes a `CustomState` class, which can represent arbitrary quantum systems with a user-defined number of basis states. Photon Weave aims to be an easy-to-use simulator, abstracting away the complexities of product space management and operation applications for the user.

Photon Weave framework places special focus on the simulation of the optical states in so called `Envelopes`. An Envelope represents a pulse with central frequency. Quantum state within an envelope is represented as :math:`\mathcal{F}\otimes\mathcal{P}`, where :math:`\mathcal{F}` corresponds to the Fock space and :math:`\mathcal{P}` corresponds to the polarization space.


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
   
