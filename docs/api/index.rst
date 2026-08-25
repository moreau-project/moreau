API Reference
=============

Moreau provides a unified API across CPU and GPU backends, with integrations for PyTorch and JAX.

.. toctree::
   :maxdepth: 2

   core
   torch
   jax

Quick Reference
---------------

Core Classes
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``moreau.Solver``
     - Single problem solver
   * - ``moreau.CompiledSolver``
     - Batched solver with shared structure
   * - ``moreau.Cones``
     - Cone specification
   * - ``moreau.Settings``
     - Solver configuration
   * - ``moreau.IPMSettings``
     - IPM algorithm settings
   * - ``moreau.SolverType``
     - Solver algorithm type enum

Framework Integrations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Framework
     - Main Class
     - Features
   * - PyTorch
     - ``moreau.torch.Solver``
     - autograd, functional API
   * - JAX
     - ``moreau.jax.Solver``
     - jit, vmap, grad

Data Types
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Type
     - Description
   * - ``Solution``
     - Primal/dual solution (x, z, s)
   * - ``BatchedSolution``
     - Batched solution arrays
   * - ``WarmStart``
     - Warm start point for single problem
   * - ``BatchedWarmStart``
     - Warm start point for batched problems
   * - ``SolveInfo``
     - Solver metadata (status, timing)
   * - ``BatchedSolveInfo``
     - Batched solver metadata
   * - ``TuneResult``
     - Auto-tune benchmark results
   * - ``SolverStatus``
     - Solve outcome enum
   * - ``TorchSolution`` / ``TorchBatchedSolution``
     - PyTorch solution types
   * - ``TorchSolveInfo`` / ``TorchBatchedSolveInfo``
     - PyTorch metadata types
   * - ``JaxSolution``
     - JAX solution NamedTuple
   * - ``JaxSolveInfo``
     - JAX metadata NamedTuple
