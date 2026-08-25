PyTorch API
===========

.. module:: moreau.torch

The PyTorch integration provides differentiable optimization layers with full autograd support.

Solver
------

.. py:class:: Solver(n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones, settings=None)

   Unified PyTorch solver with automatic device selection and autograd support.

   ``enable_grad`` is forced to ``True`` for PyTorch solvers (gradients are always enabled).

   :param n: Number of primal variables
   :param m: Number of constraints
   :param P_row_offsets: CSR row pointers for P matrix (torch.Tensor). P must be full symmetric (both upper and lower triangles).
   :param P_col_indices: CSR column indices for P matrix (torch.Tensor)
   :param A_row_offsets: CSR row pointers for A matrix (torch.Tensor)
   :param A_col_indices: CSR column indices for A matrix (torch.Tensor)
   :param cones: Cone specification (moreau.Cones object)
   :param settings: Optional solver settings (moreau.Settings object)

   **Example:**

   .. code-block:: python

      import torch
      from moreau.torch import Solver
      import moreau

      cones = moreau.Cones(num_nonneg_cones=2)
      settings = moreau.Settings(device='cuda', batch_size=64)

      solver = Solver(
          n=2, m=2,
          P_row_offsets=torch.tensor([0, 1, 2]),
          P_col_indices=torch.tensor([0, 1]),
          A_row_offsets=torch.tensor([0, 1, 2]),
          A_col_indices=torch.tensor([0, 1]),
          cones=cones,
          settings=settings,
      )

      solution = solver.solve(P_values, A_values, q, b)
      print(solver.info.status[0], solver.info.obj_val[0])

   .. py:method:: solve(P_values, A_values, q, b, *, warm_start=None)

      Solve the optimization problem.

      Stateless call — all problem data is passed as arguments.  Safe for
      chained solves in autograd graphs (e.g. MPC rollouts).  P/A setup is
      cached internally: if P and A haven't changed since the last call, the
      expensive setup step is skipped automatically.

      If any input has ``requires_grad=True``, the outputs support automatic
      differentiation via ``loss.backward()``.

      :param P_values: P matrix values, shape (batch, nnzP) or (nnzP,), dtype=float64
      :param A_values: A matrix values, shape (batch, nnzA) or (nnzA,), dtype=float64
      :param q: Linear cost vector, shape (batch, n) or (n,), dtype=float64
      :param b: Constraint RHS, shape (batch, m) or (m,), dtype=float64
      :param warm_start: Optional ``WarmStart`` or ``BatchedWarmStart`` from a
          previous solve (e.g. ``solution.to_warm_start()``). If the warm-started
          solve fails, it is automatically retried without warm start.
      :returns: Solution object (TorchSolution or TorchBatchedSolution)

   .. py:method:: backward(dx, dz=None, ds=None)

      Compute gradients via implicit differentiation.

      :param dx: Gradient w.r.t. primal solution x (torch.Tensor)
      :param dz: Optional gradient w.r.t. dual variables z (torch.Tensor)
      :param ds: Optional gradient w.r.t. slack variables s (torch.Tensor)
      :returns: Tuple of gradient tensors (dP, dA, dq, db)

   .. py:method:: setup_grad(batch_size=None)

      Pre-allocate memory for gradient computation (backward pass).

      Optional but recommended when calling ``backward()`` repeatedly.

      :param batch_size: Optional batch size for pre-allocation.

   .. py:method:: reset()

      Reset solver state.

   .. py:method:: get_dimensions()

      Return problem dimensions as a dict.

      :rtype: dict

   .. py:attribute:: info

      Metadata from the last ``solve()`` call. Returns ``None`` if ``solve()``
      has not been called yet. Type is ``TorchSolveInfo`` (single problem) or
      ``TorchBatchedSolveInfo`` (batched).

   .. py:attribute:: device
      :type: str

      Active device name ('cpu' or 'cuda').

   .. py:attribute:: n
      :type: int

      Number of primal variables.

   .. py:attribute:: m
      :type: int

      Number of constraints.

   .. py:attribute:: batch_size
      :type: int

      Current batch size.

   .. py:attribute:: is_initialized
      :type: bool

      Whether solver has been initialized (setup called).

   .. py:attribute:: nnzP
      :type: int

      Number of non-zeros in P matrix.

   .. py:attribute:: nnzA
      :type: int

      Number of non-zeros in A matrix.

   .. py:attribute:: tune_result
      :type: TuneResult or None

      Result from auto-tuning on the first ``solve()`` call. Returns ``None``
      if auto-tune has not run (e.g. device and method were set explicitly,
      or ``solve()`` has not been called).

   .. py:attribute:: grad_initialized
      :type: bool

      Whether ``setup_grad()`` has been called.


Gradient Computation
--------------------

Gradients flow through the solver via implicit differentiation:

.. code-block:: python

   import torch
   from moreau.torch import Solver
   import moreau

   cones = moreau.Cones(num_nonneg_cones=2)
   solver = Solver(
       n=2, m=2,
       P_row_offsets=torch.tensor([0, 1, 2]),
       P_col_indices=torch.tensor([0, 1]),
       A_row_offsets=torch.tensor([0, 1, 2]),
       A_col_indices=torch.tensor([0, 1]),
       cones=cones,
   )

   P_values = torch.tensor([1.0, 1.0], dtype=torch.float64)
   A_values = torch.tensor([1.0, 1.0], dtype=torch.float64)

   # Enable gradients on inputs
   q = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
   b = torch.tensor([0.5, 0.5], dtype=torch.float64)

   # Solve
   solution = solver.solve(P_values, A_values, q, b)

   # Backpropagate
   loss = solution.x.sum()
   loss.backward()

   # Access gradients
   print(q.grad)  # dL/dq


GPU Usage
---------

For GPU acceleration:

.. code-block:: python

   import torch
   from moreau.torch import Solver
   import moreau

   settings = moreau.Settings(device='cuda', batch_size=256)
   solver = Solver(n=2, m=2, ..., cones=cones, settings=settings)

   # Keep tensors on GPU
   P_values = torch.tensor([1., 1.], dtype=torch.float64, device='cuda')
   A_values = torch.tensor([1., 1.], dtype=torch.float64, device='cuda')
   q = torch.randn(256, 2, dtype=torch.float64, device='cuda', requires_grad=True)
   b = torch.randn(256, 2, dtype=torch.float64, device='cuda')

   solution = solver.solve(P_values, A_values, q, b)


Data Types
----------

TorchSolution
~~~~~~~~~~~~~

.. py:class:: TorchSolution

   Single-problem solution with PyTorch tensors.

   .. py:attribute:: x
      :type: torch.Tensor

      Primal solution, shape (n,)

   .. py:attribute:: z
      :type: torch.Tensor

      Dual variables, shape (m,)

   .. py:attribute:: s
      :type: torch.Tensor

      Slack variables, shape (m,)

   .. py:method:: to_warm_start()

      Create a ``WarmStart`` from this solution (detaches and moves to CPU).

      :rtype: WarmStart


TorchBatchedSolution
~~~~~~~~~~~~~~~~~~~~

.. py:class:: TorchBatchedSolution

   Batched solution with PyTorch tensors.

   Supports indexing (``solution[i]`` returns a ``TorchSolution``), ``len()``, and iteration.

   .. py:attribute:: x
      :type: torch.Tensor

      Primal solutions, shape (batch, n)

   .. py:attribute:: z
      :type: torch.Tensor

      Dual variables, shape (batch, m)

   .. py:attribute:: s
      :type: torch.Tensor

      Slack variables, shape (batch, m)

   .. py:method:: to_warm_start()

      Create a ``BatchedWarmStart`` from this solution (detaches and moves to CPU).

      :rtype: BatchedWarmStart


TorchSolveInfo
~~~~~~~~~~~~~~

.. py:class:: TorchSolveInfo

   Metadata from a single-problem PyTorch solve.

   .. py:attribute:: status
      :type: SolverStatus

      Solve outcome

   .. py:attribute:: obj_val
      :type: torch.Tensor

      Objective value tensor

   .. py:attribute:: iterations
      :type: torch.Tensor

      Iteration count tensor

   .. py:attribute:: solve_time
      :type: float

      Time in IPM iterations (seconds)

   .. py:attribute:: setup_time
      :type: float

      Time setting matrix values (seconds)

   .. py:attribute:: construction_time
      :type: float

      Time constructing solver (seconds)


TorchBatchedSolveInfo
~~~~~~~~~~~~~~~~~~~~~

.. py:class:: TorchBatchedSolveInfo

   Metadata from a batched PyTorch solve.

   .. py:attribute:: status
      :type: list[SolverStatus]

      Per-problem solve outcome

   .. py:attribute:: obj_val
      :type: torch.Tensor

      Per-problem objective values, shape (batch_size,)

   .. py:attribute:: iterations
      :type: torch.Tensor

      Iteration count tensor

   .. py:attribute:: solve_time
      :type: float

      Time in IPM iterations (seconds)

   .. py:attribute:: setup_time
      :type: float

      Time setting matrix values (seconds)

   .. py:attribute:: construction_time
      :type: float

      Time constructing solver (seconds)
