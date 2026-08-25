JAX API
=======

.. module:: moreau.jax

The JAX integration provides a ``Solver`` class whose ``solve`` method is compatible with ``jax.grad``, ``jax.vmap``, and ``jax.jit``.

Solver
------

.. py:class:: Solver(n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones, settings=None, jit=True)

   JAX solver with automatic device selection and gradient support.

   Supports two usage patterns:

   1. **Full signature:** ``solve(P_data, A_data, q, b)``
   2. **Two-step:** ``setup(P_data, A_data)`` then ``solve(q, b)``

   :param n: Number of primal variables
   :param m: Number of constraints
   :param P_row_offsets: CSR row pointers for P matrix (array-like). P must be full symmetric (both upper and lower triangles).
   :param P_col_indices: CSR column indices for P matrix (array-like)
   :param A_row_offsets: CSR row pointers for A matrix (array-like)
   :param A_col_indices: CSR column indices for A matrix (array-like)
   :param cones: Cone specification (moreau.Cones object)
   :param settings: Optional solver settings (moreau.Settings object)
   :param jit: If True (default), JIT-compile the solve method

   **Example (full signature):**

   .. code-block:: python

      import jax.numpy as jnp
      from moreau.jax import Solver
      import moreau

      cones = moreau.Cones(num_nonneg_cones=2)
      solver = Solver(
          n=2, m=2,
          P_row_offsets=jnp.array([0, 1, 2]),
          P_col_indices=jnp.array([0, 1]),
          A_row_offsets=jnp.array([0, 1, 2]),
          A_col_indices=jnp.array([0, 1]),
          cones=cones,
      )

      P_data = jnp.array([1.0, 1.0])
      A_data = jnp.array([1.0, 1.0])
      q = jnp.array([1.0, 1.0])
      b = jnp.array([0.5, 0.5])

      solution = solver.solve(P_data, A_data, q, b)
      print(solution.x)

   **Example (two-step):**

   .. code-block:: python

      solver = Solver(n=2, m=2, ...)
      solver.setup(P_data, A_data)  # Set matrices once
      solution = solver.solve(q, b)  # Solve with 2 args

   .. py:method:: setup(P_data, A_data)

      Set P and A matrix values for subsequent ``solve()`` calls.

      Gradients w.r.t. P and A are still computed when using the 2-arg ``solve()``.

      :param P_data: P matrix values, shape (nnzP,)
      :param A_data: A matrix values, shape (nnzA,)

   .. py:method:: solve(*args, warm_start=None)

      Solve the optimization problem.

      Two signatures supported:

      - ``solve(q, b)``: Uses P/A from ``setup()`` (raises ``RuntimeError`` if ``setup()`` not called)
      - ``solve(P_data, A_data, q, b)``: Full signature

      :param warm_start: Optional ``WarmStart`` or ``BatchedWarmStart`` from a
          previous solve (e.g. ``solution.to_warm_start()``). If the warm-started
          solve fails, it is automatically retried without warm start.
          Gradients do **not** flow through warm start values.
      :returns: JaxSolution namedtuple with x, z, s

   .. py:attribute:: info
      :type: JaxSolveInfo

      Metadata from the last ``solve()`` call. Returns ``None`` if ``solve()``
      has not been called yet.

      .. note::

         For ``jax.vmap`` calls, this returns info from the last single call,
         not the batched result.

   .. py:attribute:: device
      :type: str

      Active device name ('cpu' or 'cuda').

   .. py:attribute:: n
      :type: int

      Number of primal variables.

   .. py:attribute:: m
      :type: int

      Number of constraints.

   .. py:attribute:: construction_time
      :type: float

      Time spent constructing solver structure (seconds).

   .. py:attribute:: tune_result
      :type: TuneResult or None

      Result from auto-tuning on the first ``solve()`` call. Returns ``None``
      if auto-tune has not run (e.g. device and method were set explicitly,
      or ``solve()`` has not been called).


Gradient Computation
--------------------

Use ``jax.grad`` for automatic differentiation:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from moreau.jax import Solver
   import moreau

   cones = moreau.Cones(num_nonneg_cones=2)
   solver = Solver(
       n=2, m=2,
       P_row_offsets=jnp.array([0, 1, 2]),
       P_col_indices=jnp.array([0, 1]),
       A_row_offsets=jnp.array([0, 1, 2]),
       A_col_indices=jnp.array([0, 1]),
       cones=cones,
   )

   P_data = jnp.array([1.0, 1.0])
   A_data = jnp.array([1.0, 1.0])
   b = jnp.array([0.5, 0.5])
   solver.setup(P_data, A_data)

   # Define loss function
   def loss_fn(q):
       solution = solver.solve(q, b)
       return jnp.sum(solution.x)

   # Compute gradient
   q = jnp.array([1.0, 1.0])
   grad_q = jax.grad(loss_fn)(q)


Batching with vmap
------------------

Use ``jax.vmap`` for batched solving:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from moreau.jax import Solver
   import moreau

   cones = moreau.Cones(num_nonneg_cones=2)
   solver = Solver(
       n=2, m=2,
       P_row_offsets=jnp.array([0, 1, 2]),
       P_col_indices=jnp.array([0, 1]),
       A_row_offsets=jnp.array([0, 1, 2]),
       A_col_indices=jnp.array([0, 1]),
       cones=cones,
   )

   P_data = jnp.array([1.0, 1.0])
   A_data = jnp.array([1.0, 1.0])
   solver.setup(P_data, A_data)

   # Batch over q and b
   batched_solve = jax.vmap(solver.solve)

   q_batch = jnp.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]])
   b_batch = jnp.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

   solutions = batched_solve(q_batch, b_batch)
   print(solutions.x.shape)  # (4, 2)


JIT Compilation
---------------

The solver is JIT-compiled by default. For manual control:

.. code-block:: python

   # Disable JIT at construction
   solver = Solver(n=2, m=2, ..., cones=cones, jit=False)

   # Or JIT your own wrapper
   @jax.jit
   def my_solve(q, b):
       return solver.solve(q, b)


Data Types
----------

JaxSolution
~~~~~~~~~~~

.. py:class:: JaxSolution

   NamedTuple containing solution arrays (JAX pytree-compatible).

   .. py:attribute:: x
      :type: jax.Array

      Primal solution

   .. py:attribute:: z
      :type: jax.Array

      Dual variables

   .. py:attribute:: s
      :type: jax.Array

      Slack variables

   .. py:method:: to_warm_start()

      Create a ``WarmStart`` from this solution (converts JAX arrays to numpy).

      :rtype: WarmStart


JaxSolveInfo
~~~~~~~~~~~~

.. py:class:: JaxSolveInfo

   NamedTuple containing solver metadata (JAX pytree-compatible).

   .. py:attribute:: status
      :type: float

      Solver status as a float (``SolverStatus`` integer value cast to float for
      JAX pytree compatibility). Compare with ``int(info.status) == SolverStatus.Solved``.

   .. py:attribute:: obj_val
      :type: float

      Objective value

   .. py:attribute:: iterations
      :type: int

      IPM iterations

   .. py:attribute:: solve_time
      :type: float

      Solve time in seconds

   .. py:attribute:: setup_time
      :type: float

      Time setting matrix values (seconds)

   .. py:attribute:: construction_time
      :type: float

      Time constructing solver (seconds)
