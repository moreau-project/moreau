Core API
========

.. module:: moreau

The core API provides NumPy-based solvers for conic optimization.

Solver
------

.. py:class:: Solver(P, q, A, b, cones, settings=None)

   Single-problem conic optimization solver.

   All problem data is provided in the constructor, then call ``solve()``.

   **Problem Formulation:**

   .. code-block:: text

      minimize    (1/2)x'Px + q'x
      subject to  Ax + s = b
                  x in K1,  s in K2

   K2 constrains the slack s; K1 constrains x directly (direct-x cones).

   :param P: Quadratic objective matrix (scipy sparse or numpy array). Must be full symmetric (both triangles).
   :param q: Linear objective vector, shape (n,)
   :param A: Constraint matrix (scipy sparse or numpy array), shape (m, n)
   :param b: Constraint RHS vector, shape (m,)
   :param cones: Cone specification (``moreau.Cones`` object)
   :param settings: Optional solver settings (``moreau.Settings`` object)

   **Example:**

   .. code-block:: python

      import moreau
      from scipy import sparse
      import numpy as np

      P = sparse.diags([1.0, 1.0], format='csr')
      q = np.array([2.0, 1.0])
      A = sparse.csr_array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
      b = np.array([1.0, 0.7, 0.7])
      cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

      solver = moreau.Solver(P, q, A, b, cones)
      solution = solver.solve()
      print(solution.x, solver.info.status)

   .. py:method:: solve(warm_start=None)

      Solve the optimization problem.

      :param warm_start: Optional ``WarmStart`` from a previous solve
          (e.g. ``solution.to_warm_start()``). If the warm-started solve
          fails, it is automatically retried without warm start.
      :returns: Solution object with primal/dual solution vectors (x, z, s)
      :rtype: Solution

   .. py:method:: backward(dx, dz=None, ds=None, dz_x=None)

      Compute gradients via implicit differentiation using cached state from the
      last ``solve()`` call.

      Requires ``enable_grad=True`` in settings.

      :param dx: Gradient w.r.t. primal solution x, shape (n,)
      :param dz: Optional gradient w.r.t. dual variables z, shape (m,)
      :param ds: Optional gradient w.r.t. slack variables s, shape (m,)
      :param dz_x: Optional gradient w.r.t. direct-x cone duals z_x, shape (sum \|J\|,)
      :returns: Dict of gradients with keys ``dP_values``, ``dq``, ``dA_values``, ``db`` —
          sparse P/A gradients have the same CSR structure as the input matrices.
      :rtype: dict

   .. py:attribute:: info
      :type: SolveInfo

      Metadata from the last ``solve()`` call. Returns ``None`` if ``solve()``
      has not been called yet.

   .. py:attribute:: device
      :type: str

      The active device ('cpu' or 'cuda').

   .. py:attribute:: n
      :type: int

      Number of primal variables.

   .. py:attribute:: m
      :type: int

      Number of constraints.

   .. py:attribute:: construction_time
      :type: float

      Time spent constructing solver structure (seconds).


CompiledSolver
--------------

.. py:class:: CompiledSolver(n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones, settings=None)

   Compiled solver for batched problems with shared structure.

   Uses a three-step API: construct with structure, ``setup()`` matrix values,
   then ``solve()`` with parameters.

   :param n: Number of primal variables
   :param m: Number of constraints
   :param P_row_offsets: CSR row pointers for P matrix. P must be full symmetric (both upper and lower triangles).
   :param P_col_indices: CSR column indices for P matrix
   :param A_row_offsets: CSR row pointers for A matrix
   :param A_col_indices: CSR column indices for A matrix
   :param cones: Cone specification (``moreau.Cones`` object)
   :param settings: Solver settings (``moreau.Settings`` object)

   **Example:**

   .. code-block:: python

      import moreau
      import numpy as np

      cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
      settings = moreau.Settings(batch_size=4)

      solver = moreau.CompiledSolver(
          n=2, m=3,
          P_row_offsets=[0, 1, 2], P_col_indices=[0, 1],
          A_row_offsets=[0, 2, 3, 4], A_col_indices=[0, 1, 0, 1],
          cones=cones, settings=settings
      )

      solver.setup(P_values=[1., 1.], A_values=[1., 1., 1., 1.])
      solution = solver.solve(qs=[[2., 1.]]*4, bs=[[1., 0.7, 0.7]]*4)
      print(solution.x.shape)  # (4, 2)

   .. py:method:: setup(P_values, A_values)

      Set P and A matrix values for the batch.

      :param P_values: P matrix values. Shape (batch, nnz_P) or (nnz_P,) if shared.
      :param A_values: A matrix values. Shape (batch, nnz_A) or (nnz_A,) if shared.

   .. py:method:: solve(qs, bs, warm_start=None)

      Solve a batch of problems.

      When ``auto_tune=True`` and ``device='auto'`` or ``direct_solve_method='auto'``,
      the first call benchmarks candidate configurations and locks in the fastest
      for all subsequent solves (see :doc:`../guide/device-selection`). By default
      (``auto_tune=False``), a heuristic picks the configuration without benchmarking.

      :param qs: Linear cost vectors, shape (batch, n)
      :param bs: Constraint RHS vectors, shape (batch, m)
      :param warm_start: Optional ``WarmStart`` or ``BatchedWarmStart`` from a
          previous solve (e.g. ``solution.to_warm_start()``). If the warm-started
          solve fails, it is automatically retried without warm start.
      :returns: Batched solution with arrays of shape (batch, n) or (batch, m)
      :rtype: BatchedSolution

   .. py:method:: backward(dx, dz=None, ds=None, dz_x=None)

      Compute gradients via implicit differentiation using cached state from the
      last ``solve()`` call.

      Requires ``enable_grad=True`` in settings.

      :param dx: Gradient w.r.t. primal solutions x, shape (batch, n)
      :param dz: Optional gradient w.r.t. dual variables z, shape (batch, m)
      :param ds: Optional gradient w.r.t. slack variables s, shape (batch, m)
      :param dz_x: Optional gradient w.r.t. direct-x cone duals z_x, shape (batch, sum \|J\|)
      :returns: Dict of gradients with keys ``dP_values``, ``dq``, ``dA_values``, ``db`` —
          sparse P/A gradients have the same CSR structure as the input matrices.
      :rtype: dict

   .. py:attribute:: info
      :type: BatchedSolveInfo

      Metadata from the last ``solve()`` call. Returns ``None`` if ``solve()``
      has not been called yet.

   .. py:attribute:: device
      :type: str

      The active device ('cpu' or 'cuda').

   .. py:attribute:: n
      :type: int

      Number of primal variables.

   .. py:attribute:: m
      :type: int

      Number of constraints.

   .. py:attribute:: batch_size
      :type: int

      The batch size.

   .. py:attribute:: construction_time
      :type: float

      Time spent constructing solver structure (seconds).

   .. py:attribute:: tune_result
      :type: TuneResult or None

      Result from auto-tuning on the first ``solve()`` call. Returns ``None``
      if auto-tune has not run (e.g. device and method were set explicitly,
      or ``solve()`` has not been called). Also available on
      ``moreau.torch.Solver`` and ``moreau.jax.Solver``.


Cones
-----

.. py:class:: Cones(num_zero_cones=0, num_nonneg_cones=0, so_cone_dims=None, num_exp_cones=0, power_alphas=None, gen_power_cone_params=None, psd_dims=None, x_cones=None)

   Specification for the cone structure: the slack cones K2 (``s in K2``) and,
   via ``x_cones``, the direct-x cones K1 (``x in K1``).

   Constraints must be ordered in A and b to match:
   zero cones, then nonnegative, then second-order, then exponential, then power,
   then generalized power, then PSD.

   :param num_zero_cones: Number of equality constraints (zero cone, any dimension)
   :param num_nonneg_cones: Number of inequality constraints (nonnegative cone)
   :param so_cone_dims: List of second-order cone dimensions (each >= 2). For example,
       ``so_cone_dims=[3, 5, 10]`` creates three SOC cones of dimensions 3, 5, and 10.
       Defaults to empty list. As a backward-compatible shortcut, passing
       ``num_so_cones=N`` instead constructs ``so_cone_dims=[3] * N`` (N cones of dim 3).
   :param num_exp_cones: Number of exponential cones (dimension 3)
   :param power_alphas: List of alpha values for power cones (dimension 3 each, alpha in (0,1)). Defaults to empty list.
   :param gen_power_cone_params: List of ``(alphas, dim2)`` tuples for generalized power cones.
       Each ``alphas`` is a list of positive floats summing to 1 (length = dim1), and ``dim2 >= 1``.
       Total cone dimension is ``dim1 + dim2``. For example,
       ``gen_power_cone_params=[([0.3, 0.7], 2)]`` creates one cone with dim1=2, dim2=2 (total dim=4).
       Defaults to empty list.

       .. note::

          CUDA backward (gradient) accuracy for generalized power cones with total
          dimension >= 15 is a known limitation. CPU backward works at all dimensions.
          For high-dimensional GenPowerCone gradients, use ``device='cpu'`` or verify
          with finite differences.
   :param psd_dims: List of PSD cone matrix dimensions (each >= 1). For example,
       ``psd_dims=[3, 4]`` creates two PSD cones for 3x3 and 4x4 symmetric matrices,
       consuming 6 + 10 = 16 constraint rows (svec representation). Supported on both
       CPU and CUDA backends, including backward pass (gradient computation).
       See :doc:`../guide/psd-cones` for details. Defaults to empty list.
   :param x_cones: List of ``XConeSpec`` direct-x cone specifications that constrain
       subvectors of the primal variable ``x`` directly, without consuming rows of
       ``A``/``b``. Indices across all entries must be pairwise disjoint.
       See :doc:`../guide/direct-x-cones` for details. Defaults to empty list.

   .. py:property:: num_so_cones
      :type: int

      Number of second-order cones (read-only, computed from ``len(so_cone_dims)``).

   .. py:property:: num_power_cones
      :type: int

      Number of power cones (read-only, computed from ``len(power_alphas)``).

   .. py:property:: num_gen_power_cones
      :type: int

      Number of generalized power cones (read-only, computed from ``len(gen_power_cone_params)``).

   .. py:property:: num_psd_cones
      :type: int

      Number of PSD (SDP) cones (read-only, computed from ``len(psd_dims)``).

   .. py:method:: total_constraints()

      Total number of scalar constraints.

      :returns: Sum of all cone dimensions

   .. py:method:: degree()

      Degree of the cone (barrier function degree).

      :returns: Barrier function degree


SolverType
----------

.. py:class:: SolverType

   Solver algorithm type enum.

   .. py:attribute:: IPM

      Interior-point method (default). High accuracy, moderate speed.
      Supports automatic differentiation.


Settings
--------

.. py:class:: Settings(solver='auto', device='auto', device_id=-1, batch_size=1, enable_grad=False, auto_tune=False, max_iter=200, time_limit=inf, verbose=False, yolo=False, yolo_num_iters=15, ipm_settings=None, active_set_settings=None)

   Solver configuration.

   :param solver: Solver algorithm. ``'auto'`` (default) picks the active-set solver for small QPs and IPM otherwise. Other options: ``'ipm'`` or ``'active_set'`` (CPU only, zero+nonneg cones).
   :param device: Device selection (``'auto'``, ``'cpu'``, ``'cuda'``)
   :param device_id: CUDA device ID. ``-1`` (default) uses the current device. Ignored when ``device='cpu'``.
   :param batch_size: Batch size for CompiledSolver (default 1)
   :param enable_grad: Enable gradient computation for ``backward()`` (default False)
   :param auto_tune: If ``True``, benchmark solver configurations on the first ``solve()`` call when ``device='auto'`` or ``direct_solve_method='auto'``. If ``False`` (default), use heuristic selection without benchmarking.
   :param max_iter: Maximum IPM iterations (default 200, must be >= 1)
   :param time_limit: Time limit in seconds (default infinity, must be > 0)
   :param verbose: Print solver progress (default False)
   :param yolo: If ``True``, run a fixed number of IPM iterations with no convergence checking and no GPU↔host sync. All batches return ``MaxIterations`` status. Incompatible with ``enable_grad=True``. Default False.
   :param yolo_num_iters: Number of iterations to run in YOLO mode (default 15, must be >= 1).
   :param ipm_settings: Fine-grained IPM settings (``IPMSettings`` object). Auto-created with defaults if ``None``.
   :param active_set_settings: Fine-grained active-set settings (``ActiveSetSettings`` object). Auto-created with defaults if ``None``.


ActiveSetSettings
-----------------

.. py:class:: ActiveSetSettings(primal_tol=1e-6, dual_tol=1e-12, iter_limit=10000, diff_method='exact', diff_smoothing_mu=1e-4)

   Active-set solver settings (CPU only, zero + nonnegative cones).

   :param primal_tol: Primal feasibility tolerance (default 1e-6, must be > 0)
   :param dual_tol: Dual feasibility tolerance (default 1e-12, must be > 0)
   :param iter_limit: Maximum iterations (default 10000, must be >= 1)
   :param diff_method: Differentiation method. ``'exact'`` (default) computes the
       sensitivity of the active set's KKT system directly; ``'smoothed'`` adds a
       small μ-regularizer so gradients are well-defined on the active-set boundary.
   :param diff_smoothing_mu: Smoothing parameter μ for ``diff_method='smoothed'``
       (default 1e-4, must be > 0). Ignored when ``diff_method='exact'``.


IPMSettings
-----------

.. py:class:: IPMSettings(tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8, tol_infeas_abs=1e-8, tol_infeas_rel=1e-8, tol_ktratio=1e-6, max_step_fraction=0.99, equilibrate_enable=True, direct_solve_method='auto', reduced_tol_gap_abs=5e-5, reduced_tol_gap_rel=5e-5, reduced_tol_feas=1e-4, reduced_tol_infeas_abs=5e-12, reduced_tol_infeas_rel=5e-5, reduced_tol_ktratio=1e-4, warm_start_no_retry=None)

   Interior-point method settings.

   **Convergence tolerances** (all must be > 0):

   :param tol_gap_abs: Absolute duality gap tolerance (default 1e-8)
   :param tol_gap_rel: Relative duality gap tolerance (default 1e-8)
   :param tol_feas: Feasibility tolerance (default 1e-8)
   :param tol_infeas_abs: Absolute infeasibility detection tolerance (default 1e-8)
   :param tol_infeas_rel: Relative infeasibility detection tolerance (default 1e-8)
   :param tol_ktratio: KKT ratio tolerance (default 1e-6)

   **Algorithm control:**

   :param max_step_fraction: Maximum step size fraction per iteration (default 0.99, must be in (0, 1])
   :param equilibrate_enable: Enable matrix equilibration preprocessing (default True)
   :param direct_solve_method: KKT solver method. Valid options:

      - ``'auto'`` — benchmarked on first solve (default)
      - ``'qdldl'`` — CPU only, best for small/sparse KKT systems
      - ``'faer'`` — CPU only, multi-threaded, best for large CPU problems
      - ``'faer-1t'`` — CPU only, single-threaded faer variant
      - ``'faer-nt'`` — CPU only, faer with automatic thread count
      - ``'cudss'`` — CUDA only, best for large GPU problems
      - ``'riccati'`` — CPU / CUDA, specialized for block-tridiagonal KKT (MPC/LQR problems)
      - ``'woodbury'`` — CUDA only, specialized for diagonal P + low-rank A (portfolio-type problems)

   **Reduced tolerances** (for ``AlmostSolved`` convergence, all must be > 0):

   :param reduced_tol_gap_abs: Reduced absolute duality gap tolerance (default 5e-5)
   :param reduced_tol_gap_rel: Reduced relative duality gap tolerance (default 5e-5)
   :param reduced_tol_feas: Reduced feasibility tolerance (default 1e-4)
   :param reduced_tol_infeas_abs: Reduced absolute infeasibility tolerance (default 5e-12)
   :param reduced_tol_infeas_rel: Reduced relative infeasibility tolerance (default 5e-5)
   :param reduced_tol_ktratio: Reduced KT ratio tolerance (default 1e-4)

   **Warm start retry control:**

   :param warm_start_no_retry: Set of ``SolverStatus`` values that do **not** trigger
       an automatic cold retry when warm starting. Default (``None``) uses
       ``{Solved, AlmostSolved, MaxIterations, CallbackTerminated}``. All other statuses
       trigger an automatic cold retry with a warning. Pass a custom ``frozenset``
       of ``SolverStatus`` values to override.


Data Types
----------

Solution
~~~~~~~~

.. py:class:: Solution

   Single-problem solution.

   .. py:attribute:: x
      :type: numpy.ndarray

      Primal solution, shape (n,)

   .. py:attribute:: z
      :type: numpy.ndarray

      Dual variables, shape (m,)

   .. py:attribute:: s
      :type: numpy.ndarray

      Slack variables, shape (m,)

   .. py:attribute:: z_x
      :type: numpy.ndarray or None

      Direct-x cone duals, flat over the ``Cones.x_cones`` spec in order.
      Shape ``(sum |J|,)`` where each ``|J|`` is the size of a direct-x cone's
      primal-index set. ``None`` for problems with no direct-x cones.

   .. py:method:: to_warm_start()

      Create a ``WarmStart`` from this solution (copies arrays).

      :rtype: WarmStart


BatchedSolution
~~~~~~~~~~~~~~~

.. py:class:: BatchedSolution

   Batched solution with array outputs.

   .. py:attribute:: x
      :type: numpy.ndarray

      Primal solutions, shape (batch, n)

   .. py:attribute:: z
      :type: numpy.ndarray

      Dual variables, shape (batch, m)

   .. py:attribute:: s
      :type: numpy.ndarray

      Slack variables, shape (batch, m)

   .. py:attribute:: z_x
      :type: numpy.ndarray

      Direct-x cone duals, shape ``(batch, sum |J|)``. Empty (zero-width second
      axis) for problems with no direct-x cones.

   Supports indexing (``solution[i]`` returns a ``Solution``), ``len()``, and iteration.

   .. py:method:: to_warm_start()

      Create a ``BatchedWarmStart`` from this solution (copies arrays).

      :rtype: BatchedWarmStart


WarmStart
~~~~~~~~~

.. py:class:: WarmStart

   Warm start point for a single conic optimization problem.

   Contains primal, dual, and slack variables from a previous solve that can be
   used to accelerate convergence on a related problem. Create via
   ``solution.to_warm_start()``.

   .. py:attribute:: x
      :type: numpy.ndarray

      Primal variables, shape (n,)

   .. py:attribute:: z
      :type: numpy.ndarray

      Dual variables, shape (m,)

   .. py:attribute:: s
      :type: numpy.ndarray

      Slack variables, shape (m,)


BatchedWarmStart
~~~~~~~~~~~~~~~~

.. py:class:: BatchedWarmStart

   Warm start point for batched conic optimization problems.

   Contains primal, dual, and slack variables from a previous batched solve.
   Create via ``batched_solution.to_warm_start()``.

   Supports indexing (``ws[i]`` returns a ``WarmStart``), ``len()``, and iteration.

   .. py:attribute:: x
      :type: numpy.ndarray

      Primal variables, shape (batch_size, n)

   .. py:attribute:: z
      :type: numpy.ndarray

      Dual variables, shape (batch_size, m)

   .. py:attribute:: s
      :type: numpy.ndarray

      Slack variables, shape (batch_size, m)


SolveInfo
~~~~~~~~~

.. py:class:: SolveInfo

   Metadata from a solve.

   .. py:attribute:: status
      :type: SolverStatus

      Solve outcome

   .. py:attribute:: obj_val
      :type: float

      Objective value at solution

   .. py:attribute:: iterations
      :type: int

      Number of IPM iterations

   .. py:attribute:: solve_time
      :type: float

      Time in IPM iterations (seconds)

   .. py:attribute:: setup_time
      :type: float

      Time setting matrix values (seconds)

   .. py:attribute:: construction_time
      :type: float

      Time constructing solver (seconds)


BatchedSolveInfo
~~~~~~~~~~~~~~~~

.. py:class:: BatchedSolveInfo

   Metadata from a batched solve.

   .. py:attribute:: status
      :type: list[SolverStatus]

      Per-problem solve outcome (one per problem in batch)

   .. py:attribute:: obj_val
      :type: list[float]

      Per-problem objective values

   .. py:attribute:: iterations
      :type: list[int]

      Per-problem iteration counts

   .. py:attribute:: solve_time
      :type: float

      Time in IPM iterations (seconds)

   .. py:attribute:: setup_time
      :type: float

      Time setting matrix values (seconds)

   .. py:attribute:: construction_time
      :type: float

      Time constructing solver (seconds)



TuneResult
~~~~~~~~~~

.. py:class:: TuneResult

   Result from auto-tuning on the first ``solve()`` call (when ``auto_tune=True``
   and ``device='auto'`` or ``direct_solve_method='auto'``).

   .. py:attribute:: device
      :type: str

      Winning device name (e.g. ``'cuda'``, ``'cpu'``)

   .. py:attribute:: method
      :type: str

      Winning KKT solver method (e.g. ``'cudss'``, ``'faer'``, ``'qdldl'``)

   .. py:attribute:: time_limit
      :type: float

      Time limit set for future solves (seconds), computed as ``best_time * margin``

   .. py:attribute:: results
      :type: dict

      Per-method benchmark data. Keys are ``'device:method'`` strings (e.g.
      ``'cuda:cudss'``) when cross-device tuning, or plain method names when
      single-device. Values are dicts with ``solve_time``, ``iterations``, and ``status``.


SolverStatus
~~~~~~~~~~~~

.. py:class:: SolverStatus

   Enum (``IntEnum``) indicating solve outcome. Supports direct comparison
   (``status == SolverStatus.Solved``) and int comparison (``status == 1``).

   .. py:attribute:: Unsolved
      :value: 0

      Not yet solved

   .. py:attribute:: Solved
      :value: 1

      Optimal solution found

   .. py:attribute:: PrimalInfeasible
      :value: 2

      Problem has no feasible solution

   .. py:attribute:: DualInfeasible
      :value: 3

      Problem is unbounded

   .. py:attribute:: AlmostSolved
      :value: 4

      Near-optimal solution (reduced tolerances met)

   .. py:attribute:: AlmostPrimalInfeasible
      :value: 5

      Near-certain primal infeasibility (reduced tolerances)

   .. py:attribute:: AlmostDualInfeasible
      :value: 6

      Near-certain dual infeasibility (reduced tolerances)

   .. py:attribute:: MaxIterations
      :value: 7

      Iteration limit reached

   .. py:attribute:: MaxTime
      :value: 8

      Time limit reached

   .. py:attribute:: NumericalError
      :value: 9

      Numerical issues encountered

   .. py:attribute:: InsufficientProgress
      :value: 10

      Solver stalled (insufficient progress between iterations)

   .. py:attribute:: CallbackTerminated
      :value: 11

      Terminated by user callback


Device Functions
----------------

.. py:function:: available_devices()

   List all available devices, sorted by priority (highest first).

   Always includes ``'cpu'`` as a fallback. When CUDA is available it appears
   first (priority 100 vs CPU priority 0).

   :returns: List of device names (e.g., ``['cuda', 'cpu']`` or ``['cpu']``)
   :rtype: list[str]

.. py:function:: device_available(device)

   Check if a specific device is available.

   :param device: Device name (``'cpu'``, ``'cuda'``, etc.)
   :returns: True if available
   :rtype: bool

.. py:function:: device_error(device)

   Return the error message if device backend discovery failed.

   Useful for diagnosing why a device (e.g. CUDA) is not available.

   :param device: Device name
   :returns: Error message string, or ``None`` if no error
   :rtype: str or None

.. py:function:: default_device()

   Return the effective default device (after applying priority and overrides).

   :returns: Device name (e.g., ``'cpu'``, ``'cuda'``)
   :rtype: str

.. py:function:: get_default_device()

   Get the current default device override.

   :returns: Device name if an override is set, otherwise None
   :rtype: str or None

.. py:function:: set_default_device(device)

   Set the global default device.

   :param device: Device name (``'cpu'``, ``'cuda'``, etc.) or ``None`` to reset to automatic selection
   :raises ValueError: If the specified device is not available

.. py:function:: torch_available(device=None)

   Check if PyTorch integration is available.

   :param device: Optional device name. If ``None``, checks if any device has torch support.
   :returns: True if PyTorch solver is available
   :rtype: bool

.. py:function:: jax_available(device=None)

   Check if JAX integration is available.

   :param device: Optional device name. If ``None``, checks if any device has JAX support.
   :returns: True if JAX solver is available
   :rtype: bool
