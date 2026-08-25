#!/usr/bin/env python3
"""Security-Constrained DC Optimal Power Flow (SCOPF) with Moreau.

This example demonstrates solving Security-Constrained DC-OPF problems using
Moreau. The SCOPF formulation ensures the system remains feasible not only
under normal conditions, but also under N-1 contingencies (single line outages).

Problem Formulation:
    minimize    sum_g c_g * p_g + C_shed * sum_n load_shed_n   (generation + shedding cost)
    subject to
        --- Base Case Constraints ---
        sum_g p_g + load_shed - sum_d d = 0       (power balance at each bus)
        f_l = B_l * (theta_i - theta_j)           (DC power flow)
        p_g_min <= p_g <= p_g_max                 (generator limits)
        0 <= load_shed_n <= d_n                   (load shedding bounds)
        theta_ref = 0                             (reference bus angle)

        --- N-1 Contingency Constraints (added via Benders cuts) ---
        For each contingency k:
            f_l^k <= f_l_max for all monitored lines l

        where f_l^k = f_l + LODF_{l,k} * f_k      (post-contingency flow)

Algorithm:
    Uses Benders decomposition to iteratively add violated line limit constraints:
    1. Solve master problem (economic dispatch with accumulated cuts, no line limits initially)
    2. Check all line limits (base case + contingencies) for violations
    3. Add all violations as cuts to the master problem
    4. Repeat until no violations or max iterations reached

    Load shedding (default $5000/MWh) ensures feasibility when line limits are tight.

Key Concepts:
    - PTDF (Power Transfer Distribution Factors): Sensitivity of line flows to
      power injections. PTDF[l,n] = change in flow on line l per unit injection at bus n.

    - LODF (Line Outage Distribution Factors): How flow redistributes when a line
      is outaged. LODF[l,k] = fraction of flow on line k that transfers to line l
      when line k is outaged. Real LODFs are more complex and reflect multi-line
      outages or other types of outages such as load or generation. Mathematically,
      real LODFs behave similarly though, as a linear term in the contingency
      line flow constraints.

    - Post-contingency flow: f_l^post = f_l^pre + LODF[l,k] * f_k^pre

    - Load shedding: Flexible demand that can be curtailed at high cost to ensure
      feasibility when thermal limits cannot be satisfied.

Moreau Formulation:
    Variables: x = [p_g, theta, f, load_shed]

    Line limit constraints are added as Benders cuts:
        f_l + LODF_{l,k} * f_k <= f_l_max   (for each contingency k, line l)
        f_l + LODF_{l,k} * f_k >= -f_l_max

    This creates a sequence of progressively larger LPs that Moreau solves efficiently.

References:
    - Wood & Wollenberg, "Power Generation, Operation and Control"
    - https://www.powerworld.com/WebHelp/Content/MainDocumentation_HTML/Line_Outage_Distribution_Factors_(LODFs).htm
"""

from __future__ import annotations

import argparse
import os
import re
import time
import traceback
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import moreau
import numpy as np
from scipy import sparse

# MATPOWER column indices
BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN = range(13)
GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN = range(10)
F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS = range(11)


@dataclass
class Violation:
    """A line limit violation (base case or contingency)."""

    line: int  # Line index
    contingency: Optional[int]  # None for base case, k for contingency k
    flow: float  # Actual flow (post-contingency if applicable)
    limit: float  # Line limit
    violation: float  # |flow| - limit (positive = violated)
    direction: int  # +1 for upper bound, -1 for lower bound


@dataclass
class Scenario:
    """Per-scenario data that varies across batch."""

    gen_cost: np.ndarray  # (n_gen,) - generation costs ($/MWh)
    gen_pmax: np.ndarray  # (n_gen,) - max generation (MW)
    load_p: np.ndarray  # (n_bus,) - demand at each bus (MW)
    # gen_pmin is always 0 to ensure feasibility


@dataclass
class BendersResult:
    """Result from Benders decomposition SCOPF."""

    x: np.ndarray  # Solution vectors (batch, n_vars)
    objectives: np.ndarray  # Optimal costs (batch,)
    status: str  # Solver status (Solved if all converged)
    iterations: int  # Benders iterations
    total_solve_time: float  # Total time across all solves
    cuts_added: int  # Number of unique cuts added
    active_constraints: List[Violation]  # Final active constraints (union)
    iteration_history: List[dict] = field(default_factory=list)
    batch_size: int = 1  # Number of scenarios

    @property
    def objective(self) -> float:
        """Return scalar objective for batch_size=1, average otherwise."""
        if self.batch_size == 1:
            return float(self.objectives[0])
        return float(np.mean(self.objectives))


@dataclass
class PowerNetwork:
    """Power network data container."""

    n_bus: int
    n_gen: int
    n_branch: int

    # Bus data
    bus_ids: np.ndarray  # Original bus IDs
    bus_type: np.ndarray  # 1=PQ, 2=PV, 3=slack
    load_p: np.ndarray  # Active power demand (MW)
    base_kv: np.ndarray  # Base voltage (kV)

    # Generator data
    gen_bus: np.ndarray  # Generator bus indices (0-indexed)
    gen_pmax: np.ndarray  # Max generation (MW)
    gen_pmin: np.ndarray  # Min generation (MW)
    gen_cost: np.ndarray  # Linear cost ($/MWh)

    # Branch data
    branch_from: np.ndarray  # From bus indices (0-indexed)
    branch_to: np.ndarray  # To bus indices (0-indexed)
    branch_x: np.ndarray  # Reactance (p.u.)
    branch_b: np.ndarray  # Susceptance = 1/x
    branch_rate: np.ndarray  # Thermal limit (MW)

    # Reference bus
    ref_bus: int  # Reference bus index (0-indexed)

    # Base MVA
    base_mva: float = 100.0


def load_matpower_case(url: str, thermal_limit_factor: float = 2.0) -> PowerNetwork:
    """Load a MATPOWER case from URL.

    Args:
        url: URL to MATPOWER .m file
        thermal_limit_factor: Factor to set thermal limits based on base case flows.

    Returns:
        PowerNetwork object
    """
    print(f"Fetching MATPOWER case from {url.split('/')[-1]}...")
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read().decode("utf-8")

    return _parse_matpower_content(content, thermal_limit_factor)


def load_matpower_file(filepath: str, thermal_limit_factor: float = 2.0) -> PowerNetwork:
    """Load a MATPOWER case from a local file.

    Args:
        filepath: Path to MATPOWER .m file
        thermal_limit_factor: Factor to set thermal limits based on base case flows.

    Returns:
        PowerNetwork object
    """
    print(f"Loading MATPOWER case from {os.path.basename(filepath)}...")
    with open(filepath, "r") as f:
        content = f.read()

    return _parse_matpower_content(content, thermal_limit_factor)


def _parse_matpower_content(content: str, thermal_limit_factor: float) -> PowerNetwork:
    """Parse MATPOWER file content into PowerNetwork."""

    def parse_matrix(name: str, min_cols: int) -> np.ndarray:
        """Parse MATPOWER matrix from file content."""
        pattern = rf"mpc\.{name}\s*=\s*\[(.*?)\];"
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find mpc.{name} in case file")

        lines = match.group(1).strip().split("\n")
        data = []
        for line in lines:
            line = line.split("%")[0].strip().rstrip(";")
            if line:
                vals = [float(x) for x in line.split()]
                if len(vals) >= min_cols:
                    data.append(vals[:min_cols])
        return np.array(data)

    # Parse data
    bus_data = parse_matrix("bus", 13)
    gen_data = parse_matrix("gen", 10)
    branch_data = parse_matrix("branch", 11)

    # Try to parse gencost (may not exist)
    try:
        gencost_data = parse_matrix("gencost", 7)
    except ValueError:
        gencost_data = None

    n_bus = len(bus_data)
    n_gen = len(gen_data)
    n_branch = len(branch_data)

    # Create bus ID mapping
    bus_ids = bus_data[:, BUS_I].astype(int)
    bus_id_to_idx = {bid: i for i, bid in enumerate(bus_ids)}

    # Find reference bus (type 3)
    ref_mask = bus_data[:, BUS_TYPE] == 3
    if not ref_mask.any():
        ref_bus = 0
    else:
        ref_bus = np.where(ref_mask)[0][0]

    # Map generator buses to indices
    gen_bus_ids = gen_data[:, GEN_BUS].astype(int)
    gen_bus = np.array([bus_id_to_idx[bid] for bid in gen_bus_ids])

    # Map branch buses to indices
    branch_from_ids = branch_data[:, F_BUS].astype(int)
    branch_to_ids = branch_data[:, T_BUS].astype(int)
    branch_from = np.array([bus_id_to_idx[bid] for bid in branch_from_ids])
    branch_to = np.array([bus_id_to_idx[bid] for bid in branch_to_ids])

    # Extract generator costs
    gen_cost = np.zeros(n_gen)
    if gencost_data is not None:
        for i in range(min(n_gen, len(gencost_data))):
            if gencost_data[i, 0] == 2:  # Polynomial
                n_coeffs = int(gencost_data[i, 3])
                if n_coeffs >= 2:
                    gen_cost[i] = gencost_data[i, 5]
                else:
                    gen_cost[i] = 20.0
            else:
                gen_cost[i] = 20.0
    else:
        gen_cost[:] = 20.0

    # Ensure non-zero costs
    gen_cost = np.maximum(gen_cost, 1.0)

    # Branch susceptance
    branch_x = branch_data[:, BR_X]
    branch_x = np.maximum(branch_x, 1e-6)  # Avoid division by zero
    branch_b = 1.0 / branch_x

    # Use provided thermal limits if available, otherwise compute
    branch_rate = branch_data[:, RATE_A].copy()

    # For lines with zero or missing limits, compute from base case flows
    zero_rate_mask = branch_rate <= 0

    if zero_rate_mask.any():
        # Compute base case flows
        load_p = bus_data[:, PD]

        B = np.zeros((n_bus, n_bus))
        for l in range(n_branch):
            i, j = branch_from[l], branch_to[l]
            b = branch_b[l]
            B[i, i] += b
            B[j, j] += b
            B[i, j] -= b
            B[j, i] -= b

        p_inj = -load_p.copy()
        total_load = load_p.sum()
        total_cap = gen_data[:, PMAX].sum()
        if total_cap > 0:
            for g in range(n_gen):
                bus = gen_bus[g]
                p_inj[bus] += gen_data[g, PMAX] * (total_load / total_cap)

        B_reduced = np.delete(np.delete(B, ref_bus, axis=0), ref_bus, axis=1)
        p_reduced = np.delete(p_inj, ref_bus) / 100.0

        try:
            theta_reduced = np.linalg.solve(B_reduced, p_reduced)
        except np.linalg.LinAlgError:
            theta_reduced = np.linalg.lstsq(B_reduced, p_reduced, rcond=None)[0]

        theta = np.zeros(n_bus)
        theta[:ref_bus] = theta_reduced[:ref_bus]
        theta[ref_bus + 1 :] = theta_reduced[ref_bus:]

        base_flows = np.zeros(n_branch)
        for l in range(n_branch):
            i, j = branch_from[l], branch_to[l]
            base_flows[l] = branch_b[l] * (theta[i] - theta[j]) * 100.0

        # Set limits for zero-rate lines
        branch_rate[zero_rate_mask] = np.maximum(
            thermal_limit_factor * np.abs(base_flows[zero_rate_mask]), 50.0
        )

    # Apply thermal limit factor to existing limits too
    if thermal_limit_factor > 1.0:
        nonzero_rate_mask = ~zero_rate_mask
        branch_rate[nonzero_rate_mask] *= thermal_limit_factor
    branch_rate = np.maximum(branch_rate, 50.0)

    return PowerNetwork(
        n_bus=n_bus,
        n_gen=n_gen,
        n_branch=n_branch,
        bus_ids=bus_ids,
        bus_type=bus_data[:, BUS_TYPE],
        load_p=bus_data[:, PD],
        base_kv=bus_data[:, BASE_KV],
        gen_bus=gen_bus,
        gen_pmax=gen_data[:, PMAX],
        gen_pmin=gen_data[:, PMIN],
        gen_cost=gen_cost,
        branch_from=branch_from,
        branch_to=branch_to,
        branch_x=branch_x,
        branch_b=branch_b,
        branch_rate=branch_rate,
        ref_bus=ref_bus,
    )


def load_ieee118(thermal_limit_factor: float = 1.5) -> PowerNetwork:
    """Load IEEE 118-bus test case from MATPOWER repository."""
    url = "https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case118.m"
    return load_matpower_case(url, thermal_limit_factor)


def load_activsg2000() -> PowerNetwork:
    """Load ACTIVSg2000 (2000-bus synthetic Texas grid) from MATPOWER repository."""
    url = "https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case_ACTIVSg2000.m"
    return load_matpower_case(url, thermal_limit_factor=0.15)


def load_texas7k(filepath: Optional[str] = None) -> PowerNetwork:
    """Load Texas7k (6717-bus Texas synthetic grid) from local file.

    The Texas7k case is not publicly available via URL and must be downloaded
    from Texas A&M: https://electricgrids.engr.tamu.edu/texas7k/

    Args:
        filepath: Path to Texas7k_*.m file. If None, looks in examples/power_systems/

    Returns:
        PowerNetwork object
    """
    if filepath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "Texas7k_20210804.m"),
            os.path.join(script_dir, "Texas7k_20220923.m"),
        ]
        for path in candidates:
            if os.path.exists(path):
                filepath = path
                break

        if filepath is None:
            raise FileNotFoundError(
                "Texas7k case file not found. Download from "
                "https://electricgrids.engr.tamu.edu/texas7k/ and place in "
                "examples/power_systems/"
            )

    return load_matpower_file(filepath, thermal_limit_factor=0.5)


def compute_ptdf(net: PowerNetwork, use_gpu: bool = True) -> np.ndarray:
    """Compute Power Transfer Distribution Factors (PTDF) matrix.

    PTDF[l, n] = sensitivity of flow on line l to injection at bus n

    For DC power flow:
        PTDF = B_f @ inv(B_bus)

    where B_f maps bus angles to line flows and B_bus is the bus susceptance matrix.

    Args:
        net: Power network
        use_gpu: If True, use PyTorch for GPU acceleration (falls back to NumPy if unavailable)

    Returns:
        PTDF matrix of shape (n_branch, n_bus)
    """
    use_torch = False
    if use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                use_torch = True
                device = torch.device("cuda")
                print("  Using GPU (PyTorch) for PTDF computation...")
            else:
                print("  CUDA not available, using CPU...")
        except ImportError:
            print("  PyTorch not available, using CPU...")

    n_bus = net.n_bus
    n_branch = net.n_branch
    ref = net.ref_bus

    if use_torch:
        import torch

        B = torch.zeros((n_bus, n_bus), dtype=torch.float64, device=device)
        B_f = torch.zeros((n_branch, n_bus), dtype=torch.float64, device=device)

        for l in range(n_branch):
            i, j = int(net.branch_from[l]), int(net.branch_to[l])
            b = float(net.branch_b[l])
            B[i, i] += b
            B[j, j] += b
            B[i, j] -= b
            B[j, i] -= b
            B_f[l, i] = b
            B_f[l, j] = -b

        mask = torch.ones(n_bus, dtype=torch.bool, device=device)
        mask[ref] = False
        B_reduced = B[mask][:, mask]
        B_f_reduced = B_f[:, mask]

        X = torch.linalg.inv(B_reduced)
        PTDF_reduced = B_f_reduced @ X

        PTDF = torch.zeros((n_branch, n_bus), dtype=torch.float64, device=device)
        PTDF[:, :ref] = PTDF_reduced[:, :ref]
        PTDF[:, ref + 1 :] = PTDF_reduced[:, ref:]

        return PTDF.cpu().numpy()
    else:
        B = np.zeros((n_bus, n_bus))
        for l in range(n_branch):
            i, j = net.branch_from[l], net.branch_to[l]
            b = net.branch_b[l]
            B[i, i] += b
            B[j, j] += b
            B[i, j] -= b
            B[j, i] -= b

        B_f = np.zeros((n_branch, n_bus))
        for l in range(n_branch):
            i, j = net.branch_from[l], net.branch_to[l]
            b = net.branch_b[l]
            B_f[l, i] = b
            B_f[l, j] = -b

        B_reduced = np.delete(np.delete(B, ref, axis=0), ref, axis=1)
        B_f_reduced = np.delete(B_f, ref, axis=1)

        try:
            X = np.linalg.inv(B_reduced)
        except np.linalg.LinAlgError:
            X = np.linalg.pinv(B_reduced)

        PTDF_reduced = B_f_reduced @ X

        PTDF = np.zeros((n_branch, n_bus))
        PTDF[:, :ref] = PTDF_reduced[:, :ref]
        PTDF[:, ref + 1 :] = PTDF_reduced[:, ref:]

        return PTDF


def compute_lodf(net: PowerNetwork, ptdf: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """Compute Line Outage Distribution Factors (LODF) matrix.

    LODF[l, k] = fraction of flow on line k that transfers to line l when k is outaged

    Post-contingency flow: f_l^post = f_l^pre + LODF[l,k] * f_k^pre

    Formula: LODF[l,k] = PTDF[l, from_k] - PTDF[l, to_k]
                         / (1 - (PTDF[k, from_k] - PTDF[k, to_k]))

    Args:
        net: Power network
        ptdf: PTDF matrix
        use_gpu: If True, use PyTorch for GPU acceleration

    Returns:
        LODF matrix of shape (n_branch, n_branch)
        Diagonal elements are NaN (line can't redistribute to itself)
    """
    use_torch = False
    if use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                use_torch = True
                device = torch.device("cuda")
                print("  Using GPU (PyTorch) for LODF computation...")
        except ImportError:
            pass

    n_branch = net.n_branch

    if use_torch:
        import torch

        ptdf_t = torch.tensor(ptdf, dtype=torch.float64, device=device)
        branch_from_t = torch.tensor(net.branch_from, dtype=torch.long, device=device)
        branch_to_t = torch.tensor(net.branch_to, dtype=torch.long, device=device)

        ptdf_from = ptdf_t[:, branch_from_t]
        ptdf_to = ptdf_t[:, branch_to_t]
        ptdf_diff = ptdf_from - ptdf_to

        diag = torch.diag(ptdf_diff)
        denom = 1.0 - diag

        radial_mask = torch.abs(denom) < 1e-10
        safe_denom = torch.where(radial_mask, torch.ones_like(denom), denom)
        lodf = ptdf_diff / safe_denom.unsqueeze(0)
        lodf[:, radial_mask] = float("inf")

        lodf = lodf.cpu().numpy()
    else:
        ptdf_from = ptdf[:, net.branch_from]
        ptdf_to = ptdf[:, net.branch_to]
        ptdf_diff = ptdf_from - ptdf_to

        diag = np.diag(ptdf_diff)
        denom = 1.0 - diag

        radial_mask = np.abs(denom) < 1e-10
        safe_denom = np.where(radial_mask, 1.0, denom)
        lodf = ptdf_diff / safe_denom[np.newaxis, :]
        lodf[:, radial_mask] = np.inf

    np.fill_diagonal(lodf, np.nan)
    return lodf


def select_contingencies(
    net: PowerNetwork,
    lodf: np.ndarray,
    max_contingencies: int = 50,
    min_voltage_kv: float = 100.0,
    max_lodf: float = 0.8,
) -> List[int]:
    """Select critical contingencies for monitoring.

    Selection criteria:
    1. High-voltage lines (>= min_voltage_kv)
    2. Lines with moderate LODF impact (avoid extreme redistributions)
    3. Exclude radial lines (would cause islanding)
    4. Exclude lines with LODF close to +/-1 (cause infeasibility)

    Args:
        net: Power network
        lodf: LODF matrix
        max_contingencies: Maximum number of contingencies to monitor
        min_voltage_kv: Minimum voltage level for contingency monitoring
        max_lodf: Maximum absolute LODF value to allow (avoid near-islanding)

    Returns:
        List of branch indices to monitor as contingencies
    """
    candidates = []

    for k in range(net.n_branch):
        if np.any(np.isinf(lodf[:, k])):
            continue

        from_bus, to_bus = net.branch_from[k], net.branch_to[k]
        voltage = max(net.base_kv[from_bus], net.base_kv[to_bus])
        if voltage < min_voltage_kv:
            continue

        lodf_k = lodf[:, k].copy()
        lodf_k[k] = 0
        valid_lodf = lodf_k[~np.isnan(lodf_k) & ~np.isinf(lodf_k)]
        if len(valid_lodf) == 0:
            continue

        max_abs_lodf = np.max(np.abs(valid_lodf))

        if max_abs_lodf > max_lodf:
            continue

        candidates.append((k, max_abs_lodf))

    candidates.sort(key=lambda x: -x[1])
    selected = [k for k, _ in candidates[:max_contingencies]]

    return selected


def check_all_violations(
    f: np.ndarray,
    lodf: np.ndarray,
    branch_rate: np.ndarray,
    contingencies: List[int],
    tol: float = 1.0,
) -> List[Violation]:
    """Check all line limits (base case + contingencies).

    Args:
        f: Current line flows (MW)
        lodf: LODF matrix
        branch_rate: Thermal limits (MW)
        contingencies: List of contingency indices to check
        tol: Violation tolerance (MW) - ignore violations below this

    Returns:
        List of Violation objects, sorted by magnitude (worst first).
    """
    violations = []
    n_branch = len(f)

    # Base case violations
    for l in range(n_branch):
        excess = abs(f[l]) - branch_rate[l]
        if excess > tol:
            violations.append(
                Violation(
                    line=l,
                    contingency=None,
                    flow=f[l],
                    limit=branch_rate[l],
                    violation=excess,
                    direction=1 if f[l] > 0 else -1,
                )
            )

    # Contingency violations
    for k in contingencies:
        f_k = f[k]
        lodf_col = lodf[:, k]

        for l in range(n_branch):
            if l == k:
                continue
            lodf_lk = lodf_col[l]
            if np.isnan(lodf_lk) or np.isinf(lodf_lk):
                continue

            f_post = f[l] + lodf_lk * f_k
            excess = abs(f_post) - branch_rate[l]

            if excess > tol:
                violations.append(
                    Violation(
                        line=l,
                        contingency=k,
                        flow=f_post,
                        limit=branch_rate[l],
                        violation=excess,
                        direction=1 if f_post > 0 else -1,
                    )
                )

    violations.sort(key=lambda v: -v.violation)
    return violations


def deduplicate_violations(violations: List[Violation]) -> List[Violation]:
    """Deduplicate violations by (line, contingency, direction).

    For batch Benders, multiple scenarios may produce the same violation.
    We only need one cut per unique (line, contingency, direction) tuple.

    Args:
        violations: List of violations (may contain duplicates)

    Returns:
        List of unique violations (worst magnitude kept for each key)
    """
    seen: dict = {}
    for v in violations:
        key = (v.line, v.contingency, v.direction)
        if key not in seen or v.violation > seen[key].violation:
            seen[key] = v
    return list(seen.values())


def generate_scenarios(
    net: PowerNetwork,
    batch_size: int,
    cost_range: Tuple[float, float] = (0.8, 1.2),
    pmax_range: Tuple[float, float] = (0.9, 1.1),
    load_range: Tuple[float, float] = (0.8, 1.2),
    seed: Optional[int] = None,
) -> List[Scenario]:
    """Generate batch of scenarios by scaling base case.

    Each scenario has randomly scaled gen_cost, gen_pmax, and load_p.
    Feasibility is ensured by scaling pmax up if needed to meet demand.

    Args:
        net: Base power network
        batch_size: Number of scenarios to generate
        cost_range: (min, max) scale factors for generation costs
        pmax_range: (min, max) scale factors for max generation
        load_range: (min, max) scale factors for load
        seed: Random seed for reproducibility

    Returns:
        List of Scenario objects
    """
    rng = np.random.default_rng(seed)

    scenarios = []
    for _ in range(batch_size):
        cost_factors = rng.uniform(cost_range[0], cost_range[1], net.n_gen)
        gen_cost = net.gen_cost * cost_factors

        pmax_factors = rng.uniform(pmax_range[0], pmax_range[1], net.n_gen)
        gen_pmax = net.gen_pmax * pmax_factors

        load_factors = rng.uniform(load_range[0], load_range[1], net.n_bus)
        load_p = net.load_p * load_factors

        total_capacity = gen_pmax.sum()
        total_demand = load_p.sum()
        if total_capacity < total_demand:
            scale = (total_demand * 1.1) / total_capacity
            gen_pmax = gen_pmax * scale

        scenarios.append(
            Scenario(
                gen_cost=gen_cost,
                gen_pmax=gen_pmax,
                load_p=load_p,
            )
        )

    return scenarios


def base_scenario(net: PowerNetwork) -> Scenario:
    """Create a single scenario from the network's base case data."""
    return Scenario(
        gen_cost=net.gen_cost.copy(),
        gen_pmax=net.gen_pmax.copy(),
        load_p=net.load_p.copy(),
    )


def build_opf_with_cuts(
    net: PowerNetwork,
    scenarios: List[Scenario],
    cuts: List[Violation],
    lodf: np.ndarray,
    load_shed_cost: float = 5000.0,
) -> dict:
    """Build batched OPF problem for CompiledSolver.

    The constraint matrix A structure is shared across all scenarios.
    Only q (costs), b (demand RHS), l (lower bounds), u (upper bounds) vary.

    Variables: x = [p_g, theta, f, load_shed]

    Args:
        net: Power network (topology only - scenario data overrides gen/load)
        scenarios: List of Scenario with per-scenario costs/limits/demand
        cuts: List of Violation objects representing constraints to add
        lodf: LODF matrix
        load_shed_cost: Cost of load shedding ($/MWh), default 5000

    Returns:
        Dict with CSR structure and batched q, b, l, u arrays
    """
    batch_size = len(scenarios)
    n_bus = net.n_bus
    n_gen = net.n_gen
    n_branch = net.n_branch

    # Variable layout: [p_g, theta, f, load_shed]
    n_vars = n_gen + n_bus + n_branch + n_bus
    idx_pg = slice(0, n_gen)
    idx_theta = slice(n_gen, n_gen + n_bus)
    idx_f = slice(n_gen + n_bus, n_gen + n_bus + n_branch)
    idx_load_shed = slice(n_gen + n_bus + n_branch, n_vars)

    # Constraints
    n_balance = n_bus
    n_flow = n_branch
    n_ref = 1
    n_eq = n_balance + n_flow + n_ref
    n_ineq = len(cuts)
    n_cons = n_eq + n_ineq

    # Build constraint matrix A (shared structure)
    rows = []
    cols = []
    data = []

    # 1. Nodal power balance: C_g @ p_g + load_shed - K @ f = d
    for g in range(n_gen):
        bus = net.gen_bus[g]
        rows.append(bus)
        cols.append(g)
        data.append(1.0)

    # Load shedding
    for bus in range(n_bus):
        rows.append(bus)
        cols.append(n_gen + n_bus + n_branch + bus)
        data.append(1.0)

    for l in range(n_branch):
        i, j = net.branch_from[l], net.branch_to[l]
        rows.append(i)
        cols.append(n_gen + n_bus + l)
        data.append(-1.0)
        rows.append(j)
        cols.append(n_gen + n_bus + l)
        data.append(1.0)

    # 2. DC flow equations
    row_offset = n_bus
    for l in range(n_branch):
        row = row_offset + l
        i, j = net.branch_from[l], net.branch_to[l]
        bl = net.branch_b[l]

        rows.append(row)
        cols.append(n_gen + n_bus + l)
        data.append(1.0)

        rows.append(row)
        cols.append(n_gen + i)
        data.append(-bl)

        rows.append(row)
        cols.append(n_gen + j)
        data.append(bl)

    # 3. Reference bus angle
    row_offset = n_bus + n_branch
    rows.append(row_offset)
    cols.append(n_gen + net.ref_bus)
    data.append(1.0)

    # 4. Line limit cuts (inequalities) - shared structure
    row_offset = n_eq
    for cut_idx, cut in enumerate(cuts):
        row = row_offset + cut_idx
        l = cut.line
        k = cut.contingency
        direction = cut.direction

        rows.append(row)
        cols.append(n_gen + n_bus + l)
        data.append(direction)

        if k is not None:
            lodf_lk = lodf[l, k]
            if not np.isnan(lodf_lk) and not np.isinf(lodf_lk):
                rows.append(row)
                cols.append(n_gen + n_bus + k)
                data.append(direction * lodf_lk)

    # Build CSR matrix to get structure
    A = sparse.csr_array((data, (rows, cols)), shape=(n_cons, n_vars))

    # Extract CSR structure
    A_row_offsets = A.indptr.tolist()
    A_col_indices = A.indices.tolist()
    A_values = A.data.astype(np.float64)

    # P = 0 (linear program)
    P = sparse.csr_array((n_vars, n_vars), dtype=np.float64)
    P_row_offsets = P.indptr.tolist()
    P_col_indices = P.indices.tolist()
    P_values = P.data.astype(np.float64)

    # Build batched q, b, l, u arrays
    qs = np.zeros((batch_size, n_vars), dtype=np.float64)
    bs = np.zeros((batch_size, n_cons), dtype=np.float64)
    ls = np.zeros((batch_size, n_vars), dtype=np.float64)
    us = np.zeros((batch_size, n_vars), dtype=np.float64)

    for i, scenario in enumerate(scenarios):
        qs[i, idx_pg] = scenario.gen_cost
        qs[i, idx_load_shed] = load_shed_cost

        bs[i, :n_bus] = scenario.load_p
        for cut_idx, cut in enumerate(cuts):
            bs[i, n_eq + cut_idx] = cut.limit

        ls[i, idx_pg] = 0.0
        ls[i, idx_theta] = -np.inf
        ls[i, idx_f] = -np.inf
        ls[i, idx_load_shed] = 0.0

        us[i, idx_pg] = scenario.gen_pmax
        us[i, idx_theta] = np.inf
        us[i, idx_f] = np.inf
        us[i, idx_load_shed] = np.maximum(scenario.load_p, 0.0)

    return {
        "n": n_vars,
        "m": n_cons,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "qs": qs,
        "bs": bs,
        "ls": ls,
        "us": us,
        "n_eq": n_eq,
        "n_ineq": n_ineq,
        "n_gen": n_gen,
        "n_bus": n_bus,
        "n_branch": n_branch,
        "idx_pg": idx_pg,
        "idx_theta": idx_theta,
        "idx_f": idx_f,
        "idx_load_shed": idx_load_shed,
        "network": net,
        "scenarios": scenarios,
        "n_cuts": len(cuts),
        "batch_size": batch_size,
        "load_shed_cost": load_shed_cost,
    }


def solve_with_moreau(
    problem: dict,
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """Solve batched OPF problem using Moreau CompiledSolver.

    Args:
        problem: Problem dict from build_opf_with_cuts
        device: Device for solver ('cpu' or 'cuda')
        verbose: Print solver output

    Returns:
        Dict with batched results: x (batch, n_vars), objectives (batch,), etc.
    """
    n_eq = problem["n_eq"]
    n_ineq = problem["n_ineq"]
    batch_size = problem["batch_size"]

    cones = moreau.Cones(
        num_zero_cones=n_eq,
        num_nonneg_cones=n_ineq,
    )

    settings = moreau.Settings(verbose=verbose, device=device)

    t0 = time.perf_counter()
    solver = moreau.CompiledSolver(
        n=problem["n"],
        m=problem["m"],
        P_row_offsets=problem["P_row_offsets"],
        P_col_indices=problem["P_col_indices"],
        A_row_offsets=problem["A_row_offsets"],
        A_col_indices=problem["A_col_indices"],
        cones=cones,
        settings=settings,
        batch_size=batch_size,
    )

    solver.setup(problem["P_values"], problem["A_values"])
    construction_time = time.perf_counter() - t0

    t_solve = time.perf_counter()
    result = solver.solve(problem["qs"], problem["bs"], problem["ls"], problem["us"])
    info = solver.info
    solve_time = time.perf_counter() - t_solve

    all_solved = all(s.name in ["Solved", "AlmostSolved"] for s in info.status)

    return {
        "x": result.x,
        "objectives": info.obj_val,
        "status": "Solved" if all_solved else "PartialSolve",
        "solve_time": solve_time,
        "iterations": info.iterations,
        "construction_time": construction_time,
        "statuses": [s.name for s in info.status],
    }


def solve_with_mosek(
    problem: dict,
    verbose: bool = False,
    num_threads: int = 2,
) -> dict:
    """Solve batched OPF problem using Mosek optimizebatch().

    Args:
        problem: Problem dict from build_opf_with_cuts
        verbose: Print solver output
        num_threads: Threads per task for IPM

    Returns:
        Dict with batched results: x (batch, n_vars), objectives (batch,), etc.
    """
    try:
        import mosek
    except ImportError:
        raise ImportError("Mosek not installed. Install with: pip install mosek")

    n_vars = problem["n"]
    n_cons = problem["m"]
    n_eq = problem["n_eq"]
    batch_size = problem["batch_size"]

    A = sparse.csr_array(
        (problem["A_values"], problem["A_col_indices"], problem["A_row_offsets"]),
        shape=(n_cons, n_vars),
    )

    t0 = time.perf_counter()

    with mosek.Env() as env:
        tasks = []

        for i in range(batch_size):
            task = mosek.Task(env, 0, 0)

            if verbose and i == 0:
                task.set_Stream(mosek.streamtype.log, lambda msg: print(f"    {msg}", end=""))

            task.appendvars(n_vars)

            task.putobjsense(mosek.objsense.minimize)
            q = problem["qs"][i]
            for j in range(n_vars):
                task.putcj(j, q[j])

            l_bounds = problem["ls"][i]
            u_bounds = problem["us"][i]
            for j in range(n_vars):
                lo, up = l_bounds[j], u_bounds[j]
                if np.isinf(lo) and np.isinf(up):
                    task.putvarbound(j, mosek.boundkey.fr, 0.0, 0.0)
                elif np.isinf(lo):
                    task.putvarbound(j, mosek.boundkey.up, 0.0, up)
                elif np.isinf(up):
                    task.putvarbound(j, mosek.boundkey.lo, lo, 0.0)
                elif lo == up:
                    task.putvarbound(j, mosek.boundkey.fx, lo, up)
                else:
                    task.putvarbound(j, mosek.boundkey.ra, lo, up)

            task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
            task.putintparam(mosek.iparam.num_threads, num_threads)

            task.appendcons(n_cons)

            A_csr = A.tocsr()
            for row in range(n_cons):
                row_start = A_csr.indptr[row]
                row_end = A_csr.indptr[row + 1]
                cols = A_csr.indices[row_start:row_end].tolist()
                vals = A_csr.data[row_start:row_end].tolist()
                task.putarow(row, cols, vals)

            b = problem["bs"][i]
            for row in range(n_eq):
                task.putconbound(row, mosek.boundkey.fx, b[row], b[row])
            for row in range(n_eq, n_cons):
                task.putconbound(row, mosek.boundkey.up, -np.inf, b[row])

            tasks.append(task)

        construction_time = time.perf_counter() - t0

        threadpoolsize = os.cpu_count() or 4

        t_solve = time.perf_counter()
        trm, res = env.optimizebatch(False, -1.0, threadpoolsize, tasks)
        solve_time = time.perf_counter() - t_solve

        x_batch = np.zeros((batch_size, n_vars))
        objectives = np.zeros(batch_size)
        statuses = []
        total_iterations = 0

        for i, task in enumerate(tasks):
            # Check termination code first
            if verbose and trm[i] != mosek.rescode.ok:
                print(f"  Task {i}: termination code = {trm[i]}")

            solsta = task.getsolsta(mosek.soltype.itr)
            prosta = task.getprosta(mosek.soltype.itr)

            # Check for optimal or feasible solutions
            # Mosek solsta values: optimal, prim_feas, dual_feas, prim_and_dual_feas, unknown
            if solsta in (mosek.solsta.optimal, mosek.solsta.prim_and_dual_feas):
                task.getxx(mosek.soltype.itr, x_batch[i])
                objectives[i] = task.getprimalobj(mosek.soltype.itr)
                statuses.append("Solved")
            elif solsta in (mosek.solsta.prim_feas, mosek.solsta.dual_feas):
                # Primal or dual feasible only - still get solution
                task.getxx(mosek.soltype.itr, x_batch[i])
                objectives[i] = task.getprimalobj(mosek.soltype.itr)
                statuses.append("AlmostSolved")
            elif solsta == mosek.solsta.unknown:
                # For stalled or unknown cases, still try to get any available solution
                # Mosek may have an interior-point iterate even if not converged
                try:
                    task.getxx(mosek.soltype.itr, x_batch[i])
                    objectives[i] = task.getprimalobj(mosek.soltype.itr)
                    # If we got here without error, we have a solution
                    statuses.append("AlmostSolved")
                except mosek.MosekException:
                    objectives[i] = np.nan
                    statuses.append(f"Unknown({solsta},{prosta})")
                    if verbose:
                        print(f"  Task {i}: solsta={solsta}, prosta={prosta}")
            else:
                objectives[i] = np.nan
                statuses.append(f"Unknown({solsta},{prosta})")
                if verbose:
                    print(f"  Task {i}: solsta={solsta}, prosta={prosta}")

            total_iterations += task.getintinf(mosek.iinfitem.intpnt_iter)

    all_solved = all(s in ("Solved", "AlmostSolved") for s in statuses)

    return {
        "x": x_batch,
        "objectives": objectives,
        "status": "Solved" if all_solved else "PartialSolve",
        "solve_time": solve_time,
        "iterations": total_iterations // batch_size,
        "construction_time": construction_time,
        "statuses": statuses,
    }


class BendersSCOPF:
    """Benders decomposition solver for SCOPF.

    Solves SCOPF by iteratively:
    1. Solve master problem (economic dispatch + accumulated cuts)
    2. Check for line limit violations (base case + contingencies)
    3. Add all violations as cuts
    4. Repeat until no violations or max iterations

    Iteration 0 has NO line limits - pure economic dispatch.
    Includes load shedding at high cost to ensure feasibility.

    Supports batched solving: multiple scenarios with shared cut set.
    Use batch_size=1 for single problem (default).
    """

    def __init__(
        self,
        net: PowerNetwork,
        contingencies: List[int],
        lodf: np.ndarray,
        scenarios: Optional[List[Scenario]] = None,
        solver: str = "moreau",
        device: str = "cpu",
        verbose: bool = True,
        max_iterations: int = 5,
        violation_tol: float = 1.0,
        load_shed_cost: float = 5000.0,
    ):
        """Initialize Benders SCOPF solver.

        Args:
            net: Power network data
            contingencies: List of contingency (line outage) indices
            lodf: LODF matrix
            scenarios: List of Scenario objects. If None, uses base case (batch_size=1).
            solver: Solver backend ('moreau' or 'mosek')
            device: Solver device for moreau ('cpu' or 'cuda')
            verbose: Print iteration progress
            max_iterations: Maximum Benders iterations (default 5)
            violation_tol: Ignore violations below this (MW)
            load_shed_cost: Cost of load shedding ($/MWh), default 5000
        """
        self.net = net
        self.contingencies = contingencies
        self.lodf = lodf
        self.solver = solver
        self.device = device
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.violation_tol = violation_tol
        self.load_shed_cost = load_shed_cost

        # Default to single scenario from base case
        if scenarios is None:
            self.scenarios = [base_scenario(net)]
        else:
            self.scenarios = scenarios
        self.batch_size = len(self.scenarios)

    def solve(self) -> BendersResult:
        """Run Benders decomposition.

        Returns:
            BendersResult with solutions for all scenarios
        """
        cuts: List[Violation] = []
        iteration_history = []
        total_solve_time = 0.0

        n_gen = self.net.n_gen
        n_bus = self.net.n_bus
        n_branch = self.net.n_branch
        n_vars = n_gen + n_bus + n_branch + n_bus
        idx_f = slice(n_gen + n_bus, n_gen + n_bus + n_branch)
        idx_load_shed = slice(n_gen + n_bus + n_branch, n_vars)

        result = None

        for iteration in range(self.max_iterations):
            # Build problem with current cuts
            problem = build_opf_with_cuts(
                self.net, self.scenarios, cuts, self.lodf, self.load_shed_cost
            )

            # Solve
            if self.solver == "moreau":
                result = solve_with_moreau(problem, self.device, verbose=False)
            elif self.solver == "mosek":
                result = solve_with_mosek(problem, verbose=False)
            else:
                raise ValueError(f"Unknown solver: {self.solver}. Choose from: moreau, mosek")

            total_solve_time += result["solve_time"]

            if result["status"] not in ["Solved", "AlmostSolved"]:
                if self.verbose:
                    print(f"Iter {iteration}: solver returned {result['status']}")
                return BendersResult(
                    x=np.asarray(result["x"]),
                    objectives=np.asarray(result["objectives"]),
                    status=result["status"],
                    iterations=iteration + 1,
                    total_solve_time=total_solve_time,
                    cuts_added=len(cuts),
                    active_constraints=cuts,
                    iteration_history=iteration_history,
                    batch_size=self.batch_size,
                )

            # Check violations for ALL scenarios, collect union
            all_violations: List[Violation] = []
            total_load_shed = 0.0

            for i in range(self.batch_size):
                f = result["x"][i, idx_f]
                load_shed = result["x"][i, idx_load_shed]
                total_load_shed += np.sum(load_shed)

                violations = check_all_violations(
                    f, self.lodf, self.net.branch_rate, self.contingencies, self.violation_tol
                )
                all_violations.extend(violations)

            # Deduplicate cuts across scenarios
            new_cuts = deduplicate_violations(all_violations)

            n_base = sum(1 for v in new_cuts if v.contingency is None)
            n_cont = len(new_cuts) - n_base

            # Log iteration
            objectives_arr = np.asarray(result["objectives"])
            avg_objective = np.mean(objectives_arr)
            iter_info = {
                "iteration": iteration,
                "avg_objective": avg_objective,
                "objectives": objectives_arr.tolist(),
                "n_violations": len(new_cuts),
                "n_base_violations": n_base,
                "n_cont_violations": n_cont,
                "n_cuts": len(cuts),
                "worst_violation_mw": new_cuts[0].violation if new_cuts else 0,
                "solve_time": result["solve_time"],
                "total_load_shed": total_load_shed,
            }
            iteration_history.append(iter_info)

            if self.verbose:
                worst = new_cuts[0].violation if new_cuts else 0
                shed_str = f", shed={total_load_shed:.1f} MW" if total_load_shed > 0.1 else ""
                if self.batch_size == 1:
                    print(
                        f"Iter {iteration}: obj=${avg_objective:,.2f}, "
                        f"violations={len(new_cuts)} (base={n_base}, cont={n_cont}), "
                        f"cuts={len(cuts)}, worst={worst:.1f} MW{shed_str}"
                    )
                else:
                    print(
                        f"Iter {iteration}: avg_obj=${avg_objective:,.2f}, "
                        f"violations={len(new_cuts)} (base={n_base}, cont={n_cont}), "
                        f"cuts={len(cuts)}, worst={worst:.1f} MW{shed_str}"
                    )

            if len(new_cuts) == 0:
                if self.verbose:
                    print(f"Converged in {iteration + 1} iterations with {len(cuts)} cuts")
                return BendersResult(
                    x=np.asarray(result["x"]),
                    objectives=np.asarray(result["objectives"]),
                    status="Solved",
                    iterations=iteration + 1,
                    total_solve_time=total_solve_time,
                    cuts_added=len(cuts),
                    active_constraints=cuts,
                    iteration_history=iteration_history,
                    batch_size=self.batch_size,
                )

            cuts.extend(new_cuts)

        if self.verbose:
            print(
                f"Max iterations ({self.max_iterations}) reached with {len(new_cuts)} remaining violations"
            )
        return BendersResult(
            x=np.asarray(result["x"]),
            objectives=np.asarray(result["objectives"]),
            status="MaxIterations",
            iterations=self.max_iterations,
            total_solve_time=total_solve_time,
            cuts_added=len(cuts),
            active_constraints=cuts,
            iteration_history=iteration_history,
            batch_size=self.batch_size,
        )


def run_example(
    n_contingencies: int = 50,
    solver: str = "moreau",
    device: str = "cpu",
    thermal_limit_factor: float = 2.5,
) -> Tuple[BendersSCOPF, BendersResult]:
    """Run SCOPF example on IEEE 118-bus system using Benders decomposition.

    Args:
        n_contingencies: Number of contingencies to monitor
        solver: Solver backend ('moreau' or 'mosek')
        device: Solver device for moreau ('cpu' or 'cuda')
        thermal_limit_factor: Factor for setting thermal limits

    Returns:
        Tuple of (solver_obj, result)
    """
    print("=" * 70)
    print("Security-Constrained DC Optimal Power Flow (SCOPF)")
    print(f"IEEE 118-Bus Test System - Benders Decomposition ({solver})")
    print("=" * 70)

    net = load_ieee118(thermal_limit_factor=thermal_limit_factor)

    print(f"\nNetwork Statistics:")
    print(f"  Buses:      {net.n_bus}")
    print(f"  Generators: {net.n_gen}")
    print(f"  Branches:   {net.n_branch}")
    print(f"  Total load: {net.load_p.sum():.1f} MW")
    print(f"  Total gen capacity: {net.gen_pmax.sum():.1f} MW")

    print("\n--- Computing Sensitivity Factors ---")
    t0 = time.perf_counter()
    ptdf = compute_ptdf(net)
    lodf = compute_lodf(net, ptdf)
    sens_time = time.perf_counter() - t0
    print(f"  PTDF shape: {ptdf.shape}")
    print(f"  LODF shape: {lodf.shape}")
    print(f"  Computation time: {sens_time:.4f}s")

    print("\n--- Selecting Contingencies ---")
    contingencies = select_contingencies(net, lodf, max_contingencies=n_contingencies)
    print(f"  Selected {len(contingencies)} contingencies")
    print(f"  Potential constraint pairs: {len(contingencies) * (net.n_branch - 1):,}")

    print(f"\n--- Benders Decomposition (solver={solver}) ---")
    solver_obj = BendersSCOPF(
        net=net,
        contingencies=contingencies,
        lodf=lodf,
        solver=solver,
        device=device,
        verbose=True,
        max_iterations=5,
        violation_tol=1.0,
    )

    t0 = time.perf_counter()
    result = solver_obj.solve()
    total_time = time.perf_counter() - t0

    print(f"\n--- Summary ---")
    print(f"  Solver: {solver}")
    print(f"  Status: {result.status}")
    print(f"  Objective: ${result.objective:,.2f}")
    print(f"  Benders iterations: {result.iterations}")
    print(f"  Total cuts added: {result.cuts_added}")
    print(f"  Total solve time: {result.total_solve_time:.4f}s")
    print(f"  Total wall time: {total_time:.4f}s")

    n_base_cuts = sum(1 for c in result.active_constraints if c.contingency is None)
    n_cont_cuts = result.cuts_added - n_base_cuts
    print(f"  Base case cuts: {n_base_cuts}")
    print(f"  Contingency cuts: {n_cont_cuts}")

    print("\n" + "=" * 70)

    return solver_obj, result


def run_batch_comparison(
    network_loader: callable,
    network_name: str,
    batch_sizes: List[int] = [1, 4, 16],
    n_contingencies: int = 50,
    solvers: List[str] = ["moreau", "mosek"],
    device: str = "cpu",
    max_iterations: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> List[dict]:
    """Compare batch SCOPF performance across solvers.

    Args:
        network_loader: Function to load the network (e.g., load_ieee118)
        network_name: Name for display
        batch_sizes: List of batch sizes to test
        n_contingencies: Number of contingencies to monitor
        solvers: List of solvers to compare ('moreau', 'mosek')
        device: Solver device for Moreau ('cpu' or 'cuda')
        max_iterations: Maximum Benders iterations
        seed: Random seed for scenario generation
        verbose: Print detailed output

    Returns:
        List of result dicts
    """
    print("=" * 80)
    solver_labels = []
    for s in solvers:
        if s == "moreau":
            solver_labels.append(f"Moreau ({device})")
        else:
            solver_labels.append(s.capitalize())
    print(f"Batch SCOPF Solver Comparison: {' vs '.join(solver_labels)}")
    print(f"Network: {network_name}")
    print(f"Method: Benders Decomposition")
    print("=" * 80)

    net = network_loader()

    print(f"\nNetwork Statistics:")
    print(f"  Buses:      {net.n_bus}")
    print(f"  Generators: {net.n_gen}")
    print(f"  Branches:   {net.n_branch}")
    print(f"  Total load: {net.load_p.sum():.1f} MW")
    print(f"  Total gen capacity: {net.gen_pmax.sum():.1f} MW")

    print("\n--- Computing Sensitivity Factors ---")
    t0 = time.perf_counter()
    ptdf = compute_ptdf(net, use_gpu=(device == "cuda"))
    lodf = compute_lodf(net, ptdf, use_gpu=(device == "cuda"))
    sens_time = time.perf_counter() - t0
    print(f"  PTDF shape: {ptdf.shape}")
    print(f"  LODF shape: {lodf.shape}")
    print(f"  Computation time: {sens_time:.2f}s")

    contingencies = select_contingencies(net, lodf, max_contingencies=n_contingencies)
    print(f"\n  Selected {len(contingencies)} contingencies")

    results = []

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")

        if batch_size == 1:
            scenarios = [base_scenario(net)]
        else:
            scenarios = generate_scenarios(net, batch_size, seed=seed)

        for solver in solvers:
            try:
                benders = BendersSCOPF(
                    net=net,
                    contingencies=contingencies,
                    lodf=lodf,
                    scenarios=scenarios,
                    solver=solver,
                    device=device if solver == "moreau" else "cpu",
                    verbose=False,
                    max_iterations=max_iterations,
                    violation_tol=1.0,
                )

                t0 = time.perf_counter()
                result = benders.solve()
                wall_time = time.perf_counter() - t0

                avg_objective = np.mean(result.objectives)
                res = {
                    "solver": solver,
                    "device": device if solver == "moreau" else "cpu",
                    "batch_size": batch_size,
                    "n_contingencies": len(contingencies),
                    "status": result.status,
                    "avg_objective": avg_objective,
                    "objectives": result.objectives.tolist(),
                    "iterations": result.iterations,
                    "cuts_added": result.cuts_added,
                    "solve_time": result.total_solve_time,
                    "wall_time": wall_time,
                    "time_per_problem": result.total_solve_time / batch_size,
                }
                results.append(res)

                if verbose:
                    solver_label = f"{solver}({device})" if solver == "moreau" else solver
                    print(
                        f"  {solver_label:14s}: avg_obj=${avg_objective:,.0f}, "
                        f"solve={res['solve_time']:.3f}s, "
                        f"per_prob={res['time_per_problem']:.4f}s, "
                        f"iters={res['iterations']}, cuts={res['cuts_added']}"
                    )

            except ImportError as e:
                print(f"  {solver:8s}: not available - {e}")
                results.append(
                    {
                        "solver": solver,
                        "batch_size": batch_size,
                        "status": "UNAVAILABLE",
                        "error": str(e),
                    }
                )
            except Exception as e:
                print(f"  {solver:8s}: FAILED - {e}")
                traceback.print_exc()
                results.append(
                    {
                        "solver": solver,
                        "batch_size": batch_size,
                        "status": "FAILED",
                        "error": str(e),
                    }
                )

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY - Batch SCOPF Solve Time Comparison")
    print("=" * 80)

    by_batch = defaultdict(dict)
    for r in results:
        if "error" not in r:
            by_batch[r["batch_size"]][r["solver"]] = r

    header = f"{'Batch Size':>12}"
    for solver in solvers:
        if solver == "moreau":
            header += f" {f'Moreau({device})':>14}"
        else:
            header += f" {solver.capitalize():>14}"
    header += f" {'Per-Problem':>12}"
    header += f" {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    for batch_size in sorted(by_batch.keys()):
        row_data = by_batch[batch_size]
        row = f"{batch_size:>12}"

        for solver in solvers:
            if solver in row_data:
                row += f" {row_data[solver]['solve_time']:>12.3f}s"
            else:
                row += f" {'N/A':>13}"

        if "moreau" in row_data:
            row += f" {row_data['moreau']['time_per_problem']:>10.4f}s"
        else:
            row += f" {'N/A':>11}"

        if "moreau" in row_data and "mosek" in row_data:
            moreau_t = row_data["moreau"]["solve_time"]
            mosek_t = row_data["mosek"]["solve_time"]
            if moreau_t > 0:
                speedup = mosek_t / moreau_t
                row += f" {speedup:>8.2f}x"
            else:
                row += f" {'N/A':>9}"
        else:
            row += f" {'N/A':>9}"

        print(row)

    print("=" * 80)

    return results


def run_network(
    network_loader: Callable[[], PowerNetwork],
    network_name: str,
    solver: str = "moreau",
    device: str = "cpu",
    batch_size: int = 1,
    n_contingencies: int = 50,
    seed: int = 42,
    max_iterations: int = 5,
    violation_tol: float = 1.0,
    verbose: bool = True,
) -> BendersResult:
    """Run SCOPF on a network with Benders decomposition.

    Args:
        network_loader: Function that returns a PowerNetwork.
        network_name: Display name for the network.
        solver: Solver backend ('moreau' or 'mosek').
        device: Device for moreau solver ('cpu' or 'cuda').
        batch_size: Number of scenarios to solve.
        n_contingencies: Number of contingencies to monitor.
        seed: Random seed for scenario generation.
        max_iterations: Maximum Benders iterations.
        violation_tol: Violation tolerance in MW.
        verbose: Whether to print progress.

    Returns:
        BendersResult with solution.
    """
    net = network_loader()

    if verbose:
        print("=" * 70)
        print("Security-Constrained DC Optimal Power Flow (SCOPF)")
        print(f"{network_name} - Benders Decomposition ({solver})")
        print("=" * 70)

        print(f"\nNetwork Statistics:")
        print(f"  Buses:      {net.n_bus}")
        print(f"  Generators: {net.n_gen}")
        print(f"  Branches:   {net.n_branch}")
        print(f"  Total load: {net.load_p.sum():.1f} MW")
        print(f"  Total gen capacity: {net.gen_pmax.sum():.1f} MW")

        print("\n--- Computing Sensitivity Factors ---")

    t0 = time.perf_counter()
    ptdf = compute_ptdf(net, use_gpu=(device == "cuda"))
    lodf = compute_lodf(net, ptdf, use_gpu=(device == "cuda"))
    sens_time = time.perf_counter() - t0

    if verbose:
        print(f"  PTDF shape: {ptdf.shape}")
        print(f"  LODF shape: {lodf.shape}")
        print(f"  Computation time: {sens_time:.4f}s")

        print("\n--- Selecting Contingencies ---")

    contingencies = select_contingencies(net, lodf, max_contingencies=n_contingencies)

    if verbose:
        print(f"  Selected {len(contingencies)} contingencies")

    # Generate scenarios
    if batch_size == 1:
        scenarios = None  # Use base case
    else:
        scenarios = generate_scenarios(net, batch_size, seed=seed)

    if verbose:
        print(f"\n--- Benders Decomposition (solver={solver}, batch_size={batch_size}) ---")

    solver_obj = BendersSCOPF(
        net=net,
        contingencies=contingencies,
        lodf=lodf,
        scenarios=scenarios,
        solver=solver,
        device=device,
        verbose=verbose,
        max_iterations=max_iterations,
        violation_tol=violation_tol,
    )

    t0 = time.perf_counter()
    result = solver_obj.solve()
    total_time = time.perf_counter() - t0

    if verbose:
        print(f"\n--- Summary ---")
        solver_label = f"{solver} ({device})" if solver == "moreau" else solver
        print(f"  Solver: {solver_label}")
        print(f"  Batch size: {batch_size}")
        print(f"  Status: {result.status}")
        if batch_size == 1:
            print(f"  Objective: ${result.objective:,.2f}")
        else:
            print(f"  Avg objective: ${result.objective:,.2f}")
            print(f"  Objectives: {[f'${o:,.0f}' for o in result.objectives]}")
        print(f"  Benders iterations: {result.iterations}")
        print(f"  Total cuts added: {result.cuts_added}")
        print(f"  Total solve time: {result.total_solve_time:.4f}s")
        print(f"  Total wall time: {total_time:.4f}s")

        n_base_cuts = sum(1 for c in result.active_constraints if c.contingency is None)
        n_cont_cuts = result.cuts_added - n_base_cuts
        print(f"  Base case cuts: {n_base_cuts}")
        print(f"  Contingency cuts: {n_cont_cuts}")

        print("\n" + "=" * 70)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCOPF Example with Benders Decomposition")
    parser.add_argument(
        "--contingencies", "-c", type=int, default=20, help="Number of contingencies to monitor"
    )
    parser.add_argument(
        "--solver",
        "-s",
        type=str,
        default="moreau",
        choices=["moreau", "mosek"],
        help="Solver backend (moreau or mosek)",
    )
    parser.add_argument(
        "--network",
        "-n",
        type=str,
        default="ieee118",
        choices=["ieee118", "texas2k", "texas7k"],
        help="Network to solve (ieee118, texas2k, texas7k)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for moreau solver (cpu or cuda)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=str,
        default="1",
        help="Batch size(s) - single int or comma-separated for --compare (default: 1)",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Run solver comparison (Moreau vs Mosek)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for scenario generation (default: 42)"
    )
    args = parser.parse_args()

    network_config = {
        "ieee118": {
            "loader": load_ieee118,
            "name": "IEEE 118-Bus",
        },
        "texas2k": {
            "loader": load_activsg2000,
            "name": "Texas 2000-Bus (ACTIVSg2000)",
        },
        "texas7k": {
            "loader": load_texas7k,
            "name": "Texas7k (6717-Bus)",
        },
    }
    config = network_config[args.network]

    # Parse batch sizes (single int or comma-separated list)
    batch_sizes = [int(x.strip()) for x in args.batch_size.split(",")]

    if args.compare:
        run_batch_comparison(
            network_loader=config["loader"],
            network_name=config["name"],
            batch_sizes=batch_sizes,
            n_contingencies=args.contingencies,
            solvers=["moreau", "mosek"],
            device=args.device,
            seed=args.seed,
        )
    else:
        run_network(
            network_loader=config["loader"],
            network_name=config["name"],
            solver=args.solver,
            device=args.device,
            batch_size=batch_sizes[0],
            n_contingencies=args.contingencies,
            seed=args.seed,
        )
