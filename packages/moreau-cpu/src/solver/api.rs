#![allow(non_snake_case)]

//! High-level Rust API for moreau solver.
//!
//! This module provides two solver types:
//!
//! ## `Solver` - Single Problem
//! All data is provided in the constructor, then call `solve()`:
//! ```no_run
//! use moreau::solver::api::{Cones, Solver, Settings};
//! use moreau::algebra::CscMatrix;
//!
//! let P = CscMatrix::identity(2);
//! let q = vec![-1.0, -2.0];
//! let A = CscMatrix::zeros((3, 2));
//! let b = vec![0.0, 1.0, 1.0];
//! let cones = Cones::new(1, 2);
//!
//! let mut solver = Solver::new(&P, &q, &A, &b, &cones, None).unwrap();
//! let solution = solver.solve().unwrap();
//! println!("x = {:?}", solution.x);
//! ```
//!
//! ## `CompiledSolver` - Multiple Problems with Shared Structure
//! Structure is defined at construction, matrix values via `setup()`, solve parameters via `solve()`:
//! ```no_run
//! use moreau::solver::{SupportedConeT, DefaultSettings, CompiledSolver};
//!
//! // Create cone constraints as Vec<SupportedConeT>
//! let cones = vec![
//!     SupportedConeT::ZeroConeT(1),      // 1 equality constraint
//!     SupportedConeT::NonnegativeConeT(2), // 2 inequality constraints
//! ];
//! let settings = DefaultSettings::default();
//!
//! let mut solver = CompiledSolver::new(
//!     2, 3,
//!     &[0, 1, 2], &[0, 1],  // P pattern (CSR)
//!     &[0, 2, 3, 4], &[0, 1, 0, 1],  // A pattern (CSR)
//!     &cones, settings, 4, false
//! ).unwrap();
//!
//! // Set matrix values (batch of 4 problems)
//! let P_values = vec![vec![1.0, 1.0]; 4];
//! let A_values = vec![vec![1.0, -2.0, 1.0, 1.0]; 4];
//! solver.setup(&P_values, &A_values);
//!
//! // Solve with per-problem parameters
//! let qs = vec![vec![-1.0, -4.0]; 4];
//! let bs = vec![vec![0.0, 1.0, 1.0]; 4];
//! let solutions = solver.solve(&qs, &bs).unwrap();
//! ```

use crate::algebra::{CscMatrix, FloatT};
use crate::solver::core::cones::SupportedConeT::{self, *};
use crate::solver::implementations::default::{
    DefaultSettings, DefaultSolution, DefaultSolver, SolverError,
};
use crate::solver::IPSolver;

/// Cone specification for conic optimization problems.
///
/// This struct provides a simple way to specify cones that matches
/// the Python API. It gets converted to `Vec<SupportedConeT>` internally.
///
/// # Example
/// ```
/// use moreau::solver::api::Cones;
///
/// let cones = Cones {
///     num_zero: 1,        // 1 equality constraint
///     num_nonneg: 4,      // 4 inequality constraints
///     soc_dims: vec![3, 5], // 2 SOCs with dimensions 3 and 5
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default)]
pub struct Cones {
    /// Number of equality constraint dimensions (zero cone dimension)
    pub num_zero: usize,
    /// Number of inequality constraint dimensions (nonnegative cone dimension)
    pub num_nonneg: usize,
    /// Dimensions of second-order cones (each >= 2)
    pub soc_dims: Vec<usize>,
    /// Number of exponential cones (each is 3D)
    pub num_exp: usize,
    /// Power cone alpha parameters (each cone is 3D with parameter alpha)
    pub power_alphas: Vec<f64>,
}

impl Cones {
    /// Create cones with just zero (equality) and nonnegative (inequality) cones.
    pub fn new(num_zero: usize, num_nonneg: usize) -> Self {
        Self {
            num_zero,
            num_nonneg,
            ..Default::default()
        }
    }

    /// Total number of constraint rows across all cones.
    pub fn total_constraints(&self) -> usize {
        let mut total = self.num_zero + self.num_nonneg;
        total += self.soc_dims.iter().sum::<usize>();
        total += 3 * self.num_exp;
        total += 3 * self.power_alphas.len();
        total
    }

    /// Convert to internal cone representation.
    pub fn to_supported_cones<T: FloatT>(&self) -> Vec<SupportedConeT<T>> {
        let mut cones = Vec::new();

        if self.num_zero > 0 {
            cones.push(ZeroConeT(self.num_zero));
        }
        if self.num_nonneg > 0 {
            cones.push(NonnegativeConeT(self.num_nonneg));
        }

        for &dim in &self.soc_dims {
            cones.push(SecondOrderConeT(dim));
        }

        for _ in 0..self.num_exp {
            cones.push(ExponentialConeT());
        }

        for &alpha in &self.power_alphas {
            cones.push(PowerConeT(T::from_f64(alpha).unwrap()));
        }

        cones
    }
}

/// Solver settings with sensible defaults.
pub type Settings = DefaultSettings<f64>;

/// Solver result.
pub type Solution = DefaultSolution<f64>;

/// Single-problem solver.
///
/// All problem data is provided in the constructor, then call `solve()`.
///
/// # Problem Formulation
/// ```text
/// minimize    (1/2)x'Px + q'x
/// subject to  Ax + s = b
///             x ∈ K1,  s ∈ K2
/// ```
/// K2 constrains the slack s; K1 constrains x directly (direct-x cones).
///
/// # Example
/// ```no_run
/// use moreau::solver::api::{Cones, Solver};
/// use moreau::algebra::CscMatrix;
///
/// // P = diag(6, 4)
/// let P = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![6.0, 4.0]);
/// let q = vec![-1.0, -4.0];
///
/// // A = [[1, -2], [1, 0], [0, 1], [-1, 0], [0, -1]]
/// let A = CscMatrix::new(5, 2,
///     vec![0, 3, 6],
///     vec![0, 1, 3, 0, 2, 4],
///     vec![1.0, 1.0, -1.0, -2.0, 1.0, -1.0]
/// );
/// let b = vec![0.0, 1.0, 1.0, 1.0, 1.0];
///
/// let cones = Cones { num_zero: 1, num_nonneg: 4, ..Default::default() };
///
/// let mut solver = Solver::new(&P, &q, &A, &b, &cones, None).unwrap();
/// let solution = solver.solve().unwrap();
/// println!("x = {:?}", solution.x);
/// ```
pub struct Solver {
    /// Internal solver instance
    inner: DefaultSolver<f64>,
}

impl Solver {
    /// Create a new solver with all problem data.
    ///
    /// # Arguments
    /// * `P` - Quadratic objective matrix (CSC format, symmetric, only upper triangle used)
    /// * `q` - Linear objective vector
    /// * `A` - Constraint matrix (CSC format)
    /// * `b` - Constraint RHS vector
    /// * `cones` - Cone constraints specification
    /// * `settings` - Optional solver settings (uses defaults if None)
    pub fn new(
        P: &CscMatrix<f64>,
        q: &[f64],
        A: &CscMatrix<f64>,
        b: &[f64],
        cones: &Cones,
        settings: Option<Settings>,
    ) -> Result<Self, SolverError> {
        let settings = settings.unwrap_or_default();
        let supported_cones = cones.to_supported_cones::<f64>();

        let inner = DefaultSolver::new(P, q, A, b, &supported_cones, settings)?;

        Ok(Self { inner })
    }

    /// Solve the optimization problem.
    ///
    /// Returns reference to the solution containing primal/dual variables and solver status.
    pub fn solve(&mut self) -> Result<&DefaultSolution<f64>, SolverError> {
        self.inner.solve();
        Ok(&self.inner.solution)
    }

    /// Get the last solution without re-solving.
    pub fn solution(&self) -> &DefaultSolution<f64> {
        &self.inner.solution
    }

    /// Number of primal variables.
    pub fn n(&self) -> usize {
        self.inner.data.n
    }

    /// Number of constraints.
    pub fn m(&self) -> usize {
        self.inner.data.m
    }
}

// Re-export CompiledSolver from the implementations module
pub use crate::solver::implementations::default::CompiledSolver;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cones_to_supported() {
        let cones = Cones {
            num_zero: 1,
            num_nonneg: 2,
            soc_dims: vec![3],
            ..Default::default()
        };

        let supported: Vec<SupportedConeT<f64>> = cones.to_supported_cones();
        assert_eq!(supported.len(), 3);
    }

    #[test]
    fn test_simple_qp() {
        // Simple QP: min (1/2)x'Px + q'x s.t. Ax + s = b, s in K
        // P = diag(6, 4), q = [-1, -4]
        // A has equality and inequality constraints
        let P = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![6.0, 4.0]);
        let q = vec![-1., -4.];

        // Build A in CSC format
        // A = [[1, -2], [1, 0], [0, 1], [-1, 0], [0, -1]]
        let A = CscMatrix::new(
            5,
            2,
            vec![0, 3, 6],
            vec![0, 1, 3, 0, 2, 4],
            vec![1.0, 1.0, -1.0, -2.0, 1.0, -1.0],
        );
        let b = vec![0., 1., 1., 1., 1.];

        let cones = Cones {
            num_zero: 1,
            num_nonneg: 4,
            ..Default::default()
        };

        let mut solver = Solver::new(&P, &q, &A, &b, &cones, None).unwrap();
        let solution = solver.solve().unwrap();

        // Check solution is close to expected
        println!("x = {:?}", solution.x);
        assert!(
            (solution.x[0] - 0.4286).abs() < 0.01,
            "x[0] = {}",
            solution.x[0]
        );
        assert!(
            (solution.x[1] - 0.2143).abs() < 0.01,
            "x[1] = {}",
            solution.x[1]
        );
    }

    #[test]
    fn test_qp_with_equality() {
        // QP with equality constraint: min (1/2)x'Px s.t. sum(x) = 1
        let P = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 2.0]);
        let q = vec![0., 0.];

        // A = [1, 1] (equality constraint)
        let A = CscMatrix::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, 1.0]);
        let b = vec![1.0];

        let cones = Cones::new(1, 0);

        let mut solver = Solver::new(&P, &q, &A, &b, &cones, None).unwrap();
        let solution = solver.solve().unwrap();

        // Solution should be x = [0.5, 0.5]
        println!("x = {:?}", solution.x);
        assert!(
            (solution.x[0] - 0.5).abs() < 0.01,
            "x[0] = {}",
            solution.x[0]
        );
        assert!(
            (solution.x[1] - 0.5).abs() < 0.01,
            "x[1] = {}",
            solution.x[1]
        );
    }

    #[test]
    fn test_qp_with_inequality() {
        // QP with inequality constraints: min (1/2)x'Px s.t. sum(x) = 1, x >= 0
        // For Ax + s = b with s >= 0 to represent x >= 0, we need:
        //   -x + s = 0, s >= 0 => s = x >= 0
        let P = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 2.0]);
        let q = vec![0., 0.];

        // A = [[1, 1], [-1, 0], [0, -1]] for sum(x) = 1, x1 >= 0, x2 >= 0
        let A = CscMatrix::new(
            3,
            2,
            vec![0, 2, 4],              // colptr
            vec![0, 1, 0, 2], // rowval: col0 has entries at rows 0,1; col1 has entries at rows 0,2
            vec![1.0, -1.0, 1.0, -1.0], // values: (0,0)=1, (1,0)=-1, (0,1)=1, (2,1)=-1
        );
        let b = vec![1.0, 0.0, 0.0];

        let cones = Cones::new(1, 2);

        let mut solver = Solver::new(&P, &q, &A, &b, &cones, None).unwrap();
        let solution = solver.solve().unwrap();

        // Solution should be x = [0.5, 0.5]
        println!("x = {:?}", solution.x);
        assert!(
            (solution.x[0] - 0.5).abs() < 0.01,
            "x[0] = {}",
            solution.x[0]
        );
        assert!(
            (solution.x[1] - 0.5).abs() < 0.01,
            "x[1] = {}",
            solution.x[1]
        );
    }

    #[test]
    fn test_nonneg_only_qp() {
        // QP with ONLY nonneg cones (no zero cones)
        // min 0.5 x'Px + q'x s.t. Ax + s = b, s >= 0
        // P = diag([2, 2])
        // q = [1, 2]
        // A = diag([1, 1])
        // b = [5, 5]
        // Expected: x = [-0.5, -1.0] (unconstrained optimum, which is feasible)

        let P = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 2.0]);
        let q = vec![1.0, 2.0];
        let A = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]);
        let b = vec![5.0, 5.0];

        let cones = Cones::new(0, 2); // 0 zero cones, 2 nonneg cones

        let settings = DefaultSettings {
            verbose: true,
            ..Default::default()
        };
        let mut solver = Solver::new(&P, &q, &A, &b, &cones, Some(settings)).unwrap();
        let solution = solver.solve().unwrap();

        println!("x = {:?}", solution.x);
        let obj = 0.5 * (solution.x[0].powi(2) * 2.0 + solution.x[1].powi(2) * 2.0)
            + solution.x[0] * 1.0
            + solution.x[1] * 2.0;
        println!("objective = {}", obj);

        // Expected solution: x = [-0.5, -1.0], obj = -1.25
        assert!(
            (solution.x[0] - (-0.5)).abs() < 0.01,
            "x[0] should be -0.5, got {}",
            solution.x[0]
        );
        assert!(
            (solution.x[1] - (-1.0)).abs() < 0.01,
            "x[1] should be -1.0, got {}",
            solution.x[1]
        );
    }
}
