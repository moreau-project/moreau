#![allow(non_snake_case)]
use super::*;
use crate::algebra::*;
use crate::solver::core::traits::Residuals;
use crate::utils::debug::debug_println;

// ---------------
// Residuals type for default problem format
// ---------------

/// Standard-form solver type implementing the [`Residuals`](crate::solver::core::traits::Residuals) trait
#[derive(Debug)]
pub struct DefaultResiduals<T> {
    // the main KKT residuals
    pub(crate) rx: Vec<T>,
    pub(crate) rz: Vec<T>,
    pub(crate) rτ: T,

    // partial residuals for infeasibility checks
    pub(crate) rx_inf: Vec<T>,
    pub(crate) rz_inf: Vec<T>,

    // various inner products.
    // NB: these are invariant w.r.t equilibration
    pub(crate) dot_qx: T,
    pub(crate) dot_bz: T,
    pub(crate) dot_sz: T,
    pub(crate) dot_xPx: T,

    // the product Px by itself. Required for infeasibilty checks
    pub(crate) Px: Vec<T>,
}

impl<T> DefaultResiduals<T>
where
    T: FloatT,
{
    /// Create a new `DefaultResiduals` object
    pub fn new(n: usize, m: usize) -> Self {
        let rx = vec![T::zero(); n];
        let rz = vec![T::zero(); m];
        let rτ = T::one();

        let rx_inf = vec![T::zero(); n];
        let rz_inf = vec![T::zero(); m];

        let Px = vec![T::zero(); n];

        Self {
            rx,
            rz,
            rτ,
            rx_inf,
            rz_inf,
            Px,
            dot_qx: T::zero(),
            dot_bz: T::zero(),
            dot_sz: T::zero(),
            dot_xPx: T::zero(),
        }
    }
}

impl<T> Residuals<T> for DefaultResiduals<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;

    fn update(&mut self, variables: &DefaultVariables<T>, data: &DefaultProblemData<T>) {
        // various products used multiple times
        let qx = data.q.dot(&variables.x);
        let bz = data.b.dot(&variables.z);
        let sz = variables.s.dot(&variables.z);

        //Px = P*x, P treated as symmetric
        let symP = data.P.sym_up();
        symP.symv(&mut self.Px, &variables.x, T::one(), T::zero());

        let xPx = variables.x.dot(&self.Px);

        //partial residual calc so we can check primal/dual
        //infeasibility conditions

        //Same as:
        //rx_inf .= -data.A'* variables.z
        let At = data.A.t();
        At.gemv(&mut self.rx_inf, &variables.z, -T::one(), T::zero());

        // Direct-x cones use the Lagrangian convention `L += z_J'(s_J - x[J])`
        // with `s_J ≡ x[J]`, so stationarity reads
        //   Px + qτ + A'z - Σ_J E_J' z_J = 0
        // (dual sign flipped vs. slack). Folding `+z_J` into `rx_inf` at the
        // direct-x indices makes `rx_inf` carry the full primal-infeasibility
        // certificate residual `-(A'z - Σ_J E_J' z_J)`, which is what the
        // HSDE check `‖rx_inf‖ < tol·|b'z|` actually needs to fire on direct-x
        // problems. Downstream `rx = rx_inf − Px − qτ` then automatically
        // picks up the direct-x contribution without a second scatter.
        let mut off = 0usize;
        for xcone in &data.x_cones {
            let indices = xcone.indices();
            for (k, &idx) in indices.iter().enumerate() {
                self.rx_inf[idx] += variables.z_x[off + k];
            }
            off += indices.len();
        }

        //Same as:  residuals.rz_inf .=  data.A * variables.x + variables.s
        self.rz_inf.copy_from(&variables.s);
        let A = &data.A;
        A.gemv(&mut self.rz_inf, &variables.x, T::one(), T::one());

        //complete the residuals
        //rx = rx_inf - Px - qτ
        self.rx.waxpby(-T::one(), &self.Px, -variables.τ, &data.q);
        self.rx.axpby(T::one(), &self.rx_inf, T::one());

        // rz = rz_inf - bτ
        self.rz
            .waxpby(T::one(), &self.rz_inf, -variables.τ, &data.b);

        // rτ = qx + bz + κ + xPx/τ
        self.rτ = qx + bz + variables.κ + xPx / variables.τ;

        //save local versions
        self.dot_qx = qx;
        self.dot_bz = bz;
        self.dot_sz = sz;
        self.dot_xPx = xPx;

        debug_println!("self after update: {:?}", self);
    }
}
