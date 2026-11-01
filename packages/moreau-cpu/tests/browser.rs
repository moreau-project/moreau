//! Portable solve and differentiation checks, executed natively and in a browser.
use moreau::{algebra::CscMatrix, solver::*};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

fn settings() -> DefaultSettings<f64> {
    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    settings.ipm.diff_method = DiffMethod::Exact;
    settings
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn solves_qp_lp_socp_and_infeasible() {
    let identity = CscMatrix::identity(1);
    for (p, q, b, cones, expected) in [
        (
            identity.clone(),
            vec![-2.0],
            vec![1.0],
            vec![NonnegativeConeT(1)],
            1.0,
        ),
        (
            CscMatrix::zeros((1, 1)),
            vec![-1.0],
            vec![3.0],
            vec![NonnegativeConeT(1)],
            3.0,
        ),
    ] {
        let mut solver = DefaultSolver::new(&p, &q, &identity, &b, &cones, settings()).unwrap();
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::Solved);
        assert!((solver.solution.x[0] - expected).abs() < 1e-6);
        assert!(solver.solution.solve_time.is_finite());
    }
    let p = CscMatrix::zeros((2, 2));
    let a = CscMatrix::new_from_triplets(3, 2, vec![0, 1, 2], vec![1, 0, 1], vec![1.0, -1.0, -1.0]);
    let mut soc = DefaultSolver::new(
        &p,
        &[1.0, 0.0],
        &a,
        &[2.0, 0.0, 0.0],
        &[ZeroConeT(1), SecondOrderConeT(2)],
        settings(),
    )
    .unwrap();
    soc.solve();
    assert_eq!(soc.solution.status, SolverStatus::Solved);
    assert!((soc.solution.x[0] - 2.0).abs() < 1e-6);
    let a = CscMatrix::new_from_triplets(2, 1, vec![0, 1], vec![0, 0], vec![1.0, -1.0]);
    let mut infeasible = DefaultSolver::new(
        &identity,
        &[0.0],
        &a,
        &[0.0, -1.0],
        &[NonnegativeConeT(2)],
        settings(),
    )
    .unwrap();
    infeasible.solve();
    assert_eq!(infeasible.solution.status, SolverStatus::PrimalInfeasible);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn forward_adjoint_and_finite_difference() {
    let p = CscMatrix::identity(1);
    let a = CscMatrix::identity(1);
    let cones = [NonnegativeConeT(1)];
    let mut solver = DefaultSolver::new(&p, &[-2.0], &a, &[1.0], &cones, settings()).unwrap();
    solver.solve();
    let sol = &solver.solution;
    let zero = CscMatrix::zeros((1, 1));
    let forward = diff::differentiate(
        &p,
        &[-2.0],
        &a,
        &[1.0],
        &cones,
        &sol.x,
        &sol.s,
        &sol.z,
        1.0,
        &zero,
        &[0.0],
        &zero,
        &[1.0],
        DiffMethod::Exact,
        0.0,
    );
    let backward = diff::differentiate_adjoint(
        &p,
        &[-2.0],
        &a,
        &[1.0],
        &cones,
        &sol.x,
        &sol.s,
        &sol.z,
        1.0,
        &[1.0],
        &[0.25],
        &[0.0],
        DiffMethod::Exact,
        0.0,
    );
    assert!((forward.dx[0] + 0.25 * forward.dz[0] - backward.db[0]).abs() < 1e-6);
    let mut endpoints = Vec::new();
    for delta in [-1e-3, 1e-3] {
        let mut perturbed =
            DefaultSolver::new(&p, &[-2.0], &a, &[1.0 + delta], &cones, settings()).unwrap();
        perturbed.solve();
        endpoints.push(perturbed.solution.x[0]);
    }
    assert!((forward.dx[0] - (endpoints[1] - endpoints[0]) / 2e-3).abs() < 1e-5);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn repeated_batches_keep_order_and_gradient_state() {
    for threads in [0, 1] {
        let mut solver = CompiledSolver::new(
            1,
            1,
            &[0, 1],
            &[0],
            &[0, 1],
            &[0],
            &[NonnegativeConeT(1)],
            settings(),
            threads,
            true,
        )
        .unwrap();
        assert!(solver.num_threads() >= 1);
        let qs = vec![vec![-3.0], vec![-4.0], vec![-5.0]];
        let bs = vec![vec![1.0], vec![1.5], vec![2.0]];
        solver.setup_shared(&[1.0], &[1.0], 3);
        for _ in 0..2 {
            let results = solver.solve(&qs, &bs).unwrap();
            for (sol, b) in results.iter().zip(&bs) {
                assert_eq!(sol.status, SolverStatus::Solved);
                assert!((sol.x[0] - b[0]).abs() < 1e-6);
                assert!(sol.solve_time.is_finite());
            }
            let upstream: Vec<_> = (0..3)
                .map(|_| UpstreamGradients {
                    dx: vec![1.0],
                    ds: vec![0.0],
                    dz: vec![0.0],
                    dz_x: vec![],
                })
                .collect();
            for gradient in solver.backward(&upstream).unwrap() {
                assert!((gradient.db[0] - 1.0).abs() < 1e-5);
            }
            let xs: Vec<_> = results.iter().map(|s| s.x.clone()).collect();
            let zs: Vec<_> = results.iter().map(|s| s.z.clone()).collect();
            let ss: Vec<_> = results.iter().map(|s| s.s.clone()).collect();
            let matrices = vec![vec![1.0]; 3];
            for gradient in solver
                .backward_with_data(&upstream, &matrices, &matrices, &qs, &bs, &xs, &zs, &ss)
                .unwrap()
            {
                assert!((gradient.db[0] - 1.0).abs() < 1e-5);
            }
        }
        solver
            .setup_flat(&[2.0, 2.0, 2.0], &[1.0, 1.0, 1.0], 3)
            .unwrap();
        let problems: Vec<_> = qs
            .iter()
            .zip(&bs)
            .map(|(q, b)| BatchProblem {
                P_values: vec![2.0],
                A_values: vec![1.0],
                q: q.clone(),
                b: b.clone(),
            })
            .collect();
        for (sol, b) in solver
            .solve_batch_parallel(&problems)
            .unwrap()
            .iter()
            .zip(&bs)
        {
            assert!((sol.x[0] - b[0]).abs() < 1e-5);
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test]
fn rejects_browser_threads() {
    let result = CompiledSolver::new(
        1,
        1,
        &[0, 1],
        &[0],
        &[0, 1],
        &[0],
        &[NonnegativeConeT(1)],
        settings(),
        2,
        false,
    );
    assert!(matches!(result, Err(SolverError::BadInputData(_))));
}
