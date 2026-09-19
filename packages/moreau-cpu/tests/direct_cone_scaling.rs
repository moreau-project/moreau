/*
Copyright, the Moreau authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use moreau::{algebra::*, solver::*};

fn settings(equilibrate: bool) -> DefaultSettings<f64> {
    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    settings.ipm.equilibrate_enable = equilibrate;
    settings.ipm.presolve_enable = false;
    settings
}

fn solve(equilibrate: bool) -> DefaultSolver<f64> {
    // The cone order differs from variable order; coordinate 1 is free.
    let p = CscMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![4.0, 2.0, 0.25]);
    let a = CscMatrix::zeros((0, 3));
    let cones = vec![SupportedXConeT::NonnegativeXConeT(vec![2, 0])];
    let mut solver = DefaultSolver::new_with_xcones(
        &p,
        &[2.0, -6.0, 3.0],
        &a,
        &[],
        &[],
        &cones,
        settings(equilibrate),
    )
    .unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    solver
}

#[test]
fn direct_nonneg_dual_satisfies_unscaled_stationarity() {
    for equilibrate in [false, true] {
        let solver = solve(equilibrate);
        let sol = &solver.solution;
        for (k, j) in [2, 0].into_iter().enumerate() {
            let residual = [4.0, 2.0, 0.25][j] * sol.x[j] + [2.0, -6.0, 3.0][j];
            assert!(
                (sol.z_x[k] - residual).abs() < 1e-6,
                "equilibrate={equilibrate}, z_x[{k}]={}, stationarity={residual}",
                sol.z_x[k]
            );
        }
    }
}

#[test]
fn direct_nonneg_dual_gradient_uses_unscaled_coordinates() {
    for equilibrate in [false, true] {
        let solver = solve(equilibrate);
        // Both constrained coordinates are strictly active: z_x[0] = q[2].
        let grad = solver
            .backward_batch_with_dz_x(&[0.0; 3], &[], &[], &[1.0, 0.0], None, DiffMethod::Exact)
            .unwrap();
        for (j, expected) in [0.0, 0.0, 1.0].into_iter().enumerate() {
            assert!(
                (grad.dq[j] - expected).abs() < 1e-6,
                "equilibrate={equilibrate}, dq[{j}]={}",
                grad.dq[j]
            );
        }
    }
}

#[test]
fn direct_nonneg_accepts_external_optimal_warm_start() {
    for equilibrate in [false, true] {
        let mut config = settings(equilibrate);
        config.max_iter = 0;
        let mut solver = CompiledSolver::new_with_xcones(
            3,
            0,
            &[0, 1, 2, 3],
            &[0, 1, 2],
            &[0],
            &[],
            &[],
            &[SupportedXConeT::NonnegativeXConeT(vec![2, 0])],
            config,
            1,
            false,
        )
        .unwrap();
        solver.setup(&[vec![4.0, 2.0, 0.25]], &[vec![]]).unwrap();
        let solutions = solver
            .solve_with_warm_start(
                &[vec![2.0, -6.0, 3.0]],
                &[vec![]],
                Some(&[vec![0.0, 3.0, 0.0]]),
                Some(&[vec![]]),
                Some(&[vec![]]),
                Some(&[vec![3.0, 2.0]]),
            )
            .unwrap();
        assert_eq!(
            solutions[0].status,
            SolverStatus::Solved,
            "equilibrate={equilibrate}, residual={}",
            solutions[0].r_dual
        );
        assert_eq!(solutions[0].iterations, 0);
    }
}

#[test]
fn direct_nonneg_warm_start_across_new_equilibration() {
    let cold = solve(true).solution;
    let mut solver = CompiledSolver::new_with_xcones(
        3,
        0,
        &[0, 1, 2, 3],
        &[0, 1, 2],
        &[0],
        &[],
        &[],
        &[SupportedXConeT::NonnegativeXConeT(vec![2, 0])],
        settings(true),
        1,
        true,
    )
    .unwrap();
    // Distinct diagonal matrices and objective scales in each batch.
    solver
        .setup(
            &[vec![16.0, 2.0, 1.0], vec![1.0, 4.0, 0.0625]],
            &[vec![], vec![]],
        )
        .unwrap();
    let qs = [vec![2.1, -6.0, 3.1], vec![1.9, -12.0, 2.9]];
    let solutions = solver
        .solve_with_warm_start(
            &qs,
            &[vec![], vec![]],
            Some(&[cold.x.clone(), cold.x.clone()]),
            Some(&[cold.z.clone(), cold.z.clone()]),
            Some(&[cold.s.clone(), cold.s.clone()]),
            Some(&[cold.z_x.clone(), cold.z_x.clone()]),
        )
        .unwrap();
    for (i, sol) in solutions.iter().enumerate() {
        assert_eq!(sol.status, SolverStatus::Solved);
        assert!(sol.iterations < cold.iterations);
        for (j, expected) in [0.0, 3.0, 0.0].into_iter().enumerate() {
            assert!((sol.x[j] - expected).abs() < 1e-6);
        }
        for (k, j) in [2, 0].into_iter().enumerate() {
            assert!((sol.z_x[k] - qs[i][j]).abs() < 1e-6);
        }
    }
    let upstream = || UpstreamGradients {
        dx: vec![0.0; 3],
        dz: vec![],
        ds: vec![],
        dz_x: vec![1.0, 0.0],
    };
    let gradients = solver.backward(&[upstream(), upstream()]).unwrap();
    for grad in gradients {
        for (j, expected) in [0.0, 0.0, 1.0].into_iter().enumerate() {
            assert!((grad.dq[j] - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn direct_nonneg_unscaled_infeasibility_certificate() {
    // 8x = -8, x >= 0. The certificate must satisfy 8z - z_x = 0.
    let mut solver = DefaultSolver::new_with_xcones(
        &CscMatrix::zeros((1, 1)),
        &[0.0],
        &CscMatrix::from(&[[8.0]]),
        &[-8.0],
        &[ZeroConeT(1)],
        &[SupportedXConeT::NonnegativeXConeT(vec![0])],
        settings(true),
    )
    .unwrap();
    solver.solve();
    let sol = solver.solution;
    assert_eq!(sol.status, SolverStatus::PrimalInfeasible);
    assert!(sol.z[0] > 0.0);
    assert!(sol.z_x[0] > 0.0);
    assert!((8.0 * sol.z[0] - sol.z_x[0]).abs() < 1e-7 * sol.z_x[0]);
}
