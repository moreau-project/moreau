"""
Test to catch CPU/GPU discrepancy in exp cone backward pass.

This test demonstrates that CPU and GPU produce significantly different
gradients for exp cone problems (differences of 0.5+, not numerical precision).
"""

import numpy as np
import pytest
import scipy.sparse as sp

import moreau

requires_gpu = pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")


@requires_gpu
def test_exp_cone_backward_cpu_gpu_bug():
    """
    Test that exposes CPU/GPU gradient discrepancy in exp cone backward pass.

    This example was found via fuzzing and produces large gradient
    differences between CPU and GPU (not numerical precision issues).
    """
    # Problem found via fuzzing that triggers the bug
    P_data = [
        0.4115993608106344,
        3.7699432135898334,
        0.1,
        10.0,
        2.3998657121083418,
        0.99,
        2.00001,
        3.4791078924598207,
        2.773852279532794,
    ]
    P_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    P_indptr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    P = sp.csr_matrix((P_data, P_indices, P_indptr), shape=(9, 9))

    A_data = [
        4.67e-223,
        -4.0,
        3.927784848257472,
        -8.08e-144,
        -2.1628942004276337,
        1.1,
        1.1,
        5e-12,
        1e-06,
        2.4230084688878826,
        -9.34e-109,
        -0.2858574846429631,
        4.006941508158899,
        0.3333333333333333,
        -3.68e-283,
        -6.97e-287,
        -3.4806852569816282,
        4.748535125382054,
    ]
    A_indices = [7, 0, 1, 6, 7, 2, 3, 5, 0, 2, 7, 1, 4, 0, 5, 6, 7, 8]
    A_indptr = [0, 1, 5, 8, 11, 13, 18]
    A = sp.csr_matrix((A_data, A_indices, A_indptr), shape=(6, 9))

    q = np.array(
        [
            0.0,
            -1.1e-196,
            6.755841550307146,
            0.99,
            1.175494351e-38,
            9.932883757319363,
            8.3755448216743,
            -2.112871995707561,
            -7.712077961168307,
        ]
    )
    b = np.array(
        [-8.480659849069768, 5e-12, 2.5265503085517853, -1.175494351e-38, 8.468053690234065, 5e-324]
    )
    cones = moreau.Cones(num_exp_cones=2)

    # CPU solve
    cpu_solver = moreau.Solver(
        P, q=q, A=A, b=b, cones=cones, settings=moreau.Settings(device="cpu", enable_grad=True)
    )
    cpu_solver.solve()

    # GPU solve
    gpu_solver = moreau.Solver(
        P, q=q, A=A, b=b, cones=cones, settings=moreau.Settings(device="cuda", enable_grad=True)
    )
    gpu_solver.solve()

    assert cpu_solver.info.status.name in ["Solved", "AlmostSolved"]
    assert gpu_solver.info.status.name in ["Solved", "AlmostSolved"]

    # Backward pass
    dx_bar = np.ones(len(q))
    cpu_grads = cpu_solver.backward(dx_bar)
    gpu_grads = gpu_solver.backward(dx_bar)

    print(f"CPU dq: {cpu_grads['dq']}")
    print(f"GPU dq: {gpu_grads['dq']}")
    print(f"dq difference: {np.abs(cpu_grads['dq'] - gpu_grads['dq'])}")

    # This should fail due to the bug - differences are ~2.0, not ~1e-6
    np.testing.assert_allclose(
        cpu_grads["dq"],
        gpu_grads["dq"],
        rtol=1e-2,
        atol=1e-3,
        err_msg="CPU/GPU exp cone backward pass produces different gradients",
    )
