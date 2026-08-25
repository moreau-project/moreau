import numpy as np
from types import SimpleNamespace

import moreau_cpu._cpu as cpu_mod


def test_active_set_backward_with_data_flat_delegates_to_backend():
    calls = {}

    class FakeBackendSolver:
        def backward_with_data_flat(
            self,
            dx_flat,
            ds_flat,
            dz_flat,
            P_values_flat,
            A_values_flat,
            q_flat,
            b_flat,
            x_flat,
            z_flat,
            s_flat,
            backward_state,
            shared,
        ):
            calls["args"] = {
                "dx_flat": np.array(dx_flat, copy=True),
                "ds_flat": np.array(ds_flat, copy=True),
                "dz_flat": np.array(dz_flat, copy=True),
                "P_values_flat": np.array(P_values_flat, copy=True),
                "A_values_flat": np.array(A_values_flat, copy=True),
                "q_flat": np.array(q_flat, copy=True),
                "b_flat": np.array(b_flat, copy=True),
                "x_flat": np.array(x_flat, copy=True),
                "z_flat": np.array(z_flat, copy=True),
                "s_flat": np.array(s_flat, copy=True),
                "backward_state": backward_state,
                "shared": shared,
            }
            return {
                "dP_values": np.array([10.0, 20.0]),
                "dA_values": np.array([30.0, 40.0]),
                "dq": np.array([50.0, 60.0]),
                "db": np.array([70.0]),
            }

    fake_self = type("FakeActiveSetSelf", (), {})()
    fake_self._enable_grad = True
    fake_self._n = 2
    fake_self._m = 1
    fake_self._nnz_P = 2
    fake_self._nnz_A = 2
    fake_self._solver = FakeBackendSolver()

    fake_backward_state = object()
    result = cpu_mod.ActiveSetSolver.backward_with_data_flat(
        fake_self,
        dx_flat=np.array([1.0, 2.0]),
        ds_flat=np.array([3.0]),
        dz_flat=np.array([4.0]),
        P_values_flat=np.array([5.0, 6.0]),
        A_values_flat=np.array([7.0, 8.0]),
        q_flat=np.array([9.0, 10.0]),
        b_flat=np.array([11.0]),
        x_flat=np.array([12.0, 13.0]),
        z_flat=np.array([14.0]),
        s_flat=np.array([15.0]),
        backward_state=fake_backward_state,
        batch_size=1,
    )

    np.testing.assert_allclose(calls["args"]["dx_flat"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(calls["args"]["ds_flat"], np.array([3.0]))
    np.testing.assert_allclose(calls["args"]["dz_flat"], np.array([4.0]))
    np.testing.assert_allclose(calls["args"]["P_values_flat"], np.array([5.0, 6.0]))
    np.testing.assert_allclose(calls["args"]["A_values_flat"], np.array([7.0, 8.0]))
    np.testing.assert_allclose(calls["args"]["q_flat"], np.array([9.0, 10.0]))
    np.testing.assert_allclose(calls["args"]["b_flat"], np.array([11.0]))
    np.testing.assert_allclose(calls["args"]["x_flat"], np.array([12.0, 13.0]))
    np.testing.assert_allclose(calls["args"]["z_flat"], np.array([14.0]))
    np.testing.assert_allclose(calls["args"]["s_flat"], np.array([15.0]))
    assert calls["args"]["backward_state"] is fake_backward_state
    assert calls["args"]["shared"] is True
    np.testing.assert_allclose(result["dP_values"], np.array([[10.0, 20.0]]))
    np.testing.assert_allclose(result["dA_values"], np.array([[30.0, 40.0]]))
    np.testing.assert_allclose(result["dq"], np.array([[50.0, 60.0]]))
    np.testing.assert_allclose(result["db"], np.array([[70.0]]))


def test_active_set_backward_with_data_uses_saved_forward_state():
    cones = SimpleNamespace(num_zero_cones=0, num_nonneg_cones=1)
    solver = cpu_mod.ActiveSetSolver(
        n=1,
        m=1,
        P_row_offsets=np.array([0, 1], dtype=np.int64),
        P_col_indices=np.array([0], dtype=np.int64),
        A_row_offsets=np.array([0, 1], dtype=np.int64),
        A_col_indices=np.array([0], dtype=np.int64),
        cones=cones,
        enable_grad=True,
    )

    P_values = np.array([2.0], dtype=np.float64)
    A_values = np.array([1.0], dtype=np.float64)
    solver.setup(P_values, A_values)

    q1 = np.array([-1.0], dtype=np.float64)
    b1 = np.array([0.3], dtype=np.float64)
    result1 = solver.solve(q1, b1)
    state1 = solver._last_backward_state
    direct = solver.backward(np.array([1.0], dtype=np.float64))

    q2 = np.array([-0.1], dtype=np.float64)
    b2 = np.array([1.0], dtype=np.float64)
    solver.solve(q2, b2)

    explicit = solver.backward_with_data_flat(
        dx_flat=np.array([1.0], dtype=np.float64),
        ds_flat=np.array([0.0], dtype=np.float64),
        dz_flat=np.array([0.0], dtype=np.float64),
        P_values_flat=P_values,
        A_values_flat=A_values,
        q_flat=q1,
        b_flat=b1,
        x_flat=np.asarray(result1["x"], dtype=np.float64),
        z_flat=np.asarray(result1["z"], dtype=np.float64),
        s_flat=np.asarray(result1["s"], dtype=np.float64),
        backward_state=state1,
        batch_size=1,
    )

    np.testing.assert_allclose(
        np.asarray(explicit["dP_values"]).reshape(-1), np.asarray(direct["dP_values"]).reshape(-1)
    )
    np.testing.assert_allclose(
        np.asarray(explicit["dA_values"]).reshape(-1), np.asarray(direct["dA_values"]).reshape(-1)
    )
    np.testing.assert_allclose(
        np.asarray(explicit["dq"]).reshape(-1), np.asarray(direct["dq"]).reshape(-1)
    )
    np.testing.assert_allclose(
        np.asarray(explicit["db"]).reshape(-1), np.asarray(direct["db"]).reshape(-1)
    )
