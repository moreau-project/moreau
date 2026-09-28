"""
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
"""

from dataclasses import dataclass

import moreau
import numpy as np
from scipy import sparse

# Canonical slack order, with multiple sizes for the dimension-dependent paths.
CONE_CASES = ("nonneg", "soc3", "soc6", "psd2", "psd3", "exp", "power", "gen_power")


def svec(matrix):
    return np.array(
        [
            matrix[i, j] * (1 if i == j else np.sqrt(2))
            for j in range(len(matrix))
            for i in range(j + 1)
        ]
    )


def smat(vector, k):
    matrix = np.zeros((k, k))
    offset = 0
    for j in range(k):
        for i in range(j + 1):
            matrix[i, j] = matrix[j, i] = vector[offset] / (1 if i == j else np.sqrt(2))
            offset += 1
    return matrix


def complementary_pair(case):
    """Analytic complementary points, independent of solver outputs/projections."""
    if case == "nonneg":
        x, z, params = np.array([0.0, 1.3, 0.0]), np.array([0.7, 0.0, 1.1]), {}
        kind = case
    elif case.startswith("soc"):
        tail = np.linspace(-0.7, 0.9, int(case[3:]) - 1)
        norm = np.linalg.norm(tail)
        x, z = np.r_[norm, tail], np.r_[1.0, -tail / norm]
        kind, params = "soc", {}
    elif case.startswith("psd"):
        k = int(case[3:])
        Q, _ = np.linalg.qr(np.random.default_rng(k).normal(size=(k, k)))
        x = svec((Q[:, 1:] * np.arange(1, k)) @ Q[:, 1:].T)
        z = svec(np.outer(Q[:, 0], Q[:, 0]))
        kind, params = "psd_triangle", {"psd_k": k}
    elif case == "exp":
        t, y = 0.2, 1.3
        x = np.array([t * y, y, y * np.exp(t)])
        z = np.array([-np.exp(t), np.exp(t) * (t - 1), 1.0])
        kind, params = case, {}
    else:
        alpha = np.array([0.3, 0.7]) if case == "power" else np.array([0.2, 0.3, 0.5])
        head = np.linspace(0.8, 1.4, len(alpha))
        radius = np.prod(head**alpha)
        direction = np.array([1.0]) if case == "power" else np.array([0.6, -0.8])
        x = np.r_[head, radius * direction]
        z = np.r_[alpha * radius / head, -direction]
        kind = case
        params = {"alpha": 0.3} if case == "power" else {"alphas": alpha.tolist(), "dim2": 2}
    spec = moreau.DirectConeSpec(kind=kind, indices=list(range(len(x))), **params)
    return spec, x, z


def assert_member(spec, vector, *, dual=False, tol=2e-5):
    """Check cone inequalities directly, including the asymmetric dual cones."""
    v = np.asarray(vector) / max(1.0, np.linalg.norm(vector))
    assert np.isfinite(v).all()
    if spec.kind == "nonneg":
        assert v.min() >= -tol
    elif spec.kind == "soc":
        assert v[0] >= np.linalg.norm(v[1:]) - tol
    elif spec.kind == "psd_triangle":
        assert np.linalg.eigvalsh(smat(v, spec.psd_k)).min() >= -tol
    elif spec.kind == "exp":
        if dual:
            v = np.array([-v[1], -v[0], np.e * v[2]])
        assert v[1] >= -tol and v[2] >= -tol
        if v[1] > tol:
            assert v[0] <= v[1] * np.log(max(v[2], tol) / v[1]) + tol
        else:
            assert v[0] <= tol
    else:
        alpha = (
            np.array([spec.alpha, 1 - spec.alpha])
            if spec.kind == "power"
            else np.array(spec.alphas)
        )
        head = v[: len(alpha)]
        assert head.min() >= -tol
        if dual:
            head = head / alpha
        radius = np.prod(np.maximum(head, 0) ** alpha)
        assert np.linalg.norm(v[len(alpha) :]) <= radius + tol


def slack_cones(specs, *, zero=0, direct=()):
    return moreau.Cones(
        num_zero_cones=zero,
        num_nonneg_cones=sum(len(c.indices) for c in specs if c.kind == "nonneg"),
        so_cone_dims=[len(c.indices) for c in specs if c.kind == "soc"],
        psd_dims=[c.psd_k for c in specs if c.kind == "psd_triangle"],
        num_exp_cones=sum(c.kind == "exp" for c in specs),
        power_alphas=[c.alpha for c in specs if c.kind == "power"],
        gen_power_cone_params=[(c.alphas, c.dim2) for c in specs if c.kind == "gen_power"],
        dir_cones=list(direct),
    )


@dataclass
class KKTProblem:
    P: object
    q: np.ndarray
    A: object
    b: np.ndarray
    cones: moreau.Cones
    slack_specs: list
    optimum: moreau.WarmStart

    def solver(self, device, equilibrate=True, ipm_options=None, **kwargs):
        settings = moreau.Settings(
            device=device,
            solver="ipm",
            verbose=False,
            ipm_settings=moreau.IPMSettings(
                equilibrate_enable=equilibrate,
                chordal_decomposition_enable=False,
                **(ipm_options or {}),
            ),
            **kwargs,
        )
        return moreau.Solver(self.P, self.q, self.A, self.b, cones=self.cones, settings=settings)

    def check(self, sol, tol=2e-5):
        indices = np.concatenate([c.indices for c in self.cones.dir_cones])
        residual = self.P @ sol.x + self.q + self.A.T @ sol.z
        residual[indices] -= sol.z_x
        scale = max(1.0, np.linalg.norm(self.q, np.inf), np.linalg.norm(self.b, np.inf))
        np.testing.assert_allclose(residual / scale, 0, atol=tol)
        np.testing.assert_allclose((self.A @ sol.x + sol.s - self.b) / scale, 0, atol=tol)
        np.testing.assert_allclose(sol.s[:1], 0, atol=tol)
        for specs, primal, dual, offset in (
            (self.cones.dir_cones, sol.x[indices], sol.z_x, 0),
            (self.slack_specs, sol.s, sol.z, 1),
        ):
            for spec in specs:
                end = offset + len(spec.indices)
                u, v = primal[offset:end], dual[offset:end]
                assert_member(spec, u, tol=tol)
                assert_member(spec, v, dual=True, tol=tol)
                assert abs(u @ v) <= tol * max(1.0, np.linalg.norm(u) * np.linalg.norm(v))
                offset = end
        np.testing.assert_allclose(sol.x, self.optimum.x, atol=3e-4, rtol=3e-4)
        np.testing.assert_allclose(sol.z_x, self.optimum.z_x, atol=3e-4, rtol=3e-4)
        np.testing.assert_allclose(sol.z, self.optimum.z, atol=3e-4, rtol=3e-4)


def planted_problem(cases=CONE_CASES, seed=17):
    pairs = [complementary_pair(case) for case in cases]
    size = sum(len(x) for _, x, _ in pairs)
    n, m = 2 * size + 2, size + 1
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    indices, free = permutation[:size], permutation[size:]
    x = rng.uniform(-0.3, 0.3, n)
    direct, slack, zx, s, z = [], [], [], [np.array([0.0])], [np.array([0.7])]
    offset = 0
    for i, (spec, u, v) in enumerate(pairs):
        slots = indices[offset : offset + len(u)]
        direct.append(spec.model_copy(update={"indices": slots.tolist()}))
        slack.append(spec)
        x[slots] = u * (1 + i / 8)
        zx.append(v * (0.7 + i / 9))
        s.append(u * (0.8 + i / 7))
        z.append(v * (1.2 + i / 6))
        offset += len(u)
    s, z, zx = np.concatenate(s), np.concatenate(z), np.concatenate(zx)
    H = rng.normal(size=(n, n)) / np.sqrt(n)
    P = sparse.csr_matrix(np.diag(np.geomspace(0.5, 12.0, n)) + 0.2 * H.T @ H)
    A = rng.normal(scale=0.08, size=(m, n))
    A[:, free[:m]] += np.diag(np.geomspace(0.4, 3.0, m))
    A = sparse.csr_matrix(A)
    q = -P @ x - A.T @ z
    q[indices] += zx
    return KKTProblem(
        P,
        q,
        A,
        A @ x + s,
        slack_cones(slack, zero=1, direct=direct),
        slack,
        moreau.WarmStart(x=x, s=s, z=z, z_x=zx),
    )
