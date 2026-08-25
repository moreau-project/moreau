using Test
using Moreau
using SparseArrays

@testset "CompiledSolver" begin

# Shared problem data: simple QP
# minimize (1/2)x'Px + q'x s.t. Ax + s = b, s ∈ K
# P = I₂, q = [2, 1], A = [[1,1],[1,0],[0,1]], b = [1, 0.7, 0.7]
# K = {0}¹ × R₊²
n_qp, m_qp = 2, 3
P_qp = sparse([1, 2], [1, 2], [1.0, 1.0], n_qp, n_qp)
A_qp = sparse([1, 1, 2, 3], [1, 2, 1, 2], [1.0, 1.0, 1.0, 1.0], m_qp, n_qp)

@testset "Basic setup/solve/destroy cycle" begin
    solver = CompiledSolver(n_qp, m_qp, P_qp, A_qp;
        zero_cones=1, nonneg_cones=2, verbose=false)
    setup!(solver, [1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
    sol = solve!(solver, [2.0, 1.0], [1.0, 0.7, 0.7])
    @test sol.status == Moreau.MOREAU_STATUS_SOLVED
    @test isapprox(sol.x[1], 0.3, atol=1e-4)
    @test isapprox(sol.x[2], 0.7, atol=1e-4)
    destroy!(solver)
end

@testset "Re-solve with different q/b" begin
    solver = CompiledSolver(n_qp, m_qp, P_qp, A_qp;
        zero_cones=1, nonneg_cones=2, verbose=false)
    setup!(solver, [1.0, 1.0], [1.0, 1.0, 1.0, 1.0])

    sol1 = solve!(solver, [2.0, 1.0], [1.0, 0.7, 0.7])
    @test sol1.status == Moreau.MOREAU_STATUS_SOLVED

    # Different q
    sol2 = solve!(solver, [1.0, 2.0], [1.0, 0.7, 0.7])
    @test sol2.status == Moreau.MOREAU_STATUS_SOLVED
    # With q=[1,2], optimal shifts toward x1
    @test sol2.x[1] > sol1.x[1]

    destroy!(solver)
end

@testset "Backward pass (CPU)" begin
    solver = CompiledSolver(n_qp, m_qp, P_qp, A_qp;
        zero_cones=1, nonneg_cones=2, verbose=false, enable_grad=true)
    setup!(solver, [1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
    sol = solve!(solver, [2.0, 1.0], [1.0, 0.7, 0.7])
    @test sol.status == Moreau.MOREAU_STATUS_SOLVED

    grads = backward!(solver, ones(n_qp))
    @test length(grads.dq) == n_qp
    @test length(grads.db) == m_qp
    @test !any(isnan, grads.dq)
    @test !any(isnan, grads.db)
    @test !any(isnan, grads.dP_values)
    @test !any(isnan, grads.dA_values)

    destroy!(solver)
end

@testset "Destroy is idempotent" begin
    solver = CompiledSolver(n_qp, m_qp, P_qp, A_qp;
        zero_cones=1, nonneg_cones=2, verbose=false)
    destroy!(solver)
    destroy!(solver)  # should not error
end

@testset "Error on use after destroy" begin
    solver = CompiledSolver(n_qp, m_qp, P_qp, A_qp;
        zero_cones=1, nonneg_cones=2, verbose=false)
    destroy!(solver)
    @test_throws ErrorException setup!(solver, [1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
end

end  # @testset "CompiledSolver"

@testset "Warm Start" begin

@testset "Warm start reduces iterations" begin
    n, m = 2, 3
    P = sparse([1, 2], [1, 2], [1.0, 1.0], n, n)
    A = sparse([1, 1, 2, 3], [1, 2, 1, 2], [1.0, 1.0, 1.0, 1.0], m, n)

    solver = CompiledSolver(n, m, P, A;
        zero_cones=1, nonneg_cones=2, verbose=false)
    setup!(solver, [1.0, 1.0], [1.0, 1.0, 1.0, 1.0])

    # Cold solve
    sol1 = solve!(solver, [2.0, 1.0], [1.0, 0.7, 0.7])
    @test sol1.status == Moreau.MOREAU_STATUS_SOLVED

    # Warm solve with nearby problem
    sol2 = solve!(solver, [2.1, 1.0], [1.0, 0.7, 0.7]; warm_start=sol1)
    @test sol2.status == Moreau.MOREAU_STATUS_SOLVED
    # Warm start should take fewer or equal iterations
    @test sol2.iterations <= sol1.iterations

    destroy!(solver)
end

end  # @testset "Warm Start"

@testset "Exponential Cone" begin

@testset "Exp cone via moreau_solve" begin
    # Maximize x₁ subject to (x₁, x₂, x₃) ∈ ExpCone, x₂ = 1, x₃ = 1
    # ExpCone: x₂ * exp(x₁/x₂) <= x₃, x₂ > 0
    # With x₂ = 1, x₃ = 1: exp(x₁) <= 1 → x₁ <= 0
    # Maximize x₁ (minimize -x₁) → x₁ = 0
    n, m = 3, 5
    P = spzeros(n, n)
    q = [-1.0, 0.0, 0.0]
    # Rows 1-2: zero cones (x₂ = 1, x₃ = 1)
    # Rows 3-5: exp cone
    A = sparse(
        [1, 2, 3, 4, 5],
        [2, 3, 1, 2, 3],
        [1.0, 1.0, -1.0, -1.0, -1.0],
        m, n,
    )
    b = [1.0, 1.0, 0.0, 0.0, 0.0]

    sol = moreau_solve(n, m, P, A, q, b;
        zero_cones=2, exp_cones=1, verbose=false)

    @test sol.status == Moreau.MOREAU_STATUS_SOLVED
    @test isapprox(sol.x[1], 0.0, atol=1e-3)
    @test isapprox(sol.x[2], 1.0, atol=1e-3)
    @test isapprox(sol.x[3], 1.0, atol=1e-3)
end

end  # @testset "Exponential Cone"

@testset "Power Cone" begin

@testset "Power cone via moreau_solve" begin
    # Power cone: |x₃| <= x₁^α * x₂^(1-α), x₁,x₂ >= 0
    # With α = 0.5: |x₃| <= sqrt(x₁ * x₂)
    # Set x₁ = 1, x₂ = 1, minimize -x₃ → x₃ = 1
    n, m = 3, 5
    P = spzeros(n, n)
    q = [0.0, 0.0, -1.0]
    # Rows 1-2: zero cones (x₁ = 1, x₂ = 1)
    # Rows 3-5: power cone (α=0.5)
    A = sparse(
        [1, 2, 3, 4, 5],
        [1, 2, 1, 2, 3],
        [1.0, 1.0, -1.0, -1.0, -1.0],
        m, n,
    )
    b = [1.0, 1.0, 0.0, 0.0, 0.0]

    sol = moreau_solve(n, m, P, A, q, b;
        zero_cones=2, power_alphas=[0.5], verbose=false)

    @test sol.status == Moreau.MOREAU_STATUS_SOLVED
    @test isapprox(sol.x[3], 1.0, atol=1e-3)
end

end  # @testset "Power Cone"
