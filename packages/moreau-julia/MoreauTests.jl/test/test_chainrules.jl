using Test
using Moreau
using ChainRulesCore
using SparseArrays

@testset "ChainRules" begin

# Shared problem: simple QP
n_cr, m_cr = 2, 3
P_cr = sparse([1, 2], [1, 2], [1.0, 1.0], n_cr, n_cr)
A_cr = sparse([1, 1, 2, 3], [1, 2, 1, 2], [1.0, 1.0, 1.0, 1.0], m_cr, n_cr)

@testset "rrule for solve! returns valid gradients" begin
    solver = CompiledSolver(n_cr, m_cr, P_cr, A_cr;
        zero_cones=1, nonneg_cones=2, verbose=false, enable_grad=true)
    setup!(solver, [1.0, 1.0], [1.0, 1.0, 1.0, 1.0])

    q = [2.0, 1.0]
    b = [1.0, 0.7, 0.7]

    sol, pullback = rrule(solve!, solver, q, b)
    @test sol.status == Moreau.MOREAU_STATUS_SOLVED

    # Construct a tangent for the solution (gradient w.r.t. x only)
    Δ = Tangent{Moreau.Solution{Vector{Float64}}}(
        x = ones(n_cr),
        z = zeros(m_cr),
        s = zeros(m_cr),
    )
    no_f, no_solver, dq, db = pullback(Δ)
    @test no_f === NoTangent()
    @test no_solver === NoTangent()
    @test length(dq) == n_cr
    @test length(db) == m_cr
    @test !any(isnan, dq)
    @test !any(isnan, db)

    destroy!(solver)
end

@testset "rrule for setup_and_solve! returns valid gradients" begin
    solver = CompiledSolver(n_cr, m_cr, P_cr, A_cr;
        zero_cones=1, nonneg_cones=2, verbose=false, enable_grad=true)

    P_vals = [1.0, 1.0]
    A_vals = [1.0, 1.0, 1.0, 1.0]
    q = [2.0, 1.0]
    b = [1.0, 0.7, 0.7]

    sol, pullback = rrule(setup_and_solve!, solver, P_vals, A_vals, q, b)
    @test sol.status == Moreau.MOREAU_STATUS_SOLVED

    Δ = Tangent{Moreau.Solution{Vector{Float64}}}(
        x = ones(n_cr),
        z = zeros(m_cr),
        s = zeros(m_cr),
    )
    no_f, no_solver, dP, dA, dq, db = pullback(Δ)
    @test no_f === NoTangent()
    @test no_solver === NoTangent()
    @test length(dP) == length(P_vals)
    @test length(dA) == length(A_vals)
    @test length(dq) == n_cr
    @test length(db) == m_cr
    @test !any(isnan, dP)
    @test !any(isnan, dA)
    @test !any(isnan, dq)
    @test !any(isnan, db)

    destroy!(solver)
end

@testset "Gradients are nonzero for nontrivial problem" begin
    solver = CompiledSolver(n_cr, m_cr, P_cr, A_cr;
        zero_cones=1, nonneg_cones=2, verbose=false, enable_grad=true)
    setup!(solver, [1.0, 1.0], [1.0, 1.0, 1.0, 1.0])

    q = [2.0, 1.0]
    b = [1.0, 0.7, 0.7]

    sol, pullback = rrule(solve!, solver, q, b)
    Δ = Tangent{Moreau.Solution{Vector{Float64}}}(
        x = ones(n_cr),
        z = zeros(m_cr),
        s = zeros(m_cr),
    )
    _, _, dq, db = pullback(Δ)

    # At least one gradient component should be nonzero
    @test any(x -> abs(x) > 1e-10, dq)
    @test any(x -> abs(x) > 1e-10, db)

    destroy!(solver)
end

end  # @testset "ChainRules"
