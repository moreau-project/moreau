using Test
using Moreau
using SparseArrays

@testset "C Wrapper" begin

@testset "Version" begin
    ver = Moreau.c_moreau_version()
    @test !isempty(ver)
    @test occursin(".", ver)  # Should be a version string like "0.1.5"
end

@testset "Basic QP" begin
    # Problem:
    #   minimize  (1/2)x'Px + q'x
    #   s.t.      Ax + s = b, s ∈ K
    #
    # P = I₂, q = [2, 1]
    # A = [[1, 1], [1, 0], [0, 1]]
    # b = [1, 0.7, 0.7]
    # K = {0}¹ × R₊²  (1 zero cone + 2 nonneg cones)
    # Expected: x ≈ [0.3, 0.7]

    n, m = 2, 3
    P = sparse([1, 2], [1, 2], [1.0, 1.0], n, n)
    A = sparse([1, 1, 2, 3], [1, 2, 1, 2], [1.0, 1.0, 1.0, 1.0], m, n)
    q = [2.0, 1.0]
    b = [1.0, 0.7, 0.7]

    sol = moreau_solve(n, m, P, A, q, b;
        zero_cones=1, nonneg_cones=2, verbose=false)

    @test sol.status == Moreau.MOREAU_STATUS_SOLVED
    @test isapprox(sol.x[1], 0.3, atol=1e-4)
    @test isapprox(sol.x[2], 0.7, atol=1e-4)
    @test sol.iterations > 0
    @test sol.solve_time >= 0.0
end

@testset "Error handling: solve before setup" begin
    P_ro = Int64[0, 1, 2]
    P_ci = Int64[0, 1]
    A_ro = Int64[0, 2, 3, 4]
    A_ci = Int64[0, 1, 0, 1]

    cones = Moreau.CMoreauCones(
        Int64(1), Int64(2), Int64(0), C_NULL,
        Int64(0), Int64(0), C_NULL,
    )

    settings = Moreau.c_moreau_settings_default()

    # Rebuild with verbose=0
    settings = Moreau.CMoreauSettings(
        settings.batch_size, settings.max_iter, settings.time_limit,
        Int32(0), settings.enable_grad, settings.ipm,
    )

    solver = Moreau.c_moreau_solver_create(
        Int64(2), Int64(3),
        P_ro, P_ci, Int64(2),
        A_ro, A_ci, Int64(4),
        cones, settings,
    )

    @test_throws ErrorException Moreau.c_moreau_solver_solve(solver, [1.0, 1.0], [1.0, 1.0, 1.0])

    Moreau.c_moreau_solver_destroy(solver)
end

@testset "SOC cone" begin
    # Minimize x₁ subject to ||(x₂, x₃)|| <= x₁, x₂ = 1, x₃ = 1
    # Formulation: Ax + s = b, s ∈ K
    #   s = b - Ax, so for SOC rows we need -Ax ∈ SOC (with b=0)
    #   i.e., (-A)x must be in SOC, so negate A for SOC rows.
    # Constraints:
    #   row 1: x₂ + s₁ = 1, s₁ = 0  (zero cone: x₂ = 1)
    #   row 2: x₃ + s₂ = 1, s₂ = 0  (zero cone: x₃ = 1)
    #   rows 3-5: -[x₁, x₂, x₃] + s = 0  → s = [x₁, x₂, x₃] ∈ SOC(3)
    n, m = 3, 5
    P = spzeros(n, n)
    A = sparse(
        [1, 2, 3, 4, 5],
        [2, 3, 1, 2, 3],
        [1.0, 1.0, -1.0, -1.0, -1.0],
        m, n,
    )
    q = [1.0, 0.0, 0.0]
    b = [1.0, 1.0, 0.0, 0.0, 0.0]

    sol = moreau_solve(n, m, P, A, q, b;
        zero_cones=2, soc_dims=[3], verbose=false)

    @test sol.status == Moreau.MOREAU_STATUS_SOLVED
    @test isapprox(sol.x[1], sqrt(2), atol=1e-4)
    @test isapprox(sol.x[2], 1.0, atol=1e-4)
    @test isapprox(sol.x[3], 1.0, atol=1e-4)
end

@testset "Get dims" begin
    n, m = 2, 3
    P = sparse([1, 2], [1, 2], [1.0, 1.0], n, n)
    A = sparse([1, 1, 2, 3], [1, 2, 1, 2], [1.0, 1.0, 1.0, 1.0], m, n)

    P_ro, P_ci, P_vals = Moreau._csc_to_csr(P)
    A_ro, A_ci, A_vals = Moreau._csc_to_csr(A)

    cones = Moreau.CMoreauCones(
        Int64(1), Int64(2), Int64(0), C_NULL,
        Int64(0), Int64(0), C_NULL,
    )
    settings = Moreau.c_moreau_settings_default()
    settings = Moreau.CMoreauSettings(
        settings.batch_size, settings.max_iter, settings.time_limit,
        Int32(0), settings.enable_grad, settings.ipm,
    )

    solver = Moreau.c_moreau_solver_create(
        Int64(n), Int64(m),
        P_ro, P_ci, Int64(length(P_vals)),
        A_ro, A_ci, Int64(length(A_vals)),
        cones, settings,
    )

    qn, qm, qbs, qnnzP, qnnzA = Moreau.c_moreau_solver_get_dims(solver)
    @test qn == n
    @test qm == m
    @test qbs == 1
    @test qnnzP == length(P_vals)
    @test qnnzA == length(A_vals)

    Moreau.c_moreau_solver_destroy(solver)
end

end  # @testset "C Wrapper"
