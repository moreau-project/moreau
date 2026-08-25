using Moreau
using Test
using SparseArrays
import MathOptInterface as MOI
using JuMP

@testset "CUDA backend" begin

if !Moreau.cuda_available()
    @info "Moreau CUDA library not available — skipping CUDA tests"
else

@testset "Basic QP (CUDA)" begin
    # minimize x² + y² + 2x + y
    # s.t. x + y = 1 (zero cone)
    #      x ≤ 0.7, y ≤ 0.7 (nonneg cone)
    n, m = 2, 3
    P = sparse([1, 2], [1, 2], [2.0, 2.0], n, n)
    A = sparse([1, 1, 2, 3], [1, 2, 1, 2], [1.0, 1.0, 1.0, 1.0], m, n)
    q = [2.0, 1.0]
    b = [1.0, 0.7, 0.7]

    sol = moreau_solve(n, m, P, A, q, b;
        device=:cuda,
        zero_cones=1, nonneg_cones=2,
    )
    @test sol.status == Moreau.MOREAU_STATUS_SOLVED
    @test sol.x ≈ [0.3, 0.7] atol=1e-6
end

@testset "SOC cone (CUDA)" begin
    # minimize x₁ s.t. ‖(x₂, x₃)‖ ≤ x₁, x₂ = 1, x₃ = 1
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
        device=:cuda,
        zero_cones=2, soc_dims=[3],
    )
    @test sol.status == Moreau.MOREAU_STATUS_SOLVED
    @test sol.x[1] ≈ sqrt(2.0) atol=1e-6
end

@testset "CPU/CUDA agreement" begin
    # Same problem solved on both devices should give matching results
    n, m = 2, 3
    P = sparse([1, 2], [1, 2], [2.0, 2.0], n, n)
    A = sparse([1, 1, 2, 3], [1, 2, 1, 2], [1.0, 1.0, 1.0, 1.0], m, n)
    q = [2.0, 1.0]
    b = [1.0, 0.7, 0.7]

    sol_cpu = moreau_solve(n, m, P, A, q, b;
        device=:cpu,
        zero_cones=1, nonneg_cones=2,
    )
    sol_cuda = moreau_solve(n, m, P, A, q, b;
        device=:cuda,
        zero_cones=1, nonneg_cones=2,
    )

    @test sol_cpu.status == sol_cuda.status
    @test sol_cpu.x ≈ sol_cuda.x atol=1e-6
    @test sol_cpu.z ≈ sol_cuda.z atol=1e-6
    @test sol_cpu.s ≈ sol_cuda.s atol=1e-6
    @test sol_cpu.obj_val ≈ sol_cuda.obj_val atol=1e-6
end

# ============================================================================
# MOI / JuMP integration tests with CUDA device
# ============================================================================

@testset "MOI CUDA device attribute" begin
    opt = Moreau.Optimizer()
    MOI.set(opt, MOI.RawOptimizerAttribute("device"), :cuda)
    @test MOI.get(opt, MOI.RawOptimizerAttribute("device")) == :cuda
end

@testset "Basic QP via JuMP (CUDA)" begin
    model = Model(() -> Moreau.Optimizer(device=:cuda))
    set_silent(model)
    @variable(model, x[1:2])
    @objective(model, Min, x[1]^2 + x[2]^2 + 2x[1] + x[2])
    @constraint(model, x[1] + x[2] == 1)
    @constraint(model, x[1] <= 0.7)
    @constraint(model, x[2] <= 0.7)
    optimize!(model)

    @test termination_status(model) == OPTIMAL
    @test isapprox(value(x[1]), 0.3, atol=1e-4)
    @test isapprox(value(x[2]), 0.7, atol=1e-4)
end

@testset "LP via JuMP (CUDA)" begin
    model = Model(() -> Moreau.Optimizer(device=:cuda))
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @objective(model, Max, x + 2y)
    @constraint(model, x + y <= 10)
    @constraint(model, x <= 6)
    optimize!(model)

    @test termination_status(model) == OPTIMAL
    @test isapprox(value(x), 0.0, atol=1e-4)
    @test isapprox(value(y), 10.0, atol=1e-4)
    @test isapprox(objective_value(model), 20.0, atol=1e-4)
end

@testset "SOC via JuMP (CUDA)" begin
    model = Model(() -> Moreau.Optimizer(device=:cuda))
    set_silent(model)
    @variable(model, x[1:3])
    @objective(model, Min, x[1])
    @constraint(model, x[2] == 1)
    @constraint(model, x[3] == 1)
    @constraint(model, [x[1]; x[2]; x[3]] in SecondOrderCone())
    optimize!(model)

    @test termination_status(model) == OPTIMAL
    @test isapprox(value(x[1]), sqrt(2), atol=1e-4)
end

@testset "JuMP CPU/CUDA agreement" begin
    function solve_qp(device)
        model = Model(() -> Moreau.Optimizer(device=device))
        set_silent(model)
        @variable(model, x[1:2])
        @objective(model, Min, x[1]^2 + x[2]^2 + 2x[1] + x[2])
        @constraint(model, x[1] + x[2] == 1)
        @constraint(model, x[1] <= 0.7)
        @constraint(model, x[2] <= 0.7)
        optimize!(model)
        return termination_status(model), value.(x), objective_value(model)
    end

    status_cpu, x_cpu, obj_cpu = solve_qp(:cpu)
    status_cuda, x_cuda, obj_cuda = solve_qp(:cuda)

    @test status_cpu == status_cuda
    @test x_cpu ≈ x_cuda atol=1e-4
    @test obj_cpu ≈ obj_cuda atol=1e-4
end

# ============================================================================
# MOI.Test standard conformance suite with CUDA backend
# ============================================================================

@testset "MOI.Test standard (CUDA)" begin
    optimizer = Moreau.Optimizer(device=:cuda)
    MOI.set(optimizer, MOI.Silent(), true)
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            optimizer,
        ),
        Float64,
    )
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            atol = 1e-4,
            rtol = 1e-4,
            optimal_status = MOI.OPTIMAL,
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.VariableBasisStatus,
                MOI.ConstraintName,
                MOI.VariableName,
                MOI.ObjectiveBound,
            ],
        );
        exclude = String[
            # PSD / SDP cones (not supported)
            "test_conic_PositiveSemidefiniteCone",
            "test_conic_HermitianPositiveSemidefiniteCone",
            "test_conic_LogDetCone",
            "test_conic_RootDetCone",
            "test_conic_GeometricMeanCone",
            # Integer / discrete (not supported)
            "test_integer",
            "test_constraint_ZeroOne",
            "test_constraint_Integer",
        ],
    )
end

end # if cuda_available

end # testset
