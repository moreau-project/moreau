module TestMoreauMOI

using Test
import MathOptInterface as MOI
import Moreau

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_runtests()
    optimizer = Moreau.Optimizer()
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
    return
end

function test_SolverName()
    @test MOI.get(Moreau.Optimizer(), MOI.SolverName()) == "Moreau"
    return
end

end # module TestMoreauMOI

TestMoreauMOI.runtests()
