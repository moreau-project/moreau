using Test
using Moreau
import MathOptInterface as MOI
using JuMP

@testset "MOI Wrapper" begin

@testset "Solver name and version" begin
    opt = Moreau.Optimizer()
    @test MOI.get(opt, MOI.SolverName()) == "Moreau"
    @test !isempty(MOI.get(opt, MOI.SolverVersion()))
end

@testset "Empty optimizer" begin
    opt = Moreau.Optimizer()
    @test MOI.is_empty(opt)
    @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
end

@testset "Silent and TimeLimitSec" begin
    opt = Moreau.Optimizer()
    @test MOI.supports(opt, MOI.Silent())
    @test MOI.supports(opt, MOI.TimeLimitSec())
    MOI.set(opt, MOI.Silent(), true)
    @test MOI.get(opt, MOI.Silent()) == true
    MOI.set(opt, MOI.TimeLimitSec(), 10.0)
    @test MOI.get(opt, MOI.TimeLimitSec()) == 10.0
    MOI.set(opt, MOI.TimeLimitSec(), nothing)
    @test MOI.get(opt, MOI.TimeLimitSec()) === nothing
end

@testset "RawOptimizerAttribute" begin
    opt = Moreau.Optimizer()
    @test MOI.supports(opt, MOI.RawOptimizerAttribute("max_iter"))
    MOI.set(opt, MOI.RawOptimizerAttribute("max_iter"), 500)
    @test MOI.get(opt, MOI.RawOptimizerAttribute("max_iter")) == 500
end

end  # @testset "MOI Wrapper"

@testset "JuMP Integration" begin

@testset "Basic QP via JuMP" begin
    model = Model(Moreau.Optimizer)
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

@testset "LP via JuMP" begin
    model = Model(Moreau.Optimizer)
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

@testset "SOC via JuMP" begin
    model = Model(Moreau.Optimizer)
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

@testset "Settings via JuMP" begin
    model = Model(() -> Moreau.Optimizer(max_iter=100))
    set_silent(model)
    @variable(model, x >= 0)
    @objective(model, Min, x)
    optimize!(model)
    @test termination_status(model) == OPTIMAL
end

end  # @testset "JuMP Integration"
