# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0

using ChainRulesCore
using SparseArrays
using Test

# min p*x^2/2 + q*x subject to a*x = b. This has a nonzero equality
# multiplier and exact sensitivities for both primal and dual cotangents.
function equality_solution(p, a, q, b)
    x = b ./ a
    z = -(p .* x .+ q) ./ a
    return x, z
end

function equality_loss(p, a, q, b)
    x, z = equality_solution(p, a, q, b)
    return sum(0.7 .* x .- 0.3 .* z)
end

function finite_difference(args, index; h=1e-5)
    result = similar(args[index])
    for j in eachindex(result)
        plus, minus = deepcopy(args), deepcopy(args)
        plus[index][j] += h
        minus[index][j] -= h
        result[j] = (equality_loss(plus...) - equality_loss(minus...)) / (2h)
    end
    return result
end

function solution_tangent(sol)
    return Tangent{typeof(sol)}(; x=fill(0.7, size(sol.x)), z=fill(-0.3, size(sol.z)))
end

function check_sensitivities(grads, args)
    for (gradient, i) in zip((grads.dP_values, grads.dA_values, grads.dq, grads.db), 1:4)
        @test gradient ≈ finite_difference(args, i) atol=2e-5 rtol=2e-5
    end
end

@testset "Native sensitivities and batching" begin
    @test Base.get_extension(Moreau, :MoreauChainRulesExt) !== nothing
    for batch_size in (1, 2)
        @testset "batch_size=$batch_size" begin
            solver = CompiledSolver(1, 1, sparse([2.0;;]), sparse([3.0;;]);
                batch_size, zero_cones=1, device=:cpu, enable_grad=true)
            args = batch_size == 1 ?
                ([2.0], [3.0], [-1.0], [6.0]) :
                ([2.0 4.0], [3.0 2.0], [-1.0 0.5], [6.0 -2.0])
            try
                setup!(solver, args[1], args[2])
                sol = solve!(solver, args[3], args[4])
                x, z = equality_solution(args...)
                @test sol.x ≈ x atol=1e-6
                @test sol.z ≈ z atol=1e-6
                @test sol.s ≈ zero(sol.s) atol=1e-6
                grads = backward!(solver, fill(0.7, size(sol.x));
                    dz=fill(-0.3, size(sol.z)))
                check_sensitivities(grads, args)

                sol, pullback = rrule(solve!, solver, args[3], args[4])
                _, _, dq, db = pullback(solution_tangent(sol))
                @test dq ≈ finite_difference(args, 3) atol=2e-5
                @test db ≈ finite_difference(args, 4) atol=2e-5
                # A zero or thunked cotangent is valid ChainRules input too.
                zero_grads = pullback(ZeroTangent())
                @test all(g -> g isa AbstractZero || iszero(g), zero_grads[3:end])
                thunk_grads = pullback(Thunk(() -> solution_tangent(sol)))
                @test thunk_grads[3] ≈ dq atol=2e-5

                sol, pullback = rrule(setup_and_solve!, solver, args...)
                result = pullback(solution_tangent(sol))
                @test result[1] === NoTangent()
                @test result[2] === NoTangent()
                for i in 1:4
                    @test result[i+2] ≈ finite_difference(args, i) atol=2e-5
                end
                # Repeated setup must use the new coefficients for every batch.
                changed = (args[1] .* 1.1, args[2] .* 0.9, args[3], args[4])
                sol = setup_and_solve!(solver, changed...)
                x, z = equality_solution(changed...)
                @test sol.x ≈ x atol=1e-6
                @test sol.z ≈ z atol=1e-6
            finally
                destroy!(solver)
            end
        end
    end
end

@testset "Shared matrix values sum batch pullbacks" begin
    solver = CompiledSolver(1, 1, sparse([2.0;;]), sparse([3.0;;]);
        batch_size=2, zero_cones=1, device=:cpu, enable_grad=true)
    args = ([2.0], [3.0], [-1.0 0.5], [6.0 -2.0])
    try
        sol, pullback = rrule(setup_and_solve!, solver, args...)
        result = pullback(solution_tangent(sol))
        for i in 1:4
            @test size(result[i+2]) == size(args[i])
            @test result[i+2] ≈ finite_difference(args, i) atol=2e-5
        end
        @test_throws ErrorException solve!(solver, zeros(1, 3), args[4])
        @test_throws ErrorException setup!(solver, zeros(2, 2), zeros(1, 2))
    finally
        destroy!(solver)
    end
end
