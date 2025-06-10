using LQS
using Test
using LinearAlgebra

@testset "LQS.jl" begin
    # Write your tests here.
    @testset "LQS.solve basic functionality" begin
        # Simple 2-variable QUBO: maximize x1 + x2
        Q = [0.0 0.0; 0.0 0.0]
        l = [1.0, 1.0]
        β = 0.0
        result = solve(Q, l, β; prune=false, decompose=false)
        @test result.objective ≈ 2.0
        @test all(result.x .== 1.0)
        @test all(result.optimal)

        # QUBO with negative linear term: maximize -x1 - x2
        l = [-1.0, -1.0]
        result = solve(Q, l, β; prune=false, decompose=false)
        @test result.objective ≈ 0.0
        @test all(result.x .== 0.0)
        @test all(result.optimal)
    end

    @testset "LQS.solve with quadratic terms" begin
        # QUBO: maximize x1 + x2 + 2x1x2
        Q = [0.0 1.0; 1.0 0.0]
        l = [1.0, 1.0]
        β = 0.0
        result = solve(Q, l, β; prune=false, decompose=false)
        @test result.objective ≈ 4.0
        @test all(result.x .== 1.0)
        @test all(result.optimal)

        # QUBO: maximize x1 - 2x1x2
        Q = [0.0 -1.0; -1.0 0.0]
        l = [1.0, 0.0]
        result = solve(Q, l, β; prune=false, decompose=false)
        # Best is x1=1, x2=0
        @test result.objective ≈ 1.0
        @test result.x[1] ≈ 1.0
        @test result.x[2] ≈ 0.0
        @test all(result.optimal)
    end

    @testset "LQS.solve with pruning and decomposition" begin
        # QUBO: maximize x1 (x2 is always 0)
        Q = [0.0 0.0; 0.0 -10.0]
        l = [1.0, 0.0]
        result = solve(Q, l; prune=true, decompose=true)
        @test result.x[1] ≈ 1.0
        @test result.x[2] ≈ 0.0
        @test result.objective ≈ 1.0
        @test all(result.optimal)
    end

    @testset "LQS.solve with hot_start" begin
        Q = [0.0 2.0; 2.0 0.0]
        l = [0.0, 0.0]
        hot_start = [1.0, 0.0]
        result = solve(Q, l; hot_start=hot_start, prune=false, decompose=false)
        # Best is x1=1, x2=1
        @test result.objective ≈ 4.0
        @test all(result.x .== 1.0)
    end

    @testset "LQS.utform and compute_objective" begin
        Q = [1.0 2.0; 2.0 3.0]
        l = [0.5, -0.5]
        Q̄, d = LQS.utform(Q, l)
        # Q̄ should be [0 4; 4 0], d should be [1.5, 2.5]
        @test Q̄[1, 2] ≈ 4.0
        @test Q̄[2, 1] ≈ 4.0
        @test Q̄[1, 1] ≈ 0.0
        @test Q̄[2, 2] ≈ 0.0
        @test d[1] ≈ 1.5
        @test d[2] ≈ 2.5

        x = [1.0, 0.0]
        obj = LQS.compute_objective(Q̄, d, x)
        # Should match original QUBO objective
        @test obj ≈ x' * Q * x + l ⋅ x + 0.0
    end

    @testset "LQS.solve_exact correctness" begin
        Q = [0.0 1.0; 1.0 0.0]
        d = [1.0, 1.0]
        result = LQS.solve_exact(Q, d)
        # Best is x1=1, x2=1, objective = 1+1+2=4
        @test result.objective ≈ 3.0
        @test all(result.x .== 1.0)
    end

    @testset "LQS.separate_components and explore" begin
        Q̄ = [0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
        comps = LQS.separate_components(Q̄)
        @test length(comps) == 2
        @test sort(collect(comps[1])) == [1, 2]
        @test sort(collect(comps[2])) == [3]
    end

    @testset "LQS.prune! and unprune!" begin
        Q̄ = [0.0 0.0; 0.0 0.0]
        d = [1.0, -10.0]
        pruned_indices = []
        Q̄2 = LQS.prune!(pruned_indices, d, Q̄)
        # d should now have length 1 (second variable pruned)
        @test length(d) == 0
        # Unprune a solution
        sol = (objective=0, x=[], optimal=[])
        unpruned = LQS.unprune!(sol, pruned_indices)
        @test length(unpruned.x) == 2
        @test unpruned.x[1] ≈ 1.0
        @test unpruned.x[2] ≈ 0.0
    end

    @testset "LQS.filterx and prunex!" begin
        x = [1.0, 0.0, 1.0]
        indices = [1, 3]
        @test LQS.filterx(x, indices) == [1.0, 1.0]
        pruned_indices = [(is=[2], xᵢ=0.0, objΔ=0.0)]
        x2 = [1.0, 1.0]
        @test LQS.prunex!(x2, pruned_indices) == [1.0]
    end

    @testset "LQS.solve with hot_start and pruning" begin
        Q = [0.0 1.0; 1.0 0.0]
        l = [1.0, 1.0]
        hot_start = [1.0, 0.0]
        result = solve(Q, l; hot_start=hot_start, prune=true, decompose=false)
        @test result.objective ≈ 4.0
        @test all(result.x .== 1.0)
    end
end
