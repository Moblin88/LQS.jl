module LQS

export solve

using LinearAlgebra, Base.Threads, Random, Base.Checked
import Base.Threads: @spawn

"""
    solve(Q::AbstractMatrix{<:Real},
    𝓁::AbstractVector{<:Real}=zeros(eltype(Q), LinearAlgebra.checksquare(Q)),
    β::Real=zero(eltype(Q));
    hot_start=nothing,
    max_stagnation=size(Q, 1),
    max_candidates=10 * size(Q, 1),
    exact_threshold=min(1.5 * max_stagnation, max_candidates),
    ntasks=Threads.threadpoolsize(),
    njumps=3,
    prune=true,
    decompose=true)

Solve a Quadratic Unconstrained Binary Optimization (QUBO) problem defined by the matrix `Q`, the linear term `𝓁`, and the constant term `β`.
"""
function solve(Q::AbstractMatrix{<:Real},
    𝓁::AbstractVector{<:Real}=zeros(eltype(Q), LinearAlgebra.checksquare(Q)),
    β::Real=zero(eltype(Q));
    hot_start=nothing,
    max_stagnation=size(Q, 1),
    max_candidates=10 * size(Q, 1),
    exact_threshold=min(1.5 * max_stagnation, max_candidates),
    ntasks=Threads.threadpoolsize(),
    njumps=3,
    prune=true,
    decompose=true)
    T = promote_type(eltype(Q), eltype(𝓁)) # Determine the type for the solution vector
    Q̄, d = utform(Q, 𝓁) # separate Q into Q̄ = Q + Q' - 2 * Diagonal(Q) and d = diag(Q)
    @info "solving QUBO problem with $(length(d)) variables" # Log the problem being solved
    # prune Q̄ and d to remove elements that are always 0 or 1
    pruned_indices = @NamedTuple{is::Vector{Int}, xᵢ::T, objΔ::T}[] # Vector to hold the indices of the pruned elements
    if prune
        Q̄ = prune!(pruned_indices, d, Q̄)
        @info "After pruning, problem has $(length(d)) variables"
    end # Prune the elements of Q̄ and d that are always 0 or 1
    hot_start = prunex!(deepcopy(hot_start), pruned_indices) # Prune the hot start vector if it is provided
    if decompose
        indexsets = separate_components(Q̄) # Decompose Q̄ into connected components
    else
        indexsets = [BitSet(axes(Q̄, 2))] # If not decomposing, use all indices as a single component
    end
    partial_solutions = Channel{@NamedTuple{solution::@NamedTuple{objective::T, x::Vector{T}}, is::Vector{Int}, exact::Bool}}() # Channel to hold the partial solutions
    for indexset in indexsets # For each connected component
        is = collect(indexset) # Convert the BitSet to a vector of indices
        #@info "Solving connected component with indices $(is)"
        if length(is) <= log2(exact_threshold + 1) # If the number of indices is small enough, solve exactly
            @info "Solving connected component exactly with $(length(is)) variables"
            @spawn put!(partial_solutions, (solution=solve_exact(Q̄[is, is], d[is]), is=is, exact=true)) # Solve the connected component exactly
        else # Otherwise, use the local search algorithm
            @info "Solving connected component approximately with $(length(is)) variables"
            @spawn put!(partial_solutions, (solution=solve_approx(
                    Q̄[is, is],
                    d[is], (
                        hot_start=filterx(hot_start, is),
                        max_stagnation=max_stagnation,
                        max_candidates=max_candidates,
                        ntasks=ntasks,
                        njumps=njumps)), is=is, exact=false)) # Solve the connected component using local search
        end
    end
    obj = β # Initialize the objective value
    x = Vector{T}(undef, size(Q̄, 2)) # Initialize the solution vector
    optimal = BitVector(undef, size(Q̄, 2))
    try
        for _ in indexsets # For each index
            (; solution, is, exact) = take!(partial_solutions) # Wait for a partial solution from the channel
            obj += solution.objective # Update the objective value with the partial solution
            x[is] .= solution.x # Update the solution vector with the partial solution
            optimal[is] .= exact # Update the optimal vector with the partial solution
        end
    finally
        close(partial_solutions) # Close the channel to signal that we are done
    end

    return unprune!((objective=obj, x=x, optimal=optimal), pruned_indices) # unprune the final result
end

"""
    filterx(x, indices)

Filter the elements of `x` at the specified `indices`.
If `x` is `nothing`, return `nothing`.
"""
function filterx(x, indices)
    isnothing(x) && return nothing # If x is nothing, return nothing
    return x[indices] # Return the elements of x at the specified indices
end


"""
    solve_approx(Q̄, d,
    (; hot_start,
        max_stagnation,
        max_candidates,
        ntasks,
        njumps))

Solve the QUBO problem approximately using a local search algorithm.
This function uses multiple threads to produce candidates and checks them for the best solution.
It returns the best solution found after checking a maximum number of candidates or after stagnation.
`njumps`` is the number of jumps to perform in each candidate solution.
`hot_start` is an optional initial solution to start the search from.
"""
function solve_approx(Q̄, d,
    (; hot_start,
        max_stagnation,
        max_candidates,
        ntasks,
        njumps))
    T = promote_type(eltype(Q̄), eltype(d)) # Determine the type for the solution vector
    checker_task = Ref{Task}() # Create a reference to the checker task
    candidate_channel = Channel{@NamedTuple{objective::T, x::Vector{T}}}(; taskref=checker_task, spawn=true) do ch
        check_candidates!(ch, max_stagnation, max_candidates) # Task to check candidates in the channel
    end
    if !isnothing(hot_start)
        put_local_max!(candidate_channel, Q̄, d, njumps, deepcopy(hot_start))# produce the first candidate
    end
    for _ in 1:ntasks
        @spawn produce_candidates(candidate_channel, Q̄, d, njumps) # Spawn a thread to produce candidates
    end
    return fetch(checker_task[]) # Wait for the checker task to finish and unprune the solution
end


"""
    separate_components(Q̄)

Separate the matrix `Q̄` into connected components based on the non-zero elements in its columns.
This function returns a vector of BitSets, where each BitSet contains the indices of the columns that are connected to each other.
"""
function separate_components(Q̄)
    indexsets = BitSet[]
    remaining = BitSet(axes(Q̄, 2)) # Create a BitSet of the indices of the columns of Q̄
    while (!isempty(remaining)) # While there are unexplored indices
        i = first(remaining) # Get the first unexplored index
        index_set = explore(Q̄, i) # Explore the indices connected to the first unexplored index
        push!(indexsets, index_set) # Add the explored indices to the indexsets
        setdiff!(remaining, index_set) # Remove the explored indices from the remaining indices
    end
    return indexsets # Return the indexsets of the connected components
end


"""
    explore(Q̄, i)

Explore the indices connected to the index `i` in the matrix `Q̄`.
"""
function explore(Q̄, i)
    cols = eachcol(Q̄)
    explored = BitSet()
    unexplored = BitSet([i]) # Create a list of unexplored indices
    while !isempty(unexplored) # While there are unexplored indices
        i = first(unexplored) # Get the first index that has not been explored
        union!(unexplored, findall((!) ∘ iszero, cols[i])) # Find the indices of the non-zero elements in the ith column
        push!(explored, i) # Add the index to the explored list
        setdiff!(unexplored, explored) # Remove the index from the unexplored list
    end
    return explored # Return the set of explored indices
end


"""
    unprune((; objective, x, optimal), pruned_indices)

Reconstruct the original solution vector and objective value from a solution to a pruned problem.
"""
function unprune!((; objective, x, optimal), pruned_indices)
    for (; is, xᵢ, objΔ) in Iterators.reverse(pruned_indices)
        for i in is
            insert!(x, i, xᵢ) # Insert the pruned values back into the solution vector
            insert!(optimal, i, true) # Insert the pruned values back into the optimal vector
        end
        objective += objΔ # Update the objective value with the pruned values
    end
    return (objective=objective, x=x, optimal=optimal) # Return the objective value and solution vector
end


"""
    prunex!(x, pruned_indices)

Prune the elements of `x` at the specified `pruned_indices`.
If `x` is `nothing`, return `nothing`.
"""
function prunex!(x, pruned_indices)
    isnothing(x) && return nothing # If x is nothing, return nothing
    for (; is) in pruned_indices
        deleteat!(x, is) # Remove the pruned indices from the solution vector
    end
    #@info "pruned to x=$(x)"
    return x
end

"""
    prune!(pruned_indices, d, Q̄)

Prune the elements of `Q̄` and `d` that are always 0 or 1.
This function modifies `Q̄` and `d`, removing the elements that are always 0 or 1.
It also updates `pruned_indices` with the indices of the pruned elements and the objective change.
You can use [`LQS.unprune`](@ref) to restore the original solution vector and objective value produced by solving a pruned problem.
"""
function prune!(pruned_indices, d, Q̄)
    T = promote_type(eltype(Q̄), eltype(d)) # Determine the type for the solution vector
    e = ones(eltype(Q̄), axes(d))
    while (!isempty(d)) # While there are elements in d to prune
        resize!(e, length(d)) # each time through the loop the length of d may change
        # prune the indices of Q̄ and d where x is always 0
        zero_drops = muladd(max.(Q̄, zero(eltype(Q̄))), e, d) .<= zero(T)
        if any(zero_drops)
            Q̄ = drop_indices!(pruned_indices, d, Q̄, zero(T), zero_drops) # drop the zeros from Q̄ and d
            continue # continue pruning
        end
        # prune the indices of Q̄ and d where x is always 1
        one_drops = muladd(min.(Q̄, zero(eltype(Q̄))), e, d) .>= zero(T)
        if any(one_drops)
            Q̄ = drop_indices!(pruned_indices, d, Q̄, one(T), one_drops) # drop the ones from Q̄ and d
            continue # continue pruning
        end
        # if we reach here, we have no more elements to prune
        break
    end
    return Q̄ # return the pruned Q̄
end

"""
    drop_indices!(pruned_indices, d, Q̄, xᵢ, drops)

Drop the indices from `Q̄` and `d` where `xᵢ` is 0 or 1, and update `pruned_indices` with the dropped indices and the objective change.
"""
function drop_indices!(pruned_indices, d, Q̄, xᵢ, drops)
    if isone(xᵢ)
        objΔ = compute_objective(Q̄, d, drops)
        mul!(d, Q̄, drops, true, true) # update d
    elseif iszero(xᵢ)
        objΔ = zero(promote_type(eltype(d), eltype(Q̄))) # If xᵢ is 0, the objective value does not change
    else
        throw(ArgumentError("xᵢ must be 0 or 1, got $(xᵢ)")) # If xᵢ is not 0 or 1, throw an error
    end
    #@info "Pruning $(xᵢ) at indices $(findall(drops)) from Q̄ and d" # Log the elements being pruned
    push!(pruned_indices, (is=findall(drops), xᵢ=xᵢ, objΔ=objΔ)) # push the indices of the elements that are pruned
    deleteat!(d, drops) # delete the elements of d that are not needed anymore
    return Q̄[.!drops, .!drops] # delete the columns and rows of Q̄ that are not needed anymore  
end

"""
    check_candidates(candidate_channel, max_stagnation, max_candidates)

Task loop that checks the candidates in the `candidate_channel` and returns the best solution after stagnation or reaching the maximum number of candidates.
"""
function check_candidates!(candidate_channel, max_stagnation, max_candidates)
    candidates = 1
    stagnation = 0
    best = take!(candidate_channel)
    #@info "took first candidate with objective $(best.objective) and x=$(best.x)"
    while stagnation < max_stagnation && candidates < max_candidates
        candidates += 1 # Increment the candidate count
        result = take!(candidate_channel) # take a candidate from the channel
        #@info "took candidate $(candidates) $(result.objective) with x=$(result.x)"
        if result.objective > best.objective
            best = result # Update the best solution found so far
            stagnation = 0 # Reset stagnation since we found a better solution
        else
            stagnation += 1 # Increment stagnation since we did not find a better solution
        end
    end
    return best
end

"""
    produce_candidates(candidate_channel, Q̄, d, njumps)

Task loop that continuously produces candidates by finding local maxima using the provided `twoQmD` and `d`.
This function runs indefinitely, putting candidates into the `candidate_channel`, so be sure to bind that channel to a conunsumer task that will stop it when it has enough candidates.
"""
function produce_candidates(candidate_channel, Q̄, d, njumps)
    #@info "producing candidates in task $(current_task())"
    while isopen(candidate_channel) # Continue until the channel is closed
        put_local_max!(candidate_channel, Q̄, d, njumps)# Find a local maximum and put it in the channel
    end
end

function flips(n)
    # It's really amazing that this works, but it does.
    return Iterators.map(i -> trailing_zeros(i) + 1, Base.OneTo(checked_pow(2, n) - 1)) # Generate a sequence of indices to flip based on the binary representation of numbers from 1 to 2^n-1
end

function solve_exact(Q̄, d)
    #@info "Solving exact problem with Q̄ \n$(sprint(show,"text/plain",Q̄)) \nand d = \n$(sprint(show,"text/plain",d))" # Log the problem being solved
    T = promote_type(eltype(Q̄), eltype(d)) # Determine the type for the solution vector
    n = size(Q̄, 2) # Get the number of variables
    x = zeros(T, n) # Initialize the solution vector
    objective = zero(T) # Initialize the best objective value
    sensitivity = deepcopy(d) # Create a sensitivity vector to hold the sensitivity values
    best = (objective=objective, x=deepcopy(x))# Initialize the best solution found so far
    for i in flips(n) # For each combination of flips
        objective += sensitivity[i] # Update the objective value with the sensitivity value for the flipped index
        update_sensitivity!(sensitivity, Q̄, x, i) # Update the sensitivity vector based on the flipped index
        @inbounds x[i] = one(T) - x[i] # Flip the ith bit in the solution vector
        if objective > best.objective # If the objective value is better than the best found so far
            best = (objective=objective, x=deepcopy(x)) # Update the best solution found so far
        end
    end

    return best # Return the best objective value and solution vector
end

"""
    update_sensitivity!(sensitivity, Q̄, x, i)

Update the sensitivity vector in place based on the current solution vector `x` and a bit flip in index `i`.
Faster than having to compute the sensitivity vector from scratch usin `compute_sensitivity`.
"""
function update_sensitivity!(sensitivity, Q̄, x, i)
    col = view(Q̄, :, i) # Get the ith column of Q̄
    if isone(x[i]) # If the ith bit is 1, we need to subtract the column from the sensitivity vector
        sensitivity .-= ifelse.(iszero.(x), col, .-col) # If the ith bit is 1, subtract the column, otherwise add it
    elseif iszero(x[i]) # If the ith bit is 0, we need to add the column to the sensitivity vector
        sensitivity .+= ifelse.(iszero.(x), col, .-col) # If the ith bit is 0, add the column, otherwise subtract it
    else
        throw(ArgumentError("x[i] must be 0 or 1, got $(x[i])")) # If x[i] is not 0 or 1, throw an error    
    end
    sensitivity[i] = -sensitivity[i]
    return sensitivity
end

"""
    set_sensitivity!(sensitivity, Q̄, d, x)

Set the sensitivity vector in place based on the current solution vector `x`.
This function computes the sensitivity vector from scratch, which is slower than `update_sensitivity!`, but is used to initialize the sensitivity vector.
"""
function set_sensitivity!(sensitivity, Q̄, d, x)
    sensitivity .= d # Initialize the sensitivity vector with the diagonal elements
    mul!(sensitivity, Q̄, x, true, true) # Compute the sensitivity vector
    sensitivity .= ifelse.(iszero.(x), sensitivity, .-sensitivity) # Flip the sign of the sensitivity vector where x is 1
    return sensitivity
end

"""
    jump_update!(x, sensitivity, Q̄, d)

Update the solution vector `x` by flipping bits where the sensitivity is positive.
Update sensitivity vector after the jump.

This function may not always improve the soltion, but can be faster than a step update on a low-quality solution.
"""
function jump_update!(x, sensitivity, Q̄, d)
    x .= ifelse.(sensitivity .> zero(eltype(sensitivity)), one(eltype(x)) .- x, x) # Flip bits where sensitivity is positive
    set_sensitivity!(sensitivity, Q̄, d, x) # Recompute the objective value and sensitivity vector after the jump
end

"""
    step_update!(x, sensitivity, Q̄, objective)

Update the solution vector `x` by flipping the bit at index `i` where the sensitivity is maximum and greater than 0.
Update the sensitivity vector after the update.
Returns the updated objective value.

This function is guaranteed to improve the solution if there is a positive sensitivity value.
"""
function step_update!(x, sensitivity, Q̄, objective)
    isempty(sensitivity) && return objective # If sensitivity is empty, we are done
    improvement, i = findmax(sensitivity)
    if improvement <= zero(improvement) # If there is no positive sensitivity, we are done 
        return objective
    end
    update_sensitivity!(sensitivity, Q̄, x, i) # Update the sensitivity vector
    @inbounds x[i] = one(eltype(x)) - x[i] # Change the ith bit in the solution vector
    return objective + improvement # Update the objective value
end

"""
    put_local_max!(candidate_channel, Q̄, d, njumps[, x])

Compute a local maximum from a starting point `x` starting point using the provided `Q̄` and `d`.
If `x` is not provided, a random starting point is used.
Puts the candidate in the `candidate_channel`

"""
function put_local_max!(candidate_channel, Q̄, d, njumps, x::Vector=Vector{promote_type(eltype(Q̄), eltype(d))}(bitrand(length(d))))

    # use a vector of 1 or 0 Ts instead of a BitVector for better performance and simd support
    sensitivity = similar(d) # Create a sensitivity vector to hold the sensitivity values
    set_sensitivity!(sensitivity, Q̄, d, x) # Compute the initial sensitivity vector 
    # do series of jumps to get a better solution
    for _ in 1:njumps
        jump_update!(x, sensitivity, Q̄, d) # Update the solution vector and objective value
    end
    objective = compute_objective(Q̄, d, x) # Compute the objective value
    # iteratively improve the solution one bit at a time
    while isopen(candidate_channel) # Continue unless the channel is closed
        newobjective = step_update!(x, sensitivity, Q̄, objective) # Update the solution vector and objective value
        if newobjective == objective # If the objective value did not change, we are done
            return put!(candidate_channel, (objective=objective, x=x)) # Put the candidate in the channel
        else
            objective = newobjective # Update the objective value
        end
    end
end


"""
    compute_objective(Q̄, d, x)

Compute the objective value for a given solution vector `x`, matrix `Q̄`, and vector `d`.
"""
function compute_objective(Q̄, d, x)
    return x' * UpperTriangular(Q̄) * x + d ⋅ x
end

"""
    utform(Q)

Separate the matrix `Q` into two parts: `Q̄` which contains the off-diagonal elements of the symmetric part of `Q` doubled, and `d` which contains the diagonal elements of `Q`.
`Q` must be square and one-based indexed, but it need not be symmetric.
returns `Q̄` and `d`.
"""
function utform(Q, 𝓁=zeros(eltype(Q), LinearAlgebra.checksquare(Q)))
    T = promote_type(eltype(Q), eltype(𝓁)) # Determine the type for the solution vector
    Q̄ = Q .+ Q' .- convert(T, 2) * Diagonal(Q) # Create a new matrix with doubled off-diagonal elements
    d = convert(Vector{T}, diag(Q)) # Extract the diagonal elements in a dense vector
    d .+= 𝓁 # Add the linear term to the diagonal elements
    return Q̄, d
end

end
