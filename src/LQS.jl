module LQS

using LinearAlgebra
using Random
using Base.Threads

function solve(Q::AbstractMatrix{<:Real};
    max_stagnation=size(Q, 1),
    max_candidates=10 * size(Q, 1),
    hot_start=nothing)

    twoQmD, d = separateq(Q) # separate Q into twoQmD = 2*(Q-Diagonal(Q)) and d = diag(Q)
    # prune A so that we remove dominant elements
    T = eltype(Q) # Determine the type for the solution vector
    candidate_channel = Channel{@NamedTuple{objective::T, x::Vector{T}}}() # Channel to hold candidates
    try
        for _ in 1:Threads.threadpoolsize()
            @spawn produce_candidates(candidate_channel, twoQmD, d) # Spawn a thread to produce candidates
        end
        if isnothing(hot_start)
            first_candidate = nothing
        else
            first_candidate = find_local_max(twoQmD, d, hot_start) # If a hot start is provided, use it to produce the first candidate
        end
        checker_task = Task() do
            check_candidates(max_stagnation, max_candidates, first_candidate) do
                take!(candidate_channel) # Take a candidate from the channel
            end
        end # Task to check candidates
        checker_task.sticky = false # Make the task non-sticky to allow it to change threads
        bind(candidate_channel, checker_task) # Bind the channel to the task so it kills producers when it's done
        schedule(checker_task) # Schedule the task to run
        return fetch(checker_task) # Wait for the task to finish and return the best solution found
    finally
        isopen(candidate_channel) && close(candidate_channel) # Close the channel to stop the producer tasks
    end
end

function solve_single_thread(Q::AbstractMatrix{<:Real};
    max_stagnation=10 * size(Q, 1),
    max_candidates=100 * size(Q, 1),
    hot_start=nothing)

    twoQmD, d = separateq(Q) # separate Q into twoQmD = 2*(Q-Diagonal(Q)) and d = diag(Q)
    if isnothing(hot_start)
        first_candidate = nothing
    else
        first_candidate = find_local_max(twoQmD, d, hot_start) # If a hot start is provided, use it to produce the first candidate
    end
    return check_candidates(max_stagnation, max_candidates, first_candidate) do
        find_local_max(twoQmD, d) # Find a local maximum and return it as a candidate
    end
end


"""
    check_candidates(candidate_channel, max_stagnation, max_candidates)

Task loop that checks the candidates in the `candidate_channel` and returns the best solution after stagnation or reaching the maximum number of candidates.
"""
function check_candidates(produce_candidate, max_stagnation, max_candidates, first_candidate=nothing)
    candidates = 1
    stagnation = 0
    best = isnothing(first_candidate) ? produce_candidate() : first_candidate # Take the first candidate from the channel
    while stagnation < max_stagnation && candidates < max_candidates
        candidates += 1 # Increment the candidate count
        result = produce_candidate() # take a candidate from the channel
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
    produce_candidates(candidate_channel, twoQmD, d)

Task loop that continuously produces candidates by finding local maxima using the provided `twoQmD` and `d`.
This function runs indefinitely, putting candidates into the `candidate_channel`, so be sure to bind that channel to a conunsumer task that will stop it when it has enough candidates.
"""
function produce_candidates(candidate_channel, twoQmD, d)
    while true
        put!(candidate_channel, find_local_max(twoQmD, d)) # Find a local maximum and put it in the channel
    end
end


"""
    update_sensitivity!(sensitivity, twoQmD, x, i)

Update the sensitivity vector in place based on the current solution vector `x` and a bit flip in index `i`.
Faster than having to compute the sensitivity vector from scratch usin `compute_sensitivity`.
"""
function update_sensitivity!(sensitivity, twoQmD, x, i)
    col = view(twoQmD, :, i) # Get the ith column of twoQmD
    if isone(x[i]) # If the ith bit is 1, we need to subtract the column from the sensitivity vector
        sensitivity .-= (one(eltype(x)) .- 2 .* x) .* col # If the ith bit is 1, subtract the column, otherwise add it
    else
        sensitivity .+= (one(eltype(x)) .- 2 .* x) .* col # If the ith bit is 0, add the column, otherwise subtract it
    end
    sensitivity[i] = -sensitivity[i]
    return sensitivity
end

"""
    set_sensitivity!(sensitivity, twoQmD, d, x)

Set the sensitivity vector in place based on the current solution vector `x`.
This function computes the sensitivity vector from scratch, which is slower than `update_sensitivity!`, but is used to initialize the sensitivity vector.
"""
function set_sensitivity!(sensitivity, twoQmD, d, x)
    sensitivity .= d # Initialize the sensitivity vector with the diagonal elements
    mul!(sensitivity, twoQmD, x, true, true) # Compute the sensitivity vector
    sensitivity .= ifelse.(isone.(x), .-sensitivity, sensitivity) # Flip the sign of the sensitivity vector where x is 1
    return sensitivity
end

"""
    jump_update!(x, sensitivity, twoQmD, d)

Update the solution vector `x` by flipping bits where the sensitivity is positive.
Update sensitivity vector after the jump.

This function may not always improve the soltion, but can be faster than a step update on a low-quality solution.
"""
function jump_update!(x, sensitivity, twoQmD, d)
    x .= ifelse.(sensitivity .> zero(eltype(sensitivity)), one(eltype(x)) .- x, x) # Flip bits where sensitivity is positive
    set_sensitivity!(sensitivity, twoQmD, d, x) # Recompute the objective value and sensitivity vector after the jump
end

"""
    step_update!(x, sensitivity, twoQmD, objective)

Update the solution vector `x` by flipping the bit at index `i` where the sensitivity is maximum and greater than 0.
Update the sensitivity vector after the update.
Returns the updated objective value.

This function is guaranteed to improve the solution if there is a positive sensitivity value.
"""
function step_update!(x, sensitivity, twoQmD, objective)
    improvement, i = findmax(sensitivity)
    if improvement <= zero(improvement) # If there is no positive sensitivity, we are done 
        return objective
    end
    update_sensitivity!(sensitivity, twoQmD, x, i) # Update the sensitivity vector
    @inbounds x[i] = one(eltype(x)) - x[i] # Change the ith bit in the solution vector
    objective += improvement # Update the objective value
    return objective # Return the updated solution
end

"""
    find_local_max(twoQmD, d [,x])

Compute a local maximum from a starting point `x` starting point using the provided `twoQmD` and `d`.
If `x` is not provided, a random starting point is used.
Returns a named tuple with the objective value and the solution vector `x`.
"""
function find_local_max(twoQmD, d, x=Vector{promote_type(eltype(twoQmD), eltype(d))}(bitrand(length(d))))
    # use a vector of 1 or 0 Ts instead of a BitVector for better performance and simd support
    sensitivity = similar(d) # Create a sensitivity vector to hold the sensitivity values
    set_sensitivity!(sensitivity, twoQmD, d, x) # Compute the initial sensitivity vector 
    # do series of jumps to get a better solution
    for _ in 1:5
        jump_update!(x, sensitivity, twoQmD, d) # Update the solution vector and objective value
    end
    objective = compute_objective(twoQmD, d, x) # Compute the objective value
    # iteratively improve the solution one bit at a time
    while true
        newobjective = step_update!(x, sensitivity, twoQmD, objective) # Update the solution vector and objective value
        if newobjective == objective # If the objective value did not change, we are done
            return (objective=objective, x=x) # Return the best solution found
        else
            objective = newobjective # Update the objective value
        end
    end
end


"""
    compute_objective(twoQmD, d, x)

Compute the objective value for a given solution vector `x`, matrix `twoQmD`, and vector `d`.
"""
function compute_objective(twoQmD, d, x)
    return x' * UpperTriangular(twoQmD) * x + d ⋅ x
end

"""
    separateq(Q)

Separate the matrix `Q` into two parts: `twoQmD` which contains the off-diagonal elements of the symmetric part of `Q` doubled, and `d` which contains the diagonal elements of `Q`.
`Q` must be square and one-based indexed, but it need not be symmetric.
returns `twoQmD` and `d`.
"""
function separateq(Q)
    twoQmD = Q .+ Q' .- convert(eltype(Q), 2) * Diagonal(Q) # Create a new matrix with doubled off-diagonal elements
    d = convert(Vector, diag(Q)) # Extract the diagonal elements in a dense vector
    return twoQmD, d
end

end
