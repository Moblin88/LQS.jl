module LQS

using LinearAlgebra
using Random
using Base.Threads

function solve(Q::AbstractMatrix{<:Real};
               max_stagnation::Int = 1000,
               max_candidates::Int = 10000)
    
    twoQmD, d = separateq(Q) # separate Q into twoQmD = 2*(Q-Diagonal(Q)) and d = diag(Q)
    # prune A so that we remove dominant elements
    T = eltype(Q) # Determine the type for the solution vector
    candidate_channel = Channel{@NamedTuple{objective::T, x::Vector{T}}}() # Channel to hold candidates
    checker_task = Task() do 
        check_candidates(max_stagnation, max_candidates) do 
            take!(candidate_channel)
        end
    end # Task to check candidates
    checker_task.sticky = false # Make the task non-sticky to allow it to change threads
    bind(candidate_channel, checker_task) # Bind the channel to the task so it kills producers when it's done
    schedule(checker_task) # Schedule the task to run
    for _ in 1:Threads.threadpoolsize()
        @spawn produce_candidates(candidate_channel, twoQmD, d) # Spawn a thread to produce candidates
    end
    return fetch(checker_task) # Wait for the task to finish and return the best solution found
end

function solve_single_thread(Q::AbstractMatrix{<:Real};
               max_stagnation::Int = 1000,
               max_candidates::Int = 10000)
    
    twoQmD, d = separateq(Q) # separate Q into twoQmD = 2*(Q-Diagonal(Q)) and d = diag(Q)
    # prune A so that we remove dominant elements
    
    return check_candidates(max_stagnation, max_candidates) do 
        find_local_max(twoQmD, d)
    end
end


"""
    check_candidates(candidate_channel, max_stagnation, max_candidates)

Task loop that checks the candidates in the `candidate_channel` and returns the best solution after stagnation or reaching the maximum number of candidates.
"""
function check_candidates(produce_candidate, max_stagnation, max_candidates)
    candidates = 1
    stagnation = 0
    best = produce_candidate() # Take the first candidate from the channel
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
    if x[i]===one(eltype(x))
        sensitivity .-= (1 .- 2 .* x) .* col # If the ith bit is 1, subtract the column, otherwise add it
    else
        sensitivity .+= (1 .- 2 .* x) .* col # If the ith bit is 0, add the column, otherwise subtract it
    end
    sensitivity[i] = -sensitivity[i]  # Flip the sign of the ith sensitivity value
    return sensitivity
end


"""
    find_local_max(twoQmD, d)

Compute a local maximum from a random starting point using the provided `twoQmD` and `d`.
"""
function find_local_max(twoQmD, d)
    T = promote_type(eltype(twoQmD), eltype(d)) # Determine the type for the solution vector
    x = Vector{T}(bitrand(length(d))) # Randomly initialize the solution vector
    # use a vector of 1 or 0 Ts instead of a BitVector for better performance and simd support
    objective, sensitivity = initalize_soltion(twoQmD, d, x) # Compute the initial sensitivity vector and objective value
    # do a one time jump update
    x .= ifelse.(sensitivity .> zero(T), one(T) .- x, x) # Flip bits where sensitivity is positive
    objective, sensitivity = initalize_soltion(twoQmD, d, x) # Recompute the objective value and sensitivity vector after the jump
    # iteratively improve the solution one bit at a time
    while true
        improvement, i = findmax(sensitivity)
        if improvement <= zero(T)
            # No improvement found, we are in a local maximum
            return (objective=objective, x=x)
        else
            # we have an improvement
            update_sensitivity!(sensitivity, twoQmD, x, i) # Update the sensitivity vector
            @inbounds x[i] = one(T) - x[i] # change the ith bit in the solution vector
            objective += improvement # Update the objective value
        end
    end
end


function initalize_soltion(twoQmD, d, x)
    sensitivity =  muladd(twoQmD, x, d) # Compute the sensitivity vector
    sensitivity .= ifelse.(x .=== one(eltype(x)), .-sensitivity, sensitivity) # Flip the sign of the sensitivity vector where x is 1
    objective = x' * UpperTriangular(twoQmD) * x + d⋅x # Initial objective value
    return objective, sensitivity
end

"""
    separateq(Q)

Separate the matrix `Q` into two parts: `twoQmD` which contains the off-diagonal elements of the symmetric part of `Q` doubled, and `d` which contains the diagonal elements of `Q`.
`Q` must be square and one-based indexed, but it need not be symmetric.
returns `twoQmD` and `d`.
"""
function separateq(Q)
    twoQmD = Q .+ Q' .- 2*Diagonal(Q) # Create a new matrix with doubled off-diagonal elements
    d = convert(Vector,diag(Q)) # Extract the diagonal elements in a dense vector
    return twoQmD, d
end

end
