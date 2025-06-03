module LQS

using LinearAlgebra
using Random
using Base.Threads

function solve(Q::AbstractMatrix{<:Real};
               max_stagnation::Int = 1000,
               max_candidates::Int = 10000)
    
    twoQmD, d, T = separateq(Q) # separate Q into twoQmD = 2*(Q-Diagonal(Q)) and d = diag(Q)
    # prune A so that we remove dominant elements
    
    candidate_channel = Channel{@NamedTuple{objective::T, x::Vector{Bool}}}() # Channel to hold candidates
    checker_task = Task(()-> check_candidates(candidate_channel, max_stagnation, max_candidates)) # Task to check candidates
    checker_task.sticky = false # Make the task non-sticky to allow it to change threads
    bind(candidate_channel, checker_task) # Bind the channel to the task so it kills producers when it's done
    schedule(checker_task) # Schedule the task to run
    for _ in 1:Threads.threadpoolsize()
        @spawn produce_candidates(candidate_channel, twoQmD, d) # Spawn a thread to produce candidates
    end
    return fetch(checker_task) # Wait for the task to finish and return the best solution found
end


"""
    check_candidates(candidate_channel, max_stagnation, max_candidates)

Task loop that checks the candidates in the `candidate_channel` and returns the best solution after stagnation or reaching the maximum number of candidates.
"""
function check_candidates(candidate_channel, max_stagnation, max_candidates)
    candidates = 1
    stagnation = 0
    best = take!(candidate_channel) # Take the first candidate from the channel
    while stagnation < max_stagnation && candidates < max_candidates
        candidates += 1 # Increment the candidate count
        result = take!(candidate_channel) # take a candidate from the channel
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
    if x[i]
        for j in eachindex(sensitivity, x, col)
            sensitivity[j] -= ifelse(x[j],-col[j], col[j])
        end
    else
        for j in eachindex(sensitivity, x, col)
            sensitivity[j] += ifelse(x[j],-col[j], col[j])
        end
    end
    sensitivity[i] = -sensitivity[i]  # Flip the sign of the ith sensitivity value
    return sensitivity
end


"""
    find_local_max(twoQmD, d)

Compute a local maximum from a random starting point using the provided `twoQmD` and `d`.
"""
function find_local_max(twoQmD, d)
    x = Vector(bitrand(length(d))) # Randomly initialize the solution vector
    # use a vector of bools instead of a BitVector for better performance in the loop and simd support
    objective, sensitivity = initalize_soltion(twoQmD, d, x) # Compute the initial sensitivity vector and objective value
    # do a one time jump update
    x .= ifelse.(sensitivity .> 0, .!x, x) # Flip bits where sensitivity is positive
    objective, sensitivity = initalize_soltion(twoQmD, d, x) # Recompute the objective value and sensitivity vector after the jump
    # iteratively improve the solution one bit at a time
    while true
        improvement, i = findmax(sensitivity)
        if improvement <= 0
            # No improvement found, we are in a local maximum
            return (objective=objective, x=x)
        else
            # we have an improvement
            update_sensitivity!(sensitivity, twoQmD, x, i) # Update the sensitivity vector
            @inbounds x[i] = !x[i] # change the ith bit in the solution vector
            objective += improvement # Update the objective value
        end
    end
end

"""
    compute_sensitivity(twoQmD, d, x)

Compute the sensitivity vector based on the current solution vector `x`.
The sensitivity vector is defined as the change in the objective value when flipping each bit in the solution vector.
"""
function compute_sensitivity(twoQmD, d, x)
    sensitivity = muladd(twoQmD, x, d)
    sensitivity .= ifelse.(x,.-sensitivity, sensitivity) # Flip the sign of the sensitivity vector based on the current solution
    return sensitivity
end

function compute_objective(twoQmD::AbstractMatrix{<:Real}, d::AbstractVector{<:Real}, x::AbstractVector{Bool})
    Base.require_one_based_indexing(twoQmD, d, x) # Ensure twoQmD is one-based indexed
    n = LinearAlgebra.checksquare(twoQmD) # Ensure twoQmD is square
    length(d) == n || throw(DimensionMismatch("Length of d must match the size of twoQmD"))
    length(x) == n || throw(DimensionMismatch("Length of x must match the size of twoQmD"))
    T = promote_type(eltype(twoQmD), eltype(d)) # Determine the type for the objective value
    objective = zero(T) # Initialize the objective value
    @inbounds for j in 1:n
        x[j] || continue # Skip if the jth bit is not set
        objective += d[j] # Add the diagonal element if the bit is set
        for i in (j+1):n
            objective += ifelse(x[i], twoQmD[i,j], zero(T)) # Add the off-diagonal element if both bits are set
        end
    end
    return objective
end

function initalize_soltion(twoQmD, d, x)
    sensitivity = compute_sensitivity(twoQmD, d, x) # Compute the initial sensitivity vector
    objective = compute_objective(twoQmD, d, x) # Initial objective value
    return objective, sensitivity
end

"""
    separateq(Q)

Separate the matrix `Q` into two parts: `twoQmD` which contains the off-diagonal elements of the symmetric part of `Q` doubled, and `d` which contains the diagonal elements of `Q`.
`Q` must be square and one-based indexed, but it need not be symmetric.
returns `twoQmD` and `d`.
"""
function separateq(Q)
    Base.require_one_based_indexing(Q) 
    n = LinearAlgebra.checksquare(Q)
    T = eltype(Q)
    twoQmD = similar(Q, T, n, n)
    d = similar(Q, T, n)
    @inbounds for j in 1:n
        d[j] = Q[j,j]
        twoQmD[j,j] = zero(T) # Diagonal elements are zero
        for i in (j+1):n
            twoQmD[i,j] = twoQmD[j,i] = Q[i,j] + Q[j,i]
        end
    end
    return twoQmD, d, T
end

end
