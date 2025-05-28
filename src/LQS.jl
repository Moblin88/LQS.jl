module LQS

using LinearAlgebra
using Random
using Base.Iterators

function solve(Q::AbstractMatrix)
    if(!ishermitian(Q))
        Q = hermitianpart(Q)
    end
    d = diag(Q)
    A = 2 .* (Q .-  Diagonal(d))
    # Initialize the solution vector
    best = (0.0, falses(size(Q, 1)))
    misses = 0
    while misses < 1000
        xinit = bitrand(size(Q, 1)) # Randomly initialize the solution vector
        sensitivity = A * xinit + d # Initial sensitivity vector
        objective = xinit' * Q * xinit # Initial objective value
        # store x as a series of -1s and 1s, rather than 0s and 1s
        x = 2 .* xinit .- 1 # Convert to -1 and 1 representation
        sensitivity .*= .- x 
        while true
            improvement, i = findmax(sensitivity)
            if(improvement <= 0)
                # No improvement found, we are in a local minimum
                best = max(best, (objective, x .>= 0))
                misses += 1
                break
            else
                # we have an improvement, update the sensitivity and flip the bit
                sensitivity .+= x[i] .* x .* view(A,:, i)
                sensitivity[i] = -sensitivity[i] # Flip the sign of the sensitivity vector in row i
                x[i] = -x[i] # Flip the sign of the ith bit
                objective += improvement
            end
        end
    end
    return best
end

end
