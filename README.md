# LQS.jl - Local QUBO Solver

[![Build Status](https://github.com/Moblin88/LQS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Moblin88/LQS.jl/actions/workflows/CI.yml?query=branch%3Amain)

A high-performance Julia package for solving Quadratic Unconstrained Binary Optimization (QUBO) problems using a combination of exact methods for small subproblems and local search heuristics for larger ones.

## Overview

LQS (Local QUBO Solver) is designed to efficiently solve QUBO problems of the form:

```
maximize: x^T Q x + ℓ^T x + β
subject to: x ∈ {0,1}^n
```

where:
- `Q` is an n×n real matrix (the quadratic term)
- `ℓ` is an n-dimensional real vector (the linear term)
- `β` is a real scalar (the constant term)
- `x` is an n-dimensional binary decision vector

## Key Features

### 🚀 **Hybrid Approach**
- **Exact solving** for small subproblems (via exhaustive enumeration)
- **Local search heuristics** for larger subproblems using multi-threaded candidate generation
- **Automatic threshold selection** between exact and approximate methods

### 🔧 **Problem Preprocessing**
- **Variable pruning**: Automatically identifies and fixes variables that must be 0 or 1
- **Component decomposition**: Decomposes problems into independent subproblems based on matrix structure
- **Hot start support**: Can initialize search from a provided starting solution

### ⚡ **Performance Optimizations**
- **Multi-threading**: Parallel candidate generation and evaluation
- **Efficient sensitivity updates**: Fast incremental objective function evaluation
- **Memory-efficient**: Uses specialized data structures for binary variables

### 🎯 **Adaptive Search Strategy**
- **Jump-based exploration**: Performs multiple bit flips to escape local optima
- **Greedy improvement**: Iterative single-bit improvements for local optimization
- **Stagnation detection**: Automatically terminates when no improvement is found

## Installation

```julia
using Pkg
Pkg.add("LQS")
```

Or for the development version:
```julia
using Pkg
Pkg.add(url="https://github.com/Moblin88/LQS.jl")
```

## Quick Start

```julia
using LQS

# Define a simple QUBO problem
Q = [2 -1; -1 2]  # Quadratic matrix
ℓ = [-1, -1]      # Linear terms (optional)
β = 0             # Constant term (optional)

# Solve the problem
result = solve(Q, ℓ, β)

println("Optimal objective: ", result.objective)
println("Optimal solution: ", result.x)
println("Optimality status: ", result.optimal)
```

## API Reference

### Main Function

```julia
solve(Q::AbstractMatrix{<:Real},
      ℓ::AbstractVector{<:Real} = zeros(eltype(Q), size(Q,1)),
      β::Real = zero(eltype(Q));
      kwargs...)
```

**Parameters:**
- `Q`: Square matrix representing quadratic coefficients (does not have to be symmetric, can be sparse or dense with any signed Real eltype, including integers and BigInt/BigFloats)
- `ℓ`: Vector of linear coefficients (defaults to zeros)
- `β`: Constant term (defaults to zero)

**Keyword Arguments:**
- `hot_start=nothing`: Initial solution vector for warm start
- `max_stagnation=size(Q,1)`: Maximum iterations without improvement
- `max_candidates=10*size(Q,1)`: Maximum candidates to evaluate
- `exact_threshold=min(1.5*max_stagnation, max_candidates)`: Threshold for exact vs. approximate solving. Given in number of non-zero candidates, not number of variables.
- `ntasks=Threads.threadpoolsize()`: Number of parallel tasks for candidate generation
- `njumps=3`: Number of multi-index jumps at the start of each local search solution
- `prune=true`: Enable variable pruning preprocessing
- `decompose=true`: Enable problem decomposition

**Returns:**
A named tuple with:
- `objective::Real`: The optimal objective value found
- `x::Vector`: The optimal binary solution vector
- `optimal::BitVector`: Boolean vector indicating which variables were solved exactly

## Algorithm Details

### Problem Preprocessing

1. **Upper Triangular Form**: Converts the general QUBO matrix into an efficient representation
2. **Variable Pruning**: Identifies variables that must be 0 or 1 based on the problem structure
3. **Component Separation**: Decomposes the problem into independent subproblems

### Solving Strategy

For each connected component:

- **Small components** (≤ log₂(exact_threshold + 1) variables): Solved exactly using exhaustive enumeration
- **Large components**: Solved approximately using multi-threaded local search

### Local Search Algorithm

1. **Initialization**: Start from random solution or provided hot start
2. **Jump Phase**: Perform `njumps` multi-bit flips to explore the solution space
3. **Improvement Phase**: Iteratively flip single bits that improve the objective
4. **Termination**: Stop when no improving moves exist (local optimum reached)

## Examples

### Basic Usage

```julia
using LQS

# Max-Cut problem on a triangle
Q = [0 1 1; 1 0 1; 1 1 0]
result = solve(Q)
println("Max-Cut value: ", result.objective)
```

### Advanced Configuration

```julia
using LQS

# Large problem with custom parameters
Q = randn(100, 100)

result = solve(Q;
    hot_start=ones(Int, 100),     # Start from all-ones solution
    max_stagnation=50,            # Allow more exploration
    ntasks=8,                     # Use 8 parallel tasks
    njumps=5,                     # More jumps per candidate solution
    exact_threshold=20            # Solve components exactly if possible in 20 states or fewer
```

### Problem Decomposition Analysis

```julia
using LQS

# Block diagonal problem
Q = [1 2 0 0;
     2 1 0 0;
     0 0 3 1;
     0 0 1 3]

result = solve(Q; decompose=true)
# Will automatically solve as two 2×2 subproblems
```

## Performance Tips

1. **Enable decomposition** for structured problems with independent components
2. **Use hot starts** when you have good initial solutions
3. **Tune `ntasks`** based on your CPU cores for parallel problems
4. **Adjust `exact_threshold`** - higher values give better quality but slower performance
5. **Enable pruning** for problems with obvious variable fixings

## Theoretical Background

LQS implements a hybrid approach combining:

- **Exact enumeration** for tractable subproblems
- **Tabu-free local search** with strategic diversification
- **Problem decomposition** based on variable interaction graphs
- **Preprocessing techniques** for problem size reduction

The algorithm is particularly effective for:
- Large sparse QUBO instances
- Problems with natural decomposition structure  
- Applications requiring good solutions quickly rather than proven optimality

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use LQS.jl in your research, please cite:

```bibtex
@software{lqs_jl,
  author = {Nicholas Engelking},
  title = {LQS.jl: Local QUBO Solver},
  url = {https://github.com/Moblin88/LQS.jl},
  year = {2025}
}
```