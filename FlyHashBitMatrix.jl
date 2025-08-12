using Random, LinearAlgebra, SparseArrays, StatsBase, .Threads

"""
    random_binary_matrix(m::Int, d::Int, s::Int, seed::Int) ->
    BitMatrix

Creates a sparse binary projection matrix of size `m x d`, where each row has
exactly `s` non-zero elements, placed in random columns.

# Arguments
- `m::Int`: Number of rows.
- `d::Int`: Number of columns.
- `s::Int`: Number of non-zero elements per row.
- `seed::Int`: Seed for initializing the random number generators.

# Returns
- `BitMatrix`: The generated sparse binary matrix.
"""
function random_binary_matrix_bit(m::Int, d::Int, s::Int, seed::Int)
    # Create an independent Random Number Generator (RNG) for each thread
    # to ensure reproducibility and avoid data races on the global RNG.
    rngs = [MersenneTwister(seed + i) for i in 1:Threads.nthreads()]
    
    # Initialize BitMatrix M
    M = falses(m,d)
    
    # Use multithreading to generate rows in parallel.
    @threads for i in 1:m
        tid = threadid()

        # Use the thread-specific RNG for sampling.
        sampled_cols = sample(rngs[tid], 1:d, s; replace=false)

        # Set sampled columns to true
        M[i, sampled_cols] .= true 
    end
    
    return M
end


"""
    fly_hash(X::AbstractMatrix, M::SparseMatrixCSC, ρ::Int) ->
    BitMatrix

Computes the FlyHash of each column of matrix `X`, using the projection matrix `M`.
For each column, the FlyHash is defined by the indices of the `ρ` largest
projections.

# Arguments
- `X::AbstractMatrix`: Input matrix (d x n).
- `M::BitMatrix`: Random projection matrix (m x d).
- `ρ::Int`: Number of active hashes per column (the "top-ρ").

# Returns
- `BitMatrix`: The sparse hash matrix (m x n).
"""
function fly_hash_bit(X::AbstractMatrix, M::BitMatrix, ρ::Int)
    n = size(X, 2)
    m = size(M, 1)

    # Determine the computation type based on the element types of X and M.
    T = promote_type(eltype(X), eltype(M))

    # Thread-local storage to avoid race conditions and allocations.
    # Each thread gets its own temporary vector `x_proj` for intermediate results.
    x_proj_local = [Vector{T}(undef, m) for _ in 1:Threads.nthreads()]
    
    # Initialize BitMatrix H
    H = falses(m,n)
    
    @threads for i in 1:n
        tid = threadid()
        
        # Get the current thread's temporary vector.
        x_proj = x_proj_local[tid]
        
        # If we use a view, the execution never ends, so we create a copy
        x_col = X[:, i]
        
        # Efficient in-place matrix-vector multiplication.
        mul!(x_proj, M, x_col)

        # Find the indices of the ρ largest values.
        # `partialsortperm` is much faster than a full sort.
        topk_idx = partialsortperm(x_proj, 1:ρ; rev=true)

        H[topk_idx, i] .= true
    end

    return H
end