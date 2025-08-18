"""
    sbpm(m::Int, d::Int, s::Int, seed::Int) ->
    SparseMatrixCSC{Bool, Int}

Creates a sparse binary projection matrix of size `m x d`, where each row has
exactly `s` non-zero elements, placed in random columns.

# Arguments
- `m::Int`: Number of rows.
- `d::Int`: Number of columns.
- `s::Int`: Number of non-zero elements per row.
- `seed::Int`: Seed for initializing the random number generators.

# Returns
- `SparseMatrixCSC{Bool, Int}`: The generated sparse binary matrix.
"""
function sbpm(m::Int, d::Int, s::Int, seed::Int)
    # Set random seed.
    Random.seed!(seed)
    
    nnz = m * s
    row_idx = Vector{Int}(undef, nnz)
    col_idx = Vector{Int}(undef, nnz)
    
    # Thread-local storage to avoid race conditions and allocations.
    # Each thread gets its own temporary vector `buf` for intermediate samplings.
    buf = [Vector{Int}(undef, s) for _ in 1:nthreads()]
    
    # Use multithreading to generate rows in parallel.
    @threads for i in 1:m
        tid = threadid()
        start_pos = (i - 1) * s + 1
        end_pos   = i * s

        idxs = buf[tid]

        # Sampling.
        sample!(1:d, idxs; replace=false)

        # Fill the row and column index vectors.
        @inbounds row_idx[start_pos:end_pos] .= i
        @inbounds col_idx[start_pos:end_pos] .= idxs
    end
    
    # The values are all `true`, so we can construct the sparse matrix directly.
    return sparse(row_idx, col_idx, true, m, d)
end


"""
    fly_hash(X::AbstractMatrix, M::SparseMatrixCSC, ρ::Int) ->
    SparseMatrixCSC{Bool, Int}

Computes the FlyHash of each column of matrix `X`, using the projection matrix `M`.
For each column, the FlyHash is defined by the indices of the `ρ` largest
projections.

# Arguments
- `X::AbstractMatrix`: Input matrix (d x n).
- `M::SparseMatrixCSC`: Random projection matrix (m x d).
- `ρ::Int`: Number of active hashes per column (the "top-ρ").

# Returns
- `SparseMatrixCSC{Bool, Int}`: The sparse hash matrix (m x n).
"""
function fly_hash(X::AbstractMatrix, M::SparseMatrixCSC{Bool,Int}, ρ::Int)
    n = size(X, 2)
    m = size(M, 1)

    # Determine the computation type based on the element types of X and M.
    T = promote_type(eltype(X), eltype(M))

    # Thread-local storage to avoid race conditions and allocations.
    # Each thread gets its own temporary vector `x_proj` for intermediate results.
    x_proj_local = [Vector{T}(undef, m) for _ in 1:Threads.nthreads()]
    top_idxs_local   = [Vector{Int}(undef, ρ) for _ in 1:nthreads()]
    top_vals_local   = [Vector{T}(undef, ρ) for _ in 1:nthreads()]
    
    nnz = n * ρ
    row_idx = Vector{Int}(undef, nnz)
    col_idx = Vector{Int}(undef, nnz)

    @threads for i in 1:n
        tid = threadid()
        start_pos = (i - 1) * ρ + 1
        end_pos   = i * ρ
        
        # Get the current thread's temporary vector.
        x_proj = x_proj_local[tid]
        top_idxs = top_idxs_local[tid]
        top_vals = top_vals_local[tid]
        
        # Efficient in-place matrix-vector multiplication.
        # TODO: can I do better by addressing the sparsity prop?
        mul!(x_proj, M, X[:, i])

        # Find the indices of the ρ largest values.
        # `partialsortperm` is much faster than a full sort.
        _topk_indices!(top_idxs, top_vals, x_proj, ρ)
        
        @inbounds row_idx[start_pos:end_pos] .= top_idxs
        @inbounds col_idx[start_pos:end_pos] .= i
    end

    return sparse(row_idx, col_idx, true, m, n)
end