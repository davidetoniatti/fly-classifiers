"""
    rupm(m::Int, d::Int, seed::Int) ->
    Matrix{Float64}

Creates a random projection matrix of size `m x d`, whose row are sampled
i.i.d. from the uniform distribution over S^{d-1}.

# Arguments
- `m::Int`: Number of rows.
- `d::Int`: Number of columns.
- `seed::Int`: Seed for initializing the random number generators.

# Returns
- `Matrix{Float64}`: The generated projection matrix.
"""
function rupm(m::Int, d::Int; seed::Int = 42)
    rng = MersenneTwister(seed)
    mat = randn(rng, m, d)
    
    # Normalize each row in-place to have unit L2 norm.
    @inbounds for i in 1:m
        # Get a view of the current row to avoid copying data.
        row_view = @view mat[i, :]
        
        # Calculate the norm of the row.
        row_norm = norm(row_view)
        
        # Normalize the row. Add a small epsilon for numerical stability.
        # This prevents division by zero if a row happens to be all zeros.
        mat[i, :] ./= (row_norm + eps(eltype(mat)))
    end

    return mat
end


"""
    fit(::Type{EaS}, X::AbstractMatrix, y::AbstractVector, m::Int, k::Int,
    seed::Int) -> EaS

Trains the EaS classifier.

# Arguments
- `::Type{EaS}`: The model to be fitted.
- `X::AbstractMatrix`: Training data matrix (d x n).
- `y::AbstractVector`: Training labels (n-element vector).
- `m::Int`: The dimension of the projection space.
- `k::Int`: The number of active response regions per item.
- `seed::Int`: Random seed for reproducibility.

# Returns
- `EaS`: The trained model containing the projection matrix and weights.
"""
function fit(::Type{EaS}, X::AbstractMatrix{T}, y::AbstractVector, m::Int, k::Int, seed::Int) where T
    d, n = size(X)
    @assert length(y) == n "Number of labels does not match number of data points."
    
    # Compute random projection matrix
    P = rupm(m,d; seed)

    # Determine the computation type based on the element types of X and M.
    T_proj = promote_type(T, eltype(P))

    # Safe parallelization with thread-local storage for weights and counters
    # We initialize a weight and counter vector for each thread to prevent race conditions.
    w_local  = [zeros(Int, m) for _ in 1:nthreads()]
    ct_local = [zeros(Int, m) for _ in 1:nthreads()]
    x_proj_local = [Vector{T_proj}(undef, m) for _ in 1:Threads.nthreads()]
    top_idxs_local = [Vector{Int}(undef, k) for _ in 1:nthreads()]
    top_vals_local = [Vector{T_proj}(undef, k) for _ in 1:nthreads()]

    @threads for i in 1:n
        tid = threadid()
        
        w = w_local[tid]
        ct = ct_local[tid]
        x_proj = x_proj_local[tid]
        top_idxs = top_idxs_local[tid]
        top_vals = top_vals_local[tid]

        x_view = @view X[:, i]
        mul!(x_proj, P, x_view)
        _topk_indices!(top_idxs, top_vals, x_proj, k)

        label = y[i]
        @inbounds for j in top_idxs
            w[j] += label
            ct[j] += 1
        end
    end
    
    w_total  = reduce(+, w_local)
    ct_total = reduce(+, ct_local)
    
    w_normalized = zeros(Float64, m)
    valid_indices = ct_total .> 0
    
    @inbounds w_normalized[valid_indices] .= w_total[valid_indices] ./ ct_total[valid_indices]

    return EaS(P, w_normalized, ct_total, k)
end

"""
    predict(model::EaS, X::AbstractMatrix) -> Vector

Performs inference on new data using a trained EaS model.

# Arguments
- `model::EaS`: The trained EaS model object.
- `X::AbstractMatrix`: The data matrix (d x n).

# Returns
- `Vector`: A vector of scores for each column in `X`.
"""
function predict(model::EaS, X::AbstractMatrix{T}) where T
    n = size(X, 2)
    m = length(model.w)
    scores = zeros(n)

    T_proj = promote_type(T, eltype(model.P))

    x_proj_local = [Vector{T_proj}(undef, m) for _ in 1:Threads.nthreads()]
    top_idxs_local = [Vector{Int}(undef, model.k) for _ in 1:nthreads()]
    top_vals_local = [Vector{T_proj}(undef, model.k) for _ in 1:nthreads()]

    @threads for i in 1:n
        tid = threadid()

        x_proj = x_proj_local[tid]
        top_idxs = top_idxs_local[tid]
        top_vals = top_vals_local[tid]

        x_view = @view X[:, i]
        mul!(x_proj, model.P, x_view)
        _topk_indices!(top_idxs, top_vals, x_proj, model.k)

        # qui non dovrebbero esserci problemi di race conditions
        @inbounds scores[i] = sum(@view model.w[top_idxs]) / model.k
    end

    return scores
end