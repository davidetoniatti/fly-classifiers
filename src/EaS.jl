function random_projection_matrix(m::Int, d::Int; seed::Int = 42)
    Random.seed!(seed)
    mat = randn(m, d)
    
    norms = sqrt.(sum(mat.^2, dims=2))
    
    # Add a small constant (epsilon) to avoid division by zero
    # in case a row has zero norm.
    norms .+= eps()
    
    P = mat ./ norms

    return P
end

function fit(::Type{EaS}, X::AbstractMatrix{T}, y::AbstractVector, m::Int, k::Int, seed::Int) where T
    d, n = size(X)
    @assert length(y) == n "Number of labels does not match number of data points."
    
    # Compute random projection matrix
    P = random_projection_matrix(m,d; seed)

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