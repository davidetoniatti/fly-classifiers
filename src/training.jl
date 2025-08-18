"""
    fit(X::AbstractMatrix, y::AbstractVector, m::Int, ρ::Int, s::Int,
    γ::Real, seed::Int) -> FNN

Trains the FlyNN classifier.

# Arguments
- `X::AbstractMatrix`: Training data matrix (d x n).
- `y::AbstractVector`: Training labels (n-element vector).
- `m::Int`: The dimension of the hash space.
- `ρ::Int`: The number of active hashes per item.
- `s::Int`: The number of non-zero elements per row in the projection matrix.
- `γ::Real`: The learning rate parameter.
- `seed::Int`: Random seed for reproducibility.

# Returns
- `FNN`: The trained model containing the projection matrix, weights,
and class map.
"""
function fit(X::AbstractMatrix, y::AbstractVector, m::Int, ρ::Int, s::Int, γ::Real, seed::Int)
    d, n = size(X)
    @assert length(y) == n "Number of labels does not match number of data points."

    # Robust mapping of class labels to integer indices (1, 2, ..., l).
    # This makes the code work with non-numeric or non-sequential labels.
    class_labels = unique(y)
    l = length(class_labels)
    class_map = Dict(label => i for (i, label) in enumerate(class_labels))

    # Compute the FlyHash
    M = sbpm(d, m, s, seed)
    H = fly_hash(X, M, ρ)

    # Safe parallelization with thread-local storage for weights.
    # We initialize a weight matrix for each thread to prevent race conditions.
    W_local = [zeros(Int32, l, m) for _ in 1:nthreads()]

    @threads for i in 1:n  # Iterate over columns (data points).
        tid = threadid()
        class_idx = class_map[@inbounds y[i]]

        # Directly update weights using sparse indices.
        # We get the non-zero row indices for column `i` of `H` and increment
        # the corresponding weights directly.
        @inbounds for k in H.colptr[i]:(H.colptr[i+1]-1)
            r = H.rowval[k]
            W_local[tid][class_idx, r] += 1
        end
    end

    # Reduction (combination) of results.
    # Once all threads are finished, we combine their local weight matrices.
    W_counts = reduce(+, W_local)

    # Apply the final weight transformation.
    λ = log1p(-γ)  # = log(1-γ)
    W_final = @. exp(λ * float(W_counts))

    return FNN(M, W_final, ρ, class_labels)
end