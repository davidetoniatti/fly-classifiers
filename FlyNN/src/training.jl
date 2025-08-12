"""
    flynn_train(X::AbstractMatrix, y::AbstractVector, m::Int, ρ::Int, s::Int,
    γ::Real, seed::Int) -> FlynnModel

Trains the Flynn classifier.

# Arguments
- `X::AbstractMatrix`: Training data matrix (d x n).
- `y::AbstractVector`: Training labels (n-element vector).
- `m::Int`: The dimension of the hash space.
- `ρ::Int`: The number of active hashes per item.
- `s::Int`: The number of non-zero elements per row in the projection matrix.
- `γ::Real`: The learning rate parameter.
- `seed::Int`: Random seed for reproducibility.

# Returns
- `FlynnModel`: The trained model containing the projection matrix, weights,
and class map.
"""
function fit(X::AbstractMatrix, y::AbstractVector, m::Int, ρ::Int, s::Int, γ::Real, seed::Int)
    d, n = size(X)
    
    # Robust mapping of class labels to integer indices (1, 2, ..., l).
    # This makes the code work with non-numeric or non-sequential labels.
    class_labels = unique(y)
    l = length(class_labels)
    class_map = Dict(label => i for (i, label) in enumerate(class_labels))

    # FyHash.
    M = random_binary_matrix(m, d, s, seed)
    H = fly_hash(X, M, ρ)
    
    # Safe parallelization with thread-local storage for weights.
    # We initialize a weight matrix for each thread to prevent race conditions.
    num_threads = Threads.nthreads()
    W_local = [zeros(m, l) for _ in 1:num_threads]

    @threads for i in 1:n  # Iterate over columns (data points).
        tid = threadid()
        class_idx = class_map[y[i]]
        
        # Directly update weights using sparse indices.
        # We get the non-zero row indices for column `i` of `H` and increment
        # the corresponding weights directly.
        nz_range = H.colptr[i]:(H.colptr[i+1] - 1)
        feature_indices = @view H.rowval[nz_range]
        
        @inbounds W_local[tid][feature_indices, class_idx] .+= 1
    end

    # Reduction (combination) of results.
    # Once all threads are finished, we combine their local weight matrices.
    W_final = reduce(+, W_local)
    
    # Apply the final weight transformation.
    W_final = (1 - γ) .^ W_final

    return FNN(M, W_final, ρ, class_labels)
end