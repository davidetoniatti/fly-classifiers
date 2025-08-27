import Base: show

"""
    sbpm(d::Int, m::Int, s::Int, seed::Int) ->
    SparseMatrixCSC{Bool, Int}

Creates a sparse binary projection matrix of size `d x m`, where each column has
exactly `s` non-zero elements, placed in random rows.

# Arguments
- `d::Int`: Number of rows.
- `m::Int`: Number of columns.
- `s::Int`: Number of non-zero elements per columns.
- `seed::Int`: Seed for initializing the random number generators.

# Returns
- `SparseMatrixCSC{Bool, Int}`: The generated sparse binary matrix.
"""
function sbpm(d::Int, m::Int, s::Int, seed::Int)
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
        end_pos = i * s

        idxs = buf[tid]

        # Sampling.
        sample!(1:d, idxs; replace=false)

        # Fill the row and column index vectors.
        @inbounds row_idx[start_pos:end_pos] .= idxs
        @inbounds col_idx[start_pos:end_pos] .= i
    end

    # The values are all `true`, so we can construct the sparse matrix directly.
    return sparse(row_idx, col_idx, true, d, m)
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
- `ρ::Int`: Number of nonzeros per column (the "top-ρ").

# Returns
- `SparseMatrixCSC{Bool, Int}`: The sparse FlyHash matrix (m x n).
"""
function fly_hash(X::AbstractMatrix, M::SparseMatrixCSC{Bool,Int}, ρ::Int)
    d_X, n = size(X)
    d_M, m = size(M)

    @assert d_X == d_M "Dimension mismatch: X has $d_X rows, M has $d_M rows."

    # Determine the computation type based on the element types of X and M.
    T = promote_type(eltype(X), eltype(M))

    # Thread-local storage to avoid race conditions and allocations.
    # Each thread gets its own temporary vector `x_proj` for intermediate results.
    x_proj_local = [Vector{T}(undef, m) for _ in 1:Threads.nthreads()]
    top_idxs_local = [Vector{Int}(undef, ρ) for _ in 1:nthreads()]
    top_vals_local = [Vector{T}(undef, ρ) for _ in 1:nthreads()]

    nnz = n * ρ
    row_idx = Vector{Int}(undef, nnz)
    col_idx = Vector{Int}(undef, nnz)

    @threads for i in 1:n
        tid = threadid()
        start_pos = (i - 1) * ρ + 1
        end_pos = i * ρ

        # Get the current thread's temporary vector.
        x_proj = x_proj_local[tid]
        top_idxs = top_idxs_local[tid]
        top_vals = top_vals_local[tid]

        # In-place multiplication: x_proj = M' * X[:, i]
        manual_mul_transpose!(x_proj, M, X[:, i])

        # Find the indices of the ρ largest values.
        _topk_indices!(top_idxs, top_vals, x_proj, ρ)

        @inbounds row_idx[start_pos:end_pos] .= top_idxs
        @inbounds col_idx[start_pos:end_pos] .= i
    end

    return sparse(row_idx, col_idx, true, m, n)
end

"""
    manual_mul_transpose!(y, M, x) -> y

Performs an optimized in-place transposed matrix-vector multiplication `y = M' * x`.

This function is specifically tailored for cases where `M` is a `SparseMatrixCSC` with
`Bool` values. It leverages the sparse CSC (Compressed Sparse Column) structure for
efficient iteration over non-zero elements. Since the matrix values are boolean (effectively 1 for
non-zero entries), the dot product for each row of `M'` (column of `M`) simplifies to
a direct sum of the corresponding elements in `x`, avoiding actual multiplications.

# Arguments
- `y::Vector{T}`: The output vector to store the result. Its length must be equal to the number of columns in `M`.
- `M::SparseMatrixCSC{Bool, Int}`: The sparse boolean matrix.
- `x::AbstractVector{T}`: The vector to be multiplied.
"""
@inline function manual_mul_transpose!(y::Vector{T}, M::SparseMatrixCSC{Bool,Int}, x::AbstractVector{T}) where T
    # Extract the internal fields of the sparse matrix for direct and fast access.
    colptr = M.colptr
    rowval = M.rowval
    n_cols_M = size(M, 2)

    # Ensure the output vector has the correct dimensions before proceeding.
    @assert length(y) == n_cols_M "Output vector length must match the number of columns in the matrix."

    # Loop over each 'j' of the matrix M. This corresponds to calculating
    # the j-th element of the result vector 'y', which is the dot product of
    # the j-th column of M with the vector x.
    @inbounds for j in 1:n_cols_M
        # Initialize an accumulator for the dot product result.
        sum_val = zero(T)

        # Find the range of indices in `rowval` for the non-zero elements in column 'j'.
        start_idx = colptr[j]
        end_idx = colptr[j+1] - 1

        # For each non-zero element in the column, add the corresponding value from 'x' to the sum.
        # Since M is a boolean matrix, the dot product `dot(M[:, j], x)` simplifies
        # to `sum(x[i] for i in non-zero-rows-of-j)`.
        @simd ivdep for k in start_idx:end_idx
            # `rowval[k]` gives the row index of a non-zero element.
            sum_val += x[rowval[k]]
        end

        # Store the final sum in the j-th position of the output vector.
        y[j] = sum_val
    end
end


"""
    fit(::Type{FlyNN}, X::AbstractMatrix, y::AbstractVector, m::Int, ρ::Int, s::Int,
    γ::Real, seed::Int) -> FlyNN

Trains the FlyNN classifier.

# Arguments
- `::Type{FlyNN}`: The model to be fitted.
- `X::AbstractMatrix`: Training data matrix (d x n).
- `y::AbstractVector`: Training labels (n-element vector).
- `m::Int`: The dimension of the projection space.
- `ρ::Int`: The number of nonzeros in the FlyHash.
- `s::Int`: The number of nonzeros per column in the projection matrix.
- `γ::Real`: The decay rate parameter.
- `seed::Int`: Random seed for reproducibility.

# Returns
- `FlyNN`: The trained model containing the projection matrix, weights,
and class map.
"""
function fit(::Type{FlyNN}, X::AbstractMatrix, y::AbstractVector, m::Int, ρ::Int, s::Int, γ::Real, seed::Int)
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

    # Apply the final weight transformation with special handling for γ = 0.
    W_final = if γ == 0
        # If γ is 0, update weights to 1.0 where are 0, and to 0.0 otherwise.
        @. float(W_counts == 0)
    else
        λ = log1p(-γ)  # = log(1-γ)
        @. exp(λ * float(W_counts))
    end

    return FlyNN(M, W_final, ρ, class_labels)
end

"""
    predict(model::FlyNN, X::AbstractMatrix) -> Vector

Performs inference on new data using a trained FlyNN model.

# Arguments
- `model::FlyNN`: The trained FlyNN model object.
- `X::AbstractMatrix`: The data matrix (d x n).

# Returns
- `Vector`: A vector of predicted labels for each column in `X`.
"""
function predict(model::FlyNN, X::AbstractMatrix)
    H = fly_hash(X, model.M, model.ρ)

    fX = model.W * H

    l, n = size(fX)
    y_pred = Vector{eltype(model.class_labels)}(undef, n)
    nties = 0

    winner_buf = Vector{Int}(undef, l)

    @inbounds for i in 1:n
        min_val = typemax(eltype(fX))
        win_count = 0

        for j in 1:l
            val = fX[j, i]
            if val < min_val
                min_val = val
                win_count = 1
                winner_buf[1] = j
            elseif val == min_val
                win_count += 1
                winner_buf[win_count] = j
            end
        end

        # Tie-break
        winner_label = if win_count > 1
            nties += 1
            model.class_labels[winner_buf[rand(1:win_count)]]
        else
            model.class_labels[winner_buf[1]]
        end

        y_pred[i] = winner_label
    end

    return y_pred
end


"""
    show(io::IO, ::MIME"text/plain", model::FlyNN)

Defines the multi-line, pretty-printing for a FlyNN model (rich display).
"""
function show(io::IO, ::MIME"text/plain", model::FlyNN)
    println(io, "FlyNN Classifier")
    println(io, "├─ Projection (M): $(join(size(model.M), '×')) $(typeof(model.M))")
    println(io, "├─ Weights (W):    $(join(size(model.W), '×')) $(typeof(model.W))")
    println(io, "├─ Nonzeros (ρ):  $(model.ρ)")
    println(io, "└─ Classes:        $(length(model.class_labels)) labels of type $(eltype(model.class_labels))")
end

"""
    show(io::IO, model::FlyNN)

Defines the compact, single-line printing for a FlyNN model.
"""
function show(io::IO, model::FlyNN)
    n_classes = length(model.class_labels)
    print(io, "FlyNN($(n_classes) classes, ρ=$(model.ρ))")
end