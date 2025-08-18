"""
    _topk_indices!(top_idxs, top_vals, v, k) -> Vector{Int}

Finds the indices of the `k` largest values in a vector `v` and stores them in `top_idxs`.

This is an optimized, in-place function that avoids allocations by using pre-allocated
buffers. It maintains a buffer of the top `k` elements found so far. When it encounters
a new element larger than the current minimum in the buffer, it replaces that minimum
and then performs a linear scan of the small buffer to find the new minimum. This approach
is efficient for small `k`. In fact, it is much faster than partialsortperm!.

# Arguments
- `top_idxs::Vector{Int}`: A pre-allocated buffer of size `k` to store the resulting indices.
- `top_vals::Vector{T}`: A pre-allocated buffer of size `k` to store the corresponding values.
- `v::AbstractVector{T}`: The input vector to search for top elements.
- `k::Int`: The number of top elements to find.

# Returns
- `Vector{Int}`: The modified `top_idxs` buffer containing the indices of the top `k` values.
"""
@inline function _topk_indices!(top_idxs::Vector{Int}, top_vals::Vector{T}, v::AbstractVector{T}, k::Int) where T
    # Initialize the value buffer with the smallest possible value for type T.
    fill!(top_vals, typemin(T))
    # Initialize the index buffer with a placeholder (0 is a safe choice if indices are 1-based).
    fill!(top_idxs, 0)

    # These variables track the minimum value currently in our top-k buffer and its position.
    # This avoids having to search for the minimum every single iteration of the main loop.
    min_val_in_topk = typemin(T)
    min_pos_in_topk = 1

    # Iterate through each element of the input vector `v`.
    @inbounds for j in eachindex(v)
        val = v[j]

        # Check if the current value is larger than the smallest value in our top-k set.
        if val > min_val_in_topk
            # If it is larger, replace the smallest value in our buffer with this new value.
            top_idxs[min_pos_in_topk] = j
            top_vals[min_pos_in_topk] = val

            # We must find the new minimum for the next iteration's comparison.
            # We start the search by assuming the new minimum is the largest possible value.
            min_val_in_topk = typemax(T)

            # Perform a quick linear scan over the small `k`-sized buffer to find the new minimum.
            for i in 1:k
                if top_vals[i] < min_val_in_topk
                    min_val_in_topk = top_vals[i]
                    min_pos_in_topk = i
                end
            end
        end
    end

    return top_idxs
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