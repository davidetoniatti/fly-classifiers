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