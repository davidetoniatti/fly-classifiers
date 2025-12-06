import Base: show


"""
    fit(::Type{FlyNNA}, X::AbstractMatrix, y::AbstractVector, P::AbstractProjectionMatrix,
        k::Int) -> FlyNNA

Trains the FlyNNA classifier.

# Arguments
- `::Type{FlyNNA}`: The model to be fitted.
- `X::AbstractMatrix`: Training data matrix (d x n).
- `y::AbstractVector`: Training labels (n-element vector).
- `P::AbstractProjectionMatrix`: Random projection matrix (m x d).
- `k::Int`: The number of active response regions per item.

# Returns
- `FlyNNA`: The trained model containing the projection matrix and weights.
"""
function fit(::Type{FlyNNA}, X::AbstractMatrix{T}, y::AbstractVector, P::AbstractProjectionMatrix, k::Int) where T
    d_X, n = size(X)
    m, d_P = size(P)

    @assert d_X == d_P "Dimension mismatch: X has $d_X rows, M has $d_P rows."
    @assert length(y) == n "Number of labels does not match number of data points."

    # Robust mapping of class labels to integer indices (1, 2, ..., l).
    # This makes the code work with non-numeric or non-sequential labels.
    class_labels = unique(y)
    l = length(class_labels)
    class_map = Dict(label => i for (i, label) in enumerate(class_labels))

    # Determine the computation type based on the element types of X and P.
    # T_proj = promote_type(T, eltype(P.matrix))
    # 
    H = FlyHash(X, P, k).matrix

    # Safe parallelization with thread-local storage for weights and counters
    W_local = [zeros(Int, l, m) for _ in 1:nthreads()]
    ct_local = [zeros(Int, m) for _ in 1:nthreads()]
    
    

    # x_proj_local = [Vector{T_proj}(undef, m) for _ in 1:nthreads()]
    # top_idxs_local = [Vector{Int}(undef, k) for _ in 1:nthreads()]
    # top_vals_local = [Vector{T_proj}(undef, k) for _ in 1:nthreads()]

    @threads for i in 1:n
        tid = threadid()
        class_idx = class_map[@inbounds y[i]]

        @inbounds for j in H.colptr[i]:(H.colptr[i+1]-1)
            r = H.rowval[j]
            W_local[tid][class_idx, r] += 1
            ct_local[tid][r] += 1
        end
    end

    W_total = reduce(+, W_local)
    ct_total = reduce(+, ct_local)

    W_normalized = zeros(Float64, l, m)
    valid_indices = ct_total .> 0

    @inbounds for c in 1:l
        W_normalized[c, valid_indices] .= W_total[c, valid_indices] ./ ct_total[valid_indices]
    end

    return FlyNNA(P, W_normalized, ct_total, k, class_labels)
end

"""
    predict(model::FlyNNA, X::AbstractMatrix) -> Vector

Performs inference on new data using a trained FlyNNA model.
Ties are broken deterministically by selecting the first class with maximum score.

# Arguments
- `model::FlyNNA`: The trained FlyNNA model object.
- `X::AbstractMatrix`: The data matrix (d x n).

# Returns
- `Vector`: A vector of predictions for each column in `X`.
"""
function predict(model::FlyNNA, X::AbstractMatrix{T}) where T
    n = size(X, 2)
    l, m = size(model.W)
    
    y_pred = Vector{eltype(model.class_labels)}(undef, n)
    
    T_proj = promote_type(T, eltype(model.P.matrix))
    
    # Thread-local buffers
    x_proj_local = [Vector{T_proj}(undef, m) for _ in 1:nthreads()]
    top_idxs_local = [Vector{Int}(undef, model.k) for _ in 1:nthreads()]
    top_vals_local = [Vector{T_proj}(undef, model.k) for _ in 1:nthreads()]
    class_scores_local = [Vector{eltype(model.W)}(undef, l) for _ in 1:nthreads()]
    
    @threads for i in 1:n
        tid = threadid()
        
        x_proj = x_proj_local[tid]
        top_idxs = top_idxs_local[tid]
        top_vals = top_vals_local[tid]
        class_scores = class_scores_local[tid]
        
        x_view = @view X[:, i]
        mul!(x_proj, model.P, x_view)
        _topk_indices!(top_idxs, top_vals, x_proj, model.k)
        
        fill!(class_scores, 0)
        active_weights = @view model.W[:, top_idxs]
        sum!(class_scores, active_weights)
        
        # Deterministic tie-break: argmax returns the first maximum
        winner_idx = argmax(class_scores)
        
        @inbounds y_pred[i] = model.class_labels[winner_idx]
    end
    
    return y_pred
end


"""
    show(io::IO, ::MIME"text/plain", model::FlyNNA)

Defines the multi-line, pretty-printing for an FlyNNA model (rich display).
"""
function show(io::IO, ::MIME"text/plain", model::FlyNNA)
    println(io, "FlyNNA Classifier")
    println(io, "├─ Projection (P): $(join(size(model.P), '×')) $(typeof(model.P))")
    println(io, "├─ Weights (W):    $(join(size(model.W), '×')) $(typeof(model.W))")
    println(io, "├─ Activations (k): $(model.k)")
    println(io, "└─ Classes:        $(length(model.class_labels)) labels of type $(eltype(model.class_labels))")
end


"""
    show(io::IO, model::FlyNNA)

Defines the compact, single-line printing for an FlyNNA model.
"""
function show(io::IO, model::FlyNNA)
    n_classes = length(model.class_labels)
    print(io, "FlyNNA($(n_classes) classes, k=$(model.k))")
end
