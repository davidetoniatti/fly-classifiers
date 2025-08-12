include("FlyHashNaive.jl")
using LinearAlgebra, .Threads

function flynn_train_naive(X::AbstractMatrix, y::AbstractVector, m::Int, ρ::Real, s::Int, γ::Real, seed::Int)
    d, n = size(X)
    
    unique_classes = unique(y)
    l = length(unique_classes)
    class_map = Dict(label => i for (i, label) in enumerate(unique_classes))

    M = random_binary_matrix(m, d, s, seed)
    H = fly_hash(X, M, ρ)

    W = zeros(m, l)

    for (features, label) in zip(eachcol(H), y)
        W[:,class_map[label]] .+= features
    end

    W = (1-γ) .^ W

    return M, W
end

function flynn_infer(X,M,ρ,W) 
    H = fly_hash(X, M, ρ)
    fX = transpose(W) * Int.(H)
    classes = unique(y)
    
    n = size(fX,2)
    
    min_bf_scores = mapslices(minimum, fX; dims=1)[:]
    
    y_pred = Vector{eltype(y)}()
    nties = 0
    
    for i in 1:n
        fx = fX[:, i]
        y_set = classes[fx .== min_bf_scores[i]]
        
        if length(y_set) > 1
            nties += 1
            l = y_set[rand(1:length(y_set))]
        else
            l = y_set[1]
        end
        
        push!(y_pred, l)
    end
    
    return y_pred
end
