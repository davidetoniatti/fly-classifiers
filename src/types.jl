"""
    FNN

A struct to hold the components of a trained FNN model.
"""
struct FNN{T}
    M::SparseMatrixCSC{Bool,Int}
    W::Matrix{Float64}
    ρ::Int
    class_labels::Vector{T}
end