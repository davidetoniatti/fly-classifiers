abstract type AbstractFliesClassifier end

"""
    FNN

A struct to hold the components of a trained FNN model.
"""
struct FlyNN{T} <: AbstractFliesClassifier
    M::SparseMatrixCSC{Bool,Int}
    W::Matrix{Float64}
    ρ::Int
    class_labels::Vector{T}
end

struct EaS <: AbstractFliesClassifier
    P::Matrix{Float64}
    w::Vector{Float64}
    ct::Vector{Int64}
    k::Int
end