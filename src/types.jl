"""
    AbstractFliesClassifier

Supertype for FlyNN and EaS classifiers.
"""
abstract type AbstractFliesClassifier end

"""
    FlyNN

A struct to hold the components of a trained FlyNN model.
"""
struct FlyNN{T} <: AbstractFliesClassifier
    M::SparseMatrixCSC{Bool,Int}
    W::Matrix{Float64}
    ρ::Int
    class_labels::Vector{T}
end

"""
    EaS

A struct to hold the components of a trained EaS model.
"""
struct EaS <: AbstractFliesClassifier
    P::Matrix{Float64}
    W::Matrix{Float64}
    ct::Vector{Int64}
    k::Int
    class_labels::Vector
end