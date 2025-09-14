"""
    FlyHash

A struct to hold the components of a FlyHash.
"""
struct FlyHash
    matrix::SparseMatrixCSC{Bool,Int}
end


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
    P::AbstractProjectionMatrix
    W::Matrix{Float64}
    k::Int
    class_labels::Vector{T}
end

"""
    EaS

A struct to hold the components of a trained EaS model.
"""
struct EaS <: AbstractFliesClassifier
    P::AbstractProjectionMatrix
    W::Matrix{Float64}
    ct::Vector{Int64}
    k::Int
    class_labels::Vector
end