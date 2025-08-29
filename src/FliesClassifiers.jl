# Begin module
module FliesClassifiers

# Dependencies
using LinearAlgebra
using SparseArrays
using StatsBase
using Random
using .Threads

abstract type AbstractProjectionMatrix{T} <: AbstractMatrix{T} end
include("projections/RandomBinaryProjectionMatrix.jl")
include("projections/RandomUniformProjectionMatrix.jl")

# Files inclusion
include("types.jl")
include("utils.jl")
include("FlyNN.jl")
include("EaS.jl")

# Export of public types and functions
export AbstractFliesClassifier, FlyNN, EaS
export AbstractProjectionMatrix, RandomBinaryProjectionMatrix, RandomUniformProjectionMatrix
export fit, predict

end # module