module FlyNN

# Dependencies
using LinearAlgebra
using SparseArrays
using StatsBase
using Random
using .Threads

# Files inclusion
include("types.jl")
include("FlyHash.jl")
include("training.jl")
include("inference.jl")

# Export of public types and functions
export FNN, fit, predict

end