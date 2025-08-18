module FlyNN

# Dependencies
using LinearAlgebra
using SparseArrays
using StatsBase
using Random
using .Threads

# Files inclusion
include("types.jl")
include("FlyHashFast.jl")
include("training.jl")
include("inference.jl")
include("utils.jl")

# Export of public types and functions
export FNN, fit, predict

end # module