# Begin module
module FliesClassifiers

# Dependencies
using LinearAlgebra
using SparseArrays
using StatsBase
using Random
using .Threads

# Files inclusion
include("types.jl")
include("utils.jl")
include("FlyNN.jl")
include("EaS.jl")

# Export of public types and functions
export AbstractFliesClassifier, FlyNN, EaS
export fit, predict

end # module
