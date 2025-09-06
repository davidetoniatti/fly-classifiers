using MLUtils: flatten
using MLDatasets, StatsBase, MultivariateStats
using OpenML

include("grid_search.jl")

# Data Preparation
table = OpenML.load(14) |> DataFrame
            
y = table[:, end]
X = transpose(Matrix(table[:, 1:end-1]))

# Standardization
dt = StatsBase.fit(ZScoreTransform, X; dims=1)
X = StatsBase.transform(dt, X)

# PCA
# M = MultivariateStats.fit(PCA, X; maxoutdim=20)
# X = MultivariateStats.predict(M, X)

seed = 42
k_folds = 5
d, n = size(X)


#-- EaS HP TUNING (with multiple projection types) --#
eas_param_grid = Dict{Symbol, Vector}(
    :projection_type => [RandomBinaryProjectionMatrix, RandomUniformProjectionMatrix],
    :m => round.(Int, exp.(range(log(64d), stop=log(2048d), length=5))),
    :k => [8, 64, 128, 256],
    :s_ratio => [0.1, 0.3]  # Only used when projection_type is RandomBinaryProjectionMatrix
)

best_eas_params = grid_search(EaS, X, y, k_folds, seed, eas_param_grid)


#-- FlyNN HP TUNING (with multiple projection types) --#
# I've consolidated the different FlyNN grids into one for this example.
flynn_param_grid = Dict{Symbol, Vector}(
    :projection_type => [RandomBinaryProjectionMatrix, RandomUniformProjectionMatrix],
    :m => unique(round.(Int, exp.(range(log(64d), stop=log(2048d), length=5)))),
    :s_ratio => [0.1, 0.3],
    :k => [8, 64, 128, 256],
    :γ => [0.5, 0.65, 0.8]
)

best_flynn_params = grid_search(FlyNN, X, y, k_folds, seed, flynn_param_grid, "data/results_flynn.csv")


#-- FINAL SUMMARY --#
println("\n\nFINAL SUMMARY:")
println("Best parameters for EaS:", best_eas_params)
println("Best parameters for FlyNN:", best_flynn_params)