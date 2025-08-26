using MLUtils: flatten
using MLDatasets, StatsBase, MultivariateStats

include("grid_search.jl")

# Data Preparation
X, y = MNIST()[:]
X= flatten(X)

# Standardization
dt = StatsBase.fit(ZScoreTransform, X; dims=1)
X= StatsBase.transform(dt, X)

# PCA
M = MultivariateStats.fit(PCA, X; maxoutdim=20)
X= MultivariateStats.predict(M, X)

seed = 42
k_folds = 5

d, n = size(X)


#-- EAS HP TUNING --#
eas_param_grid = Dict{Symbol, Vector}(
    :m => round.(Int, exp.(range(log(64d), stop=log(4096d), length=10))),
    :k => [8, 64, 128, 256]
)

best_eas_params = grid_search(EaS, X, y, k_folds, seed, eas_param_grid, "data/results_eas.csv")


#-- FLYNN HP TUNING --#
flynn_grid_m = Dict{Symbol, Vector}(
    :m       => unique(round.(Int, exp.(range(log(4d), stop=log(4096d), length=10)))),
    :s_ratio => [0.1, 0.3],
    :ρ       => [8, 32],
    :γ       => [0.0, 0.5]
)

flynn_grid_s_ratio = Dict{Symbol, Vector}(
    :m       => [256d, 1024d],
    :s_ratio => round.(collect(range(0.1, stop=0.8, length=10)); digits = 3),
    :ρ       => [8, 32],
    :γ       => [0.0, 0.5]
)

flynn_grid_ρ = Dict{Symbol, Vector}(
    :m       => [256d, 1024d],
    :s_ratio => [0.1, 0.3],
    :ρ       => unique(round.(Int, exp.(range(log(4), stop=log(256), length=10)))),
    :γ       => [0.0, 0.5]
)

flynn_grid_γ = Dict{Symbol, Vector}(
    :m       => [256d, 1024d],
    :s_ratio => [0.1, 0.3],
    :ρ       => [8, 32],
    :γ       => vcat([0.0], round.(collect(range(0.1, stop=0.8, length=10)); digits = 3))
)

best_params_m = grid_search(FlyNN, X, y, k_folds, seed, flynn_grid_m, "data/results_flynn_m.csv")
best_params_s = grid_search(FlyNN, X, y, k_folds, seed, flynn_grid_s_ratio, "data/results_flynn_s.csv")
best_params_ρ = grid_search(FlyNN, X, y, k_folds, seed, flynn_grid_ρ, "data/results_flynn_rho.csv")
best_params_γ = grid_search(FlyNN, X, y, k_folds, seed, flynn_grid_γ, "data/results_flynn_gamma.csv")

all_results = [best_params_m, best_params_s, best_params_ρ, best_params_γ]

best_overall_params = all_results[argmax(d[:accuracy] for d in all_results)]

#-- FINAL SUMMARY --#
println("\n\nFINAL SUMMARY:")
println("Best parameters for EaS:", best_eas_params)
println("Best parameters for FlyNN:", best_overall_params)