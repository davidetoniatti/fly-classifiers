using MLUtils: MLUtils, flatten, mapobs, splitobs
using MLDatasets, BenchmarkTools
using MultivariateStats
using StatsBase

include("../classification_report.jl")

# Data Preparation
X_train, y_train = MNIST(split=:train)[:]
X_test, y_test = MNIST(split=:test)[:]

X_train = flatten(X_train)
X_test = flatten(X_test)

# Standardization
dt = StatsBase.fit(ZScoreTransform, X_train; dims=1)
X_train = StatsBase.transform(dt, X_train)

dt = StatsBase.fit(ZScoreTransform, X_test; dims=1)
X_test = StatsBase.transform(dt, X_test)

# PCA
M = MultivariateStats.fit(PCA, X_train; maxoutdim=20)

X_train = MultivariateStats.predict(M, X_train)
X_test = MultivariateStats.predict(M, X_test)

# Training
seed = 42

d = size(X_train)[1]
m = 50_000 # projection dimensionality

s = 5

ρ = floor(Int64, d*log(ℯ,m))
γ = 0.8

# model = FliesClassifiers.fit(EaS, X_train,y_train,m,ρ,s,γ,seed)
# y_pred = FliesClassifiers.predict(model, X_test);

# model = FliesClassifiers.fit(EaS, X_train, y_train, m, ρ, seed)
# y_pred = FliesClassifiers.predict(model, X_test);

# @btime FlyNN.fit(X_train,y_train,m,ρ,s,γ,seed)
# @profview fit(X_train,y_train,m,ρ,s,γ,seed)
# @btime predict(X_test, model);