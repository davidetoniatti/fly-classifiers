using MLUtils: MLUtils, flatten, mapobs, splitobs
using MLDatasets, BenchmarkTools
using MultivariateStats, StatsBase
using Revise, FliesClassifiers

include("../classification_report.jl")

# Data Preparation
X, y = MNIST()[:]
X = flatten(X)

# Standardization
dt = StatsBase.fit(ZScoreTransform, X; dims=2)
X = StatsBase.transform(dt, X)
replace!(X, NaN => 0)

# Split train/test (80% - 20%)
train, test = splitobs((X, y); at = 0.8, shuffle = true)
X_train, y_train = train
X_test, y_test = test;

# PCA
M = MultivariateStats.fit(PCA, X_train; maxoutdim=20)
X_train = MultivariateStats.predict(M, X_train)
X_test = MultivariateStats.predict(M, X_test)

# Training
seed = 42

d = size(X_train)[1]
m = 50_000 # projection dimensionality

s = 5

k = 128
γ = 0.8

B = RandomBinaryProjectionMatrix(m, d, s)
U = RandomUniformProjectionMatrix(m, d)

model1 = FliesClassifiers.fit(FlyNN, X_train, y_train, B, k, γ)
model2 = FliesClassifiers.fit(FlyNN, X_train, y_train, U, k, γ)
model3 = FliesClassifiers.fit(EaS, X_train, y_train, U, k)
model4 = FliesClassifiers.fit(EaS, X_train, y_train, B, k)

y_pred = FliesClassifiers.predict(model1, X_test)
classification_report(y_test, y_pred)

y_pred = FliesClassifiers.predict(model2, X_test)
classification_report(y_test, y_pred)

y_pred = FliesClassifiers.predict(model3, X_test)
classification_report(y_test, y_pred)

y_pred = FliesClassifiers.predict(model4, X_test)
classification_report(y_test, y_pred)