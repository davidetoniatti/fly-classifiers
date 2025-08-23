using MLUtils: MLUtils, flatten, mapobs, splitobs
using MLDatasets, BenchmarkTools
using MultivariateStats

# Data Preparation
X_train, y_train = FashionMNIST(UInt8, split=:train)[:]
X_test, y_test = FashionMNIST(UInt8, split=:test)[:]

X_train = flatten(X_train)
X_test = flatten(X_test)

# PCA
M = MultivariateStats.fit(PCA, X_train; maxoutdim=20)

X_train = MultivariateStats.predict(M, X_train)
X_test = MultivariateStats.predict(M, X_test)

# Training
seed = 42

d = size(X)[1]
m = 50_000 # projection dimensionality

s = 5

ρ = 128
γ = 0.8

model = FlyNN.fit(X_train,y_train,m,ρ,s,γ,seed)
y_pred = FlyNN.predict(X_test, model);

@btime FlyNN.fit(X_train,y_train,m,ρ,s,γ,seed)
