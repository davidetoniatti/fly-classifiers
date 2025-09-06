using MLUtils: MLUtils, flatten, mapobs, splitobs
using MLDatasets, BenchmarkTools
using MultivariateStats
using StatsBase
using OpenML, DataFrames, CategoricalArrays, FliesClassifiers

include("../classification_report.jl")

df = OpenML.load(14) |> DataFrame

target_col_index = findfirst(col -> col isa CategoricalArray, eachcol(df))
y = df[!, target_col_index]
X = transpose(Matrix(select(df, Not(target_col_index))))

dt = StatsBase.fit(ZScoreTransform, X, dims=2)
X = StatsBase.transform(dt, X)

# Split train/test (80% - 20%)
train, test = splitobs((X, y); at = 0.8, shuffle = true)
X_train, y_train = train
X_test, y_test = test;

# Training
seed = 42

d = size(X_train)[1]
m = 50_000 # projection dimensionality

s = 10
k = 128
γ = 0.9

B = RandomBinaryProjectionMatrix(m,d,s;seed)

model = FliesClassifiers.fit(FlyNN, X_train, y_train, B, k, γ)
y_pred = FliesClassifiers.predict(model, X_test);