push!(LOAD_PATH, "./FlyNN/src/")

using FlyNN
using CSV, DataFrames, MLUtils
using BenchmarkTools
using StatisticalMeasures

# Dataset loading
df = CSV.read("/home/davide/repos/eas-distributed/data/mnist.csv", DataFrame)
X = transpose(Matrix(df[:, 1:end-1]))
y = Vector(df[:, end])

println("X size: ", size(X))
println("y length: ", length(y))

# Split train/test (80% - 20%)
train, test = splitobs((X, y), at = 0.8)

X_train, y_train = train
X_test, y_test = test;

d = size(X)[1]
m = 50_000 # projection dimensionality

s = 5

ρ = 128
γ = 0.8

model = fit(X_train,y_train,m,ρ,s,γ,42)
y_pred = predict(X_test, model)

precision_score = StatisticalMeasures.precision(y_pred, y_test)
recall_score = recall(y_pred, y_test)
f1_score = f1score(y_pred, y_test)
accuracy_score = accuracy(y_pred, y_test)
println("Precision: ", precision_score)
println("Recall: ", recall_score)
println("F1-score: ", f1_score)
println("Accuracy: ", accuracy_score)

pop!(LOAD_PATH)