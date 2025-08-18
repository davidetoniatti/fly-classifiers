using CSV, DataFrames, MLDatasets
using BenchmarkTools
using MLUtils: flatten, mapobs

# Dataset loading
df = CSV.read("/home/davide/repos/eas-distributed/data/wind.csv", DataFrame)
X = transpose(Matrix(df[:, 1:end-1]))
y = Vector(df[:, end])

println("Dataset loaded.")

# Split train/test (80% - 20%)
train, test = splitobs((X, y); at = 0.8, shuffle = true)

X_train, y_train = train
X_test, y_test = test;


seed = 42

d = size(X)[1]
m = 50_000 # projection dimensionality

s = 5

ρ = 128
γ = 0.8

model = fit(X_train,y_train,m,ρ,s,γ,seed)
y_pred = predict(X_test, model);

@btime fit(X_train,y_train,m,ρ,s,γ,seed)
#@profview fit(X_train,y_train,m,ρ,s,γ,seed)
# @btime predict(X_test, model);