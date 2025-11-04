using MLUtils: MLUtils, flatten, mapobs, splitobs, shuffleobs, kfolds
using MLDatasets, BenchmarkTools
using MultivariateStats
using StatsBase
using OpenML, DataFrames, CategoricalArrays, FliesClassifiers

include("../classification_report.jl")

df = OpenML.load(40497) |> DataFrame

target_col_index = findfirst(col -> eltype(col) <: CategoricalValue, eachcol(df))
y = df[!, target_col_index]
X = transpose(Matrix(select(df, Not(target_col_index))))

folds = kfolds(shuffleobs((X, y)), k=5)

for (i, (train_data, test_data)) in enumerate(folds)
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    dt = StatsBase.fit(ZScoreTransform, X_train; dims=2)
    dt.scale .+= eps()

    X_train = StatsBase.transform(dt, X_train)
    replace!(X_train, NaN => 0)

    X_test = StatsBase.transform(dt, X_test)
    replace!(X_test, NaN => 0)
    println("norm ok")


    # Training
    seed = 42

    d = size(X_train)[1]
    m = 8*d # projection dimensionality

    s = max(1, round(Int, 0.1 * d))
    k = 128
    γ = 0.9

    P = RandomUniformProjectionMatrix(m,d ;seed)

    model = FliesClassifiers.fit(FlyNN, X_train, y_train, P, k, γ);
    FliesClassifiers.predict(model, X_test)
end
#println("Start benchmark")
#@elapsed FliesClassifiers.fit(EaS, X_train, y_train, P, k);