# Environment setup
# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()

using OpenML
using MLDatasets
using DataFrames
using CSV
using Statistics
using MLUtils
using ProgressMeter
using StatsBase
using Tables
using CategoricalArrays
using LinearAlgebra
using Random
using SparseArrays

# Including local modules
include("../classification_report.jl")
include("../src/FliesClassifiers.jl")
using .FliesClassifiers

const K_FOLDS = 5

# Struct to hold all hyperparameters
struct Hyperparameters
    exp_values::Vector{Int}
    nnz_values::Vector{Float64}
    gamma::Float64
    s_ratio::Float64
end

# Define configurations for each experiment type
const OPENML_CONFIG = Hyperparameters([2^i for i in 3:8], [2.0^-i for i in 1:8], 0.9, 0.1)
const VISION_CONFIG = Hyperparameters([2^i for i in 6:6], [2.0^-i for i in 1:8], 0.9, 0.1)

println("Julia environment for experiments configured successfully.")


"""
Performs a single cross-validation run for a given model and configuration.
"""
function cross_validation(model_type, proj_type, X, y, m, k, s, γ)
    d = size(X, 1)
    accuracies = Float64[]
    
    folds = kfolds(shuffleobs((X, y)), k=K_FOLDS)

    for (i, (train_data, test_data)) in enumerate(folds)
        X_train, y_train = train_data
        X_test, y_test = test_data

        P = if proj_type == RandomBinaryProjectionMatrix
            RandomBinaryProjectionMatrix(m, d, s; seed=42 + i)
        else # RandomUniformProjectionMatrix
            RandomUniformProjectionMatrix(m, d; seed=42 + i)
        end

        model = if model_type == FlyNN
            FliesClassifiers.fit(FlyNN, X_train, y_train, P, k, γ)
        else # EaS
            FliesClassifiers.fit(EaS, X_train, y_train, P, k)
        end

        y_pred = FliesClassifiers.predict(model, X_test)
        accuracy = mean(y_pred .== y_test)
        push!(accuracies, accuracy)
    end

    return round(mean(accuracies); digits=4)
end

"""
Runs the suite of four model configurations for a given dataset.
"""
function run_model_suite(X, y, m, k, s, gamma)
    acc_flynn_binary = cross_validation(FlyNN, RandomBinaryProjectionMatrix, X, y, m, k, s, gamma)
    acc_flynn_uniform = cross_validation(FlyNN, RandomUniformProjectionMatrix, X, y, m, k, s, gamma)
    acc_eas_binary = cross_validation(EaS, RandomBinaryProjectionMatrix, X, y, m, k, s, 0.0)
    acc_eas_uniform = cross_validation(EaS, RandomUniformProjectionMatrix, X, y, m, k, s, 0.0)
    return (acc_flynn_binary, acc_flynn_uniform, acc_eas_binary, acc_eas_uniform)
end


"""
Runs experiments on a collection of datasets from OpenML.
"""
function openml_datasets_experiments(results_dir::String)
    config = OPENML_CONFIG

    dataset_ids = [14, 16, 22, 28, 44, 60, 182, 715, 718, 722, 723, 734, 752, 761,
                   797, 806, 833, 837, 849, 866, 903, 904, 917, 971, 979, 980, 995, 1020,
                   1049, 1050, 1067, 1068, 1444, 1453, 1466, 1475, 1479, 1487, 1494, 1496,
                   1497, 1504, 1507, 1547, 1560, 1566, 4538, 40497, 40499, 40900, 40982,
                   41146, 41156, 41671, 41946, 12, 300, 978, 1038, 1041, 1042, 1468, 1485,
                   1501, 40666, 40910, 40979, 41082, 41144, 41145, 41158]
    println("Found $(length(dataset_ids)) datasets to process from the OpenML list.")

    for exp in config.exp_values
        results_for_exp = DataFrame(
            DatasetName=String[],
            FlyNN_Binary=Float64[], FlyNN_Uniform=Float64[],
            EaS_Binary=Float64[], EaS_Uniform=Float64[]
        )

        println("\nStarting processing for m = $(exp)d. Number of datasets: $(length(dataset_ids)).")
        progress = Progress(length(dataset_ids); desc="Processing datasets for m=$(exp)d: ")

        for dataset_id in dataset_ids
            dataset_name = "ID_$dataset_id"
            try
                dataset_info = OpenML.list_datasets(output_format=DataFrame, filter="data_id/$dataset_id")
                dataset_name = isempty(dataset_info) ? dataset_name : first(dataset_info.name)
                df = OpenML.load(dataset_id) |> DataFrame
                
                target_col_index = findfirst(col -> eltype(col) <: CategoricalValue, eachcol(df))
                y = df[!, target_col_index]
                X = transpose(Matrix(select(df, Not(target_col_index))))

                d = size(X, 1)
                s = max(1, round(Int, config.s_ratio * d))
                m = exp * d

                all_accs = [run_model_suite(X, y, m, ceil(Int, nnz * m), s, config.gamma) for nnz in config.nnz_values]

                # Average the accuracies across all nnz values
                avg_accs = (
                    mean(getindex.(all_accs, 1)), mean(getindex.(all_accs, 2)),
                    mean(getindex.(all_accs, 3)), mean(getindex.(all_accs, 4))
                )

                push!(results_for_exp, (dataset_name, avg_accs...))

            catch e
                @warn "Error while processing dataset '$dataset_name' (ID: $dataset_id): $e"
            end
            next!(progress)
        end

        if nrow(results_for_exp) > 0
            !isdir(results_dir) && mkpath(results_dir)
            output_path = joinpath(results_dir, "accuracies_m_$(exp)d_openml.csv")
            CSV.write(output_path, results_for_exp)
            println("\nResults for m = $(exp)d saved to '$output_path'")
        else
            println("\nNo results obtained for m = $(exp)d.")
        end
    end
    println("\n\n--- OpenML processing complete! ---")
end


"""
Runs experiments on vision datasets (FashionMNIST, CIFAR10).
"""
function vision_datasets_experiments(results_dir::String)
    config = VISION_CONFIG

    datasets_to_process = [
        (name="CIFAR10_2v5", loader=CIFAR10, classes=[2, 5])
        (name="FashionMNIST_3v8", loader=FashionMNIST, classes=[3, 8])
    ]

    for exp in config.exp_values
        results_for_exp = DataFrame(
            DatasetName=String[], nnz=Float64[], k=Int[],
            FlyNN_Binary=Float64[], FlyNN_Uniform=Float64[],
            EaS_Binary=Float64[], EaS_Uniform=Float64[]
        )

        println("\nStarting processing for m = $(exp)d. Number of datasets: $(length(datasets_to_process)).")
        
        for d_config in datasets_to_process
            try
                X_all, y_all = d_config.loader()[:]
                X_flat = MLUtils.flatten(X_all)
                
                mask = map(y -> y in d_config.classes, y_all)
                X = X_flat[:, mask]
                y = y_all[mask]

                dt = StatsBase.fit(ZScoreTransform, X; dims=2)
                X = StatsBase.transform(dt, X)
                replace!(X, NaN => 0)

                d = size(X, 1)
                s = max(1, round(Int, config.s_ratio * d))
                m = round(Int, exp * d)

                println("Processing $(d_config.name) with m=$m...")
                progress = Progress(length(config.nnz_values); desc="$(d_config.name): ")

                for nnz in config.nnz_values
                    k = ceil(Int, nnz * m)
                    accuracies = run_model_suite(X, y, m, k, s, config.gamma)
                    push!(results_for_exp, (d_config.name, nnz, k, accuracies...))
                    next!(progress)
                end
            catch e
                @warn "Error while processing dataset '$(d_config.name)': $e."
            end
        end

        if nrow(results_for_exp) > 0
            !isdir(results_dir) && mkpath(results_dir)
            output_path = joinpath(results_dir, "accuracies_m_$(exp)d_vision.csv")
            CSV.write(output_path, results_for_exp)
            println("\nResults for m = $(exp)d saved to '$output_path'")
        else
            println("\nNo results for m = $(exp)d.")
        end
    end
    println("\n\n--- Vision processing complete! ---")
end


# Script entry point
function main()
    if !(1 <= length(ARGS) <= 2) || !(ARGS[1] in ["vision", "openml"])
        println("Usage: julia $(@__FILE__) [vision|openml] [results_output_dir]")
        println("\nArguments:")
        println("  [vision|openml] : (Required) The type of experiment to run.")
        println("  [results_output_dir]: (Optional) Directory to save CSV results. Defaults to '../results'.")
        return
    end

    experiment = ARGS[1]
    results_dir = length(ARGS) == 2 ? ARGS[2] : joinpath("..", "results")

    println("Starting experiment: $experiment. Results will be saved to: $results_dir")

    if experiment == "vision"
        vision_datasets_experiments(results_dir)
    elseif experiment == "openml"
        openml_datasets_experiments(results_dir)
    end
end

main()