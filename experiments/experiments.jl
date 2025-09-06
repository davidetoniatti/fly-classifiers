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
using Plots
using ProgressMeter
using StatsBase
using Tables
using CategoricalArrays
using LinearAlgebra
using Random
using SparseArrays
using LaTeXStrings 

# Including local modules
include("../classification_report.jl")
include("../src/FliesClassifiers.jl")
using .FliesClassifiers

# Number of folds for cross-validation
const K_FOLDS = 3 

println("Julia environment configured successfully.")

"""
Generates plots to compare model performance (for Vision experiments).
"""
function generate_vision_plots(df::DataFrame, exp::Int)
    model_names = names(df)[end-3:end]
    dataset_names = unique(df.DatasetName)

    for name in dataset_names
        println("Generating plot for dataset: $name")

        subset_df = filter(row -> row.DatasetName == name, df)
        sort!(subset_df, :nnz, rev=true)
        x_positions = 1:nrow(subset_df)
        exponents = Int.(round.(-log2.(subset_df.nnz)))
        x_labels = [L"2^{-%$e}" for e in exponents]

        p = plot(
            title = name,
            xlabel = "Sparsity (nnz)",
            ylabel = "Accuracy (%)",
            legend = :bottomright,
            xticks = (x_positions, x_labels),
            minorgrid = true
        )

        for model in model_names
            plot!(p,
                x_positions,
                subset_df[!, model] .* 100,
                label = model,
                marker = :circle,
                linewidth = 2
            )
        end
        
        filename = "plots/vision/accuracies_m_$(exp)d_$(name).png"
        savefig(p, filename)
    end

end


"""
Generates scatter plots to compare model performance (for OpenML experiments).
"""
function generate_scatter_plots(results_df::DataFrame, exp::Int)
    println("\nCreating scatter plots for m = $(exp)d...")
    !isdir("plots") && mkdir("plots")

    if nrow(results_df) < 1
        @warn "Results DataFrame is empty for m=$(exp)d. Skipping plot generation."
        return
    end

    comparisons = [
        (:FlyNN_Binary, :FlyNN_Uniform, "FlyNN: Binary vs Uniform"),
        (:EaS_Binary, :EaS_Uniform, "EaS: Binary vs Uniform"),
        (:FlyNN_Binary, :EaS_Binary, "Binary Proj: FlyNN vs EaS"),
        (:FlyNN_Uniform, :EaS_Uniform, "Uniform Proj: FlyNN vs EaS"),
        (:FlyNN_Binary, :EaS_Uniform, "FlyNN Binary vs EaS Uniform"),
        (:FlyNN_Uniform, :EaS_Binary, "FlyNN Uniform vs EaS Binary")
    ]

    model_cols = select(results_df, Not(:DatasetName))
    min_acc, max_acc = extrema(Matrix(model_cols))
    
    plot_margin = (max_acc - min_acc) * 0.05
    lims = (min_acc - plot_margin, max_acc + plot_margin)

    for (x_col, y_col, title_str) in comparisons
        plt = scatter(
            results_df[!, x_col],
            results_df[!, y_col],
            xlabel=string(x_col),
            ylabel=string(y_col),
            title="$title_str (m=$(exp)d)",
            legend=false,
            aspect_ratio=1,
            xlims=lims,
            ylims=lims,
            markersize=5,
            markerstrokewidth=0,
            alpha=0.6
        )
        plot!(plt, [lims[1], lims[2]], [lims[1], lims[2]], style=:dash, color=:red, label="")

        filename = "plots/comparison_$(x_col)_vs_$(y_col)_m_$(exp)d.png"
        savefig(plt, filename)
    end
    println("Scatter plots for m = $(exp)d saved to 'plots/' directory.")
end


"""
Performs a single cross-validation run for a given model and configuration.
"""
function cross_validation(model_type, proj_type, X, y, m, k, s, γ)
    d = size(X, 1)
    accuracies = Float64[]
    
    # `shuffleobs` is safer than `shuffle` to maintain the X-y correspondence
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
            # `γ` is only used by FlyNN
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
Runs experiments on a collection of datasets from OpenML.
"""
function openml_datasets_experiments()
    # Hyperparameters
    exp_values = [2^i for i in 3:8]
    nnz_values = [2.0^-i for i in 1:8]
    gamma = 0.9
    s_ratio = 0.1

    # List of dataset IDs from OpenML
    dataset_ids = [14, 16, 22, 28, 44, 60, 182, 715, 718, 722, 723, 734, 752, 761,
                   797, 806, 833, 837, 849, 866, 903, 904, 917, 971, 979, 980, 995, 1020,
                   1049, 1050, 1067, 1068, 1444, 1453, 1466, 1475, 1479, 1487, 1494, 1496,
                   1497, 1504, 1507, 1547, 1560, 1566, 4538, 40497, 40499, 40900, 40982,
                   41146, 41156, 41671, 41946, 12, 300, 978, 1038, 1041, 1042, 1468, 1485,
                   1501, 40666, 40910, 40979, 41082, 41144, 41145, 41158]
    println("Found $(length(dataset_ids)) datasets to process from the OpenML list.")

    for exp in exp_values
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
                
                # Find the target column (the first Categorical one)
                target_col_index = findfirst(col -> eltype(col) <: CategoricalValue, eachcol(df))
                y = df[!, target_col_index]
                X = transpose(Matrix(select(df, Not(target_col_index))))

                d = size(X, 1)
                s = max(1, round(Int, s_ratio * d))
                m = exp * d

                # For each dataset, we calculate the mean accuracy over all nnz values
                accs = Dict(
                    :FlyNN_Binary => Float64[], :FlyNN_Uniform => Float64[],
                    :EaS_Binary => Float64[], :EaS_Uniform => Float64[]
                )

                for nnz in nnz_values
                    k = ceil(Int, nnz * m)
                    push!(accs[:FlyNN_Binary], cross_validation(FlyNN, RandomBinaryProjectionMatrix, X, y, m, k, s, gamma))
                    push!(accs[:FlyNN_Uniform], cross_validation(FlyNN, RandomUniformProjectionMatrix, X, y, m, k, s, gamma))
                    push!(accs[:EaS_Binary], cross_validation(EaS, RandomBinaryProjectionMatrix, X, y, m, k, s, 0.0))
                    push!(accs[:EaS_Uniform], cross_validation(EaS, RandomUniformProjectionMatrix, X, y, m, k, s, 0.0))
                end

                push!(results_for_exp, (
                    dataset_name,
                    mean(accs[:FlyNN_Binary]), mean(accs[:FlyNN_Uniform]),
                    mean(accs[:EaS_Binary]), mean(accs[:EaS_Uniform])
                ))

            catch e
                @warn "Error while processing dataset '$dataset_name' (ID: $dataset_id): $e"
            end
            next!(progress)
        end

        if nrow(results_for_exp) > 0
            !isdir("results") && mkdir("results")
            CSV.write("results/accuracies_m_$(exp)d_openml.csv", results_for_exp)
            println("\nResults for m = $(exp)d saved to 'results/accuracies_m_$(exp)d_openml.csv'")
            generate_scatter_plots(results_for_exp, exp)
        else
            println("\nNo results obtained for m = $(exp)d.")
        end
    end
    println("\n\n--- OpenML processing complete! ---")
end


"""
Runs experiments on vision datasets (FashionMNIST, CIFAR10).
"""
function vision_datasets_experiments()
    # Hyperparameters
    exp_values = [2^i for i in 6:6]
    nnz_values = [2.0^-i for i in 1:8]
    gamma = 0.9
    s_ratio = 0.1

    # Definition of datasets to process
    datasets_to_process = [
    #    (name="Fashion MNIST 3v8", loader=FashionMNIST, classes=[3, 8]),
        (name="CIFAR10 2v5", loader=CIFAR10, classes=[2, 5])
    ]

    for exp in exp_values
        results_for_exp = DataFrame(
            DatasetName=String[], nnz=Float64[], k=Int[],
            FlyNN_Binary=Float64[], FlyNN_Uniform=Float64[],
            EaS_Binary=Float64[], EaS_Uniform=Float64[]
        )

        println("\nStarting processing for m = $(exp)d. Number of datasets: $(length(datasets_to_process)).")
        
        for config in datasets_to_process
            try
                # Data loading and preparation
                X_all, y_all = config.loader()[:]
                X_flat = MLUtils.flatten(X_all)
                
                mask = map(y -> y in config.classes, y_all)
                X = X_flat[:, mask]
                y = y_all[mask]

                # Z-score standardization
                dt = StatsBase.fit(ZScoreTransform, X; dims=2)
                X = StatsBase.transform(dt, X)
                replace!(X, NaN => 0) # Handles features with zero variance

                d = size(X, 1)
                s = max(1, round(Int, s_ratio * d))
                m = round(Int, exp * d)

                println("Processing $(config.name) with m=$m...")
                progress = Progress(length(nnz_values); desc="$(config.name): ")

                for nnz in nnz_values
                    k = ceil(Int, nnz * m)

                    acc_flynn_binary = cross_validation(FlyNN, RandomBinaryProjectionMatrix, X, y, m, k, s, gamma)
                    acc_flynn_uniform = cross_validation(FlyNN, RandomUniformProjectionMatrix, X, y, m, k, s, gamma)
                    acc_eas_binary = cross_validation(EaS, RandomBinaryProjectionMatrix, X, y, m, k, s, 0.0)
                    acc_eas_uniform = cross_validation(EaS, RandomUniformProjectionMatrix, X, y, m, k, s, 0.0)

                    push!(results_for_exp, (
                        config.name, nnz, k,
                        acc_flynn_binary, acc_flynn_uniform,
                        acc_eas_binary, acc_eas_uniform
                    ))
                    next!(progress)
                end
            catch e
                @warn "Error while processing dataset '$(config.name)': $e."
            end
        end

        if nrow(results_for_exp) > 0
            !isdir("results") && mkdir("results")
            CSV.write("results/accuracies_m_$(exp)d_vision.csv", results_for_exp)
            println("\nResults for m = $(exp)d saved to 'results/accuracies_m_$(exp)d_vision.csv'")
            generate_vision_plots(results_for_exp, exp)
        else
            println("\nNo results for m = $(exp)d.")
        end
    end
    println("\n\n--- Vision processing complete! ---")
end


# Script entry point
function main()
    if isempty(ARGS) || length(ARGS) > 1 || !(ARGS[1] in ["vision", "openml"])
        println("Usage: julia $(@__FILE__) [vision|openml]")
        println("\nAvailable arguments:")
        println("  vision: Runs experiments on FashionMNIST and CIFAR10.")
        println("  openml: Runs experiments on a collection of datasets from OpenML.")
        return
    end

    experiment = ARGS[1]

    if experiment == "vision"
        vision_datasets_experiments()
    elseif experiment == "openml"
        openml_datasets_experiments()
    end
end

# Run the main function
main()