using Printf
using Statistics, Random
using MLUtils: kfolds
using CSV, DataFrames

using FliesClassifiers


# K-fold cross validation
function kfold_cross_validation(
    model_type::Type{<:AbstractFliesClassifier},
    X::AbstractMatrix,
    y::AbstractVector,
    k_folds::Int,
    seed::Int,
    params::Dict{Symbol,Real}
)
    accuracies = Float64[]
    folds = kfolds((X, y), k=k_folds)

    for (i, (train_data, test_data)) in enumerate(folds)
        X_train, y_train = train_data
        X_test, y_test = test_data

        model = if model_type == FlyNN
            FliesClassifiers.fit(FlyNN, X_train, y_train, params[:m], params[:ρ], params[:s], params[:γ], seed + i)
        elseif model_type == EaS
            FliesClassifiers.fit(EaS, X_train, y_train, params[:m], params[:k], seed + i)
        else
            error("Unsupported Model type: $model_type")
        end

        y_pred = FliesClassifiers.predict(model, X_test)
        accuracy = sum(y_pred .== y_test) / length(y_test)
        push!(accuracies, accuracy)
    end

    return round(mean(accuracies); digits = 3)
end


# Grid Search
function grid_search(
    model_type::Type{<:AbstractFliesClassifier},
    X::AbstractMatrix,
    y::AbstractVector,
    k_folds::Int,
    seed::Int,
    param_grid::Dict{Symbol,Vector},
    results_filename::Union{String, Nothing}=nothing
)
    best_params = Dict{Symbol,Any}(:accuracy => -1.0)
    param_names = collect(keys(param_grid))
    param_values = collect(values(param_grid))
    
    all_results = []

    # Generate all hyperparameter combinations
    total_combinations = prod(length, param_values)
    println("\n--- Starting Grid Search for model: $model_type ---")
    println("Total combinations to test: $total_combinations")

    for (i, p_values) in enumerate(Iterators.product(param_values...))

        current_params = Dict{Symbol, Real}(zip(param_names, p_values))

        # Special handling for 's', which depends on 'd' and 's_ratio' for FlyNN
        if model_type == FlyNN
            d = size(X, 1)
            # Calculate 's' and add it to the parameters if not already present
            if :s_ratio in keys(current_params)
                current_params[:s] = max(1, round(Int, current_params[:s_ratio] * d))
            end
        end

        # Run cross-validation with the current parameters
        accuracy = kfold_cross_validation(model_type, X, y, k_folds, seed, current_params)
        
        # Save current hp combination and accuracy 
        result_row = copy(current_params)
        result_row[:accuracy] = accuracy
        push!(all_results, result_row)

        # Print the progress
        param_str = join(["$k=$v" for (k, v) in current_params if k != :s], ", ")
        @printf("[%d/%d] Parameters: {%s} -> Accuracy: %.4f\n", i, total_combinations, param_str, accuracy)

        # Update the best parameters if accuracy has improved
        if accuracy > best_params[:accuracy]
            # Clear previous best parameters and insert the new ones
            empty!(best_params)
            best_params[:accuracy] = accuracy
            merge!(best_params, current_params)
            println("New best result found!")
        end
    end
    
    # Save results
    if results_filename !== nothing
        try
            df = DataFrame(all_results)
            CSV.write(results_filename, df)
            println("\nFull grid search results saved to: $results_filename")
        catch e
            @warn "Could not save results to file: $e"
        end
    end

    println("\n--- Grid Search for $model_type completed ---")
    @printf "Best Accuracy: %.4f\n" best_params[:accuracy]
    println("With the following hyperparameters:")
    for (key, val) in best_params
        if key != :accuracy
            println("  $key = $val")
        end
    end

    return best_params
end