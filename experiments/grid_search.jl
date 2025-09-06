using Printf
using Statistics, Random
using MLUtils: kfolds
using CSV, DataFrames

using FliesClassifiers

function kfold_cross_validation(
    model_type::Type{<:AbstractFliesClassifier},
    X::AbstractMatrix,
    y::AbstractVector,
    k_folds::Int,
    seed::Int,
    params::Dict{Symbol,Any}
)
    accuracies = Float64[]
    folds = kfolds((X, y), k=k_folds)
    d = size(X, 1)

    for (i, (train_data, test_data)) in enumerate(folds)
        X_train, y_train = train_data
        X_test, y_test = test_data

        local P::AbstractProjectionMatrix
        proj_type = params[:projection_type]

        if proj_type == RandomBinaryProjectionMatrix
            P = RandomBinaryProjectionMatrix(params[:m], d, params[:s]; seed=seed + i)
        elseif proj_type == RandomUniformProjectionMatrix
            P = RandomUniformProjectionMatrix(params[:m], d; seed=seed + i)
        else
            error("Unsupported projection type: $proj_type")
        end

        model = if model_type == FlyNN
            FliesClassifiers.fit(FlyNN, X_train, y_train, P, params[:k], params[:γ])
        elseif model_type == EaS
            FliesClassifiers.fit(EaS, X_train, y_train, P, params[:k])
        else
            error("Unsupported Model type: $model_type")
        end

        y_pred = FliesClassifiers.predict(model, X_test)
        accuracy = sum(y_pred .== y_test) / length(y_test)
        push!(accuracies, accuracy)
    end

    return round(mean(accuracies); digits=4)
end


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
    combinations = collect(Iterators.product(param_values...))
    total_combinations = length(combinations)
    println("\n--- Starting Grid Search for model: $model_type ---")
    println("Total combinations to generate: $total_combinations")

    valid_combination_count = 0
    for (i, p_values) in enumerate(combinations)
        current_params = Dict{Symbol, Any}(zip(param_names, p_values))

        proj_type = current_params[:projection_type]

        if proj_type == RandomBinaryProjectionMatrix && :s_ratio in keys(current_params)
            d = size(X, 1)
            current_params[:s] = max(1, round(Int, current_params[:s_ratio] * d))
        end

        valid_combination_count += 1
        
        # Run cross-validation with the current parameters
        accuracy = kfold_cross_validation(model_type, X, y, k_folds, seed, current_params)
        
        # Save current hp combination and accuracy  
        result_row = copy(current_params)
        result_row[:accuracy] = accuracy
        push!(all_results, result_row)

        # Print the progress
        params_for_printing = copy(current_params)

        if params_for_printing[:projection_type] == RandomUniformProjectionMatrix
            if haskey(params_for_printing, :s_ratio)
                delete!(params_for_printing, :s_ratio)
            end
        end

        if haskey(params_for_printing, :s)
            delete!(params_for_printing, :s)
        end
        
        params_for_printing[:projection_type] = last(split(string(params_for_printing[:projection_type]), "."))
        
        param_str = join(["$k=$v" for (k, v) in params_for_printing], ", ")
        @printf("[%d/%d] Parameters: {%s} -> Accuracy: %.4f\n", i, total_combinations, param_str, accuracy)

        # Update the best parameters if accuracy has improved
        if accuracy > best_params[:accuracy]
            empty!(best_params)
            best_params[:accuracy] = accuracy
            merge!(best_params, current_params)
            println("New best result found!")
        end
    end
    
    println("\nTotal valid combinations tested: $valid_combination_count")

    # Save results
    if results_filename !== nothing && !isempty(all_results)
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
            # Pretty print for type
            val_str = val isa Type ? last(split(string(val), ".")) : val
            println("  $key = $val_str")
        end
    end

    return best_params
end