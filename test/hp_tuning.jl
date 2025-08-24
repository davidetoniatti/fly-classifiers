using Random
using Printf
using Statistics
using MLUtils

function kfold_cross_validation(X::AbstractMatrix, y::AbstractVector, k::Int, m::Int, ρ::Int, s::Int, γ::Real, seed::Int)
    # Set seed
    Random.seed!(seed)
    
    accuracies = Float64[]
    
    folds = kfolds((X, y), k=k)

    for (i, (train_data, test_data)) in enumerate(folds)
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        # Training and prediction
        model = FlyNN.fit(X_train, y_train, m, ρ, s, γ, seed + i)
        y_pred = FlyNN.predict(X_test, model)
        
        # Compute accuracy
        accuracy = sum(y_pred .== y_test) / length(y_test)
        push!(accuracies, accuracy)
    end
    
    return mean(accuracies)
end


function update_best_params!(best_params::Dict, accuracy::Float64, current_params::Dict)
    if accuracy > best_params[:accuracy]
        best_params[:accuracy] = accuracy
        merge!(best_params, current_params)
        return true
    end
    return false
end

function tune_m(X, y, k, s_ratio, ρ, γ, seed, best_params)
    println("Tuning 'm' with s/d=$s_ratio, ρ=$ρ, γ=$γ")
    d = size(X, 1)
    s = round(Int, s_ratio * d)
    
    m_values = round.(Int, exp.(range(log(4d), stop=log(4096d), length=10)))
    
    for m in m_values
        accuracy = kfold_cross_validation(X, y, k, m, ρ, s, γ, seed)
        @printf "  m = %-6d -> Accuracy: %.4f\n" m accuracy
        
        current_params = Dict(:m => m, :s_ratio => s_ratio, :s => s, :ρ => ρ, :γ => γ)
        update_best_params!(best_params, accuracy, current_params)
    end
end

function tune_s_ratio(X, y, k, m, ρ, γ, seed, best_params)
    println("Tuning 's/d' with m=$m, ρ=$ρ, γ=$γ")
    d = size(X, 1)
    
    s_ratio_values = range(0.1, stop=0.8, length=10)
    
    for s_ratio in s_ratio_values
        s = max(1, round(Int, s_ratio * d))
        accuracy = kfold_cross_validation(X, y, k, m, ρ, s, γ, seed)
        @printf "  s/d = %.2f (s=%d) -> Accuracy: %.4f\n" s_ratio s accuracy
        
        current_params = Dict(:m => m, :s_ratio => s_ratio, :s => s, :ρ => ρ, :γ => γ)
        update_best_params!(best_params, accuracy, current_params)
    end
end

function tune_rho(X, y, k, m, s_ratio, γ, seed, best_params)
    println("Tuning 'ρ' with m=$m, s/d=$s_ratio, γ=$γ")
    d = size(X, 1)
    s = round(Int, s_ratio * d)
    
    rho_values = unique(round.(Int, exp.(range(log(4), stop=log(256), length=10))))
    
    for ρ in rho_values
        accuracy = kfold_cross_validation(X, y, k, m, ρ, s, γ, seed)
        @printf "  ρ = %-4d -> Accuracy: %.4f\n" ρ accuracy

        current_params = Dict(:m => m, :s_ratio => s_ratio, :s => s, :ρ => ρ, :γ => γ)
        update_best_params!(best_params, accuracy, current_params)
    end
end

function tune_gamma(X, y, k, m, s_ratio, ρ, seed, best_params)
    println("Tuning 'γ' with m=$m, s/d=$s_ratio, ρ=$ρ")
    d = size(X, 1)
    s = round(Int, s_ratio * d)
    
    gamma_values = range(0.1, stop=0.8, length=10)
    
    for γ in gamma_values
        accuracy = kfold_cross_validation(X, y, k, m, ρ, s, γ, seed)
        @printf "  γ = %.2f -> Accuracy: %.4f\n" γ accuracy
        
        current_params = Dict(:m => m, :s_ratio => s_ratio, :s => s, :ρ => ρ, :γ => γ)
        update_best_params!(best_params, accuracy, current_params)
    end
end


function run_full_search(X::AbstractMatrix, y::AbstractVector, k::Int, seed::Int=123)
    println("--- HYPERPARAMETER SEARCH FLYNN ---")
    
    best_params = Dict{Symbol, Any}(:accuracy => -1.0)
    
    println("\n====================== HP 'm' ======================")
    for s_ratio in [0.1, 0.3], ρ in [8, 32], γ in [0.1, 0.5]
        tune_m(X, y, k, s_ratio, ρ, γ, seed, best_params)
        println("-"^40)
    end

    println("\n====================== HP 's/d' ======================")
    for m in [256, 1024], ρ in [8, 32], γ in [0.1, 0.5]
        tune_s_ratio(X, y, k, m, ρ, γ, seed, best_params)
        println("-"^40)
    end

    println("\n====================== HP 'ρ' ======================")
    for m in [256, 1024], s_ratio in [0.1, 0.3], γ in [0.1, 0.5]
        tune_rho(X, y, k, m, s_ratio, γ, seed, best_params)
        println("-"^40)
    end
    
    println("\n====================== HP 'γ' ======================")
    for m in [256, 1024], s_ratio in [0.1, 0.3], ρ in [8, 32]
        tune_gamma(X, y, k, m, s_ratio, ρ, seed, best_params)
        println("-"^40)
    end
    
    println("\n--- HYPERPARAMETER SEARCH COMPLETED ---")
    @printf "Best Accuracy: %.4f\n" best_params[:accuracy]
    println("With the following hyperparameters:")
    for (key, val) in best_params
        if key != :accuracy
            println("  $key = $val")
        end
    end

    return best_params
end