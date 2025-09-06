using BenchmarkTools
using DataFrames
using CSV
using Plots
using Random

include("../src/FliesClassifiers.jl")
using .FliesClassifiers

## Synthetic Data Generation
println("Generating synthetic data...")
const d = 128
const n= 10000
const l = 10
const s_factor = 0.1
const γ = 0.9

const X_train = randn(Float32, d, n)
const y_train = rand(1:l, n)
println("Data generation complete.")

## Benchmark Configuration
const exp_values = [2^i for i in 0:6]
const nnz_values = [2.0^-i for i in 1:8]

results = DataFrame(
    model = String[],
    projection = String[],
    m = Int[],
    k = Int[],
    fit_time_s = Float64[],
)

## Benchmark Execution
println("Starting benchmark for 'fit' function...")
for exp in exp_values
    m = exp * d
    for nnz in nnz_values
        k = ceil(Int, nnz * m)
        println("   Testing with m=$m, k=$k")

        s = round(Int, s_factor * d)

        P_binary = RandomBinaryProjectionMatrix(m, d, s)
        P_uniform = RandomUniformProjectionMatrix(m, d)

        b_fit_fly_bin = @benchmark fit(FlyNN, $X_train, $y_train, $P_binary, $k, $γ)
        push!(results, ("FlyNN", "Binary", m, k, median(b_fit_fly_bin).time / 1e9))

        b_fit_fly_uni = @benchmark fit(FlyNN, $X_train, $y_train, $P_uniform, $k, $γ)
        push!(results, ("FlyNN", "Uniform", m, k, median(b_fit_fly_uni).time / 1e9))

        b_fit_eas_bin = @benchmark fit(EaS, $X_train, $y_train, $P_binary, $k)
        push!(results, ("EaS", "Binary", m, k, median(b_fit_eas_bin).time / 1e9))
        
        b_fit_eas_uni = @benchmark fit(EaS, $X_train, $y_train, $P_uniform, $k)
        push!(results, ("EaS", "Uniform", m, k, median(b_fit_eas_uni).time / 1e9))
    end
end
println("Benchmark finished.")

##  Saving and Visualizing Results
!isdir("results/benchmarks") && mkdir("results/benchmarks")
CSV.write("results/benchmarks/benchmark_fit_results.csv", results)
println("Results saved to benchmark_fit_results.csv")

# Visualization
println("Generating plots...")

function create_fit_plot(results_df, fixed_param, varying_param)
    if fixed_param == :k
        val = 32 # Fix k to an intermediate value
        df_plot = filter(row -> row.k == val, results_df)
        title_suffix = "with k = $val (fixed)"
        xlabel = "Projection Dimension (m)"
    else
        val = 2048 # Fix m to an intermediate value
        df_plot = filter(row -> row.m == val, results_df)
        title_suffix = "with m = $val (fixed)"
        xlabel = "Number of Activations (k)"
    end

    df_plot[!, :label] = df_plot.model .* " " .* df_plot.projection

    p_fit = plot(
        title="Training Time (fit) $title_suffix",
        xlabel=xlabel,
        ylabel="Time (s, log scale)",
        legend=:topleft,
        yaxis=:log10
    )

    for group in groupby(df_plot, :label)
        label = group.label[1]
        plot!(p_fit, group[!, varying_param], group.fit_time_s, label=label, marker=:circle)
    end
    
    return p_fit
end

# Plot varying m (k fixed)
!isdir("plots/benchmarks") && mkdir("plots/benchmarks")
p_fit_m = create_fit_plot(results, :k, :m)
savefig(p_fit_m, "plots/benchmarks/benchmark_fit_vs_m.png")

# Plot varying k (m fixed)
p_fit_k = create_fit_plot(results, :m, :k)
savefig(p_fit_k, "plots/benchmarks/benchmark_fit_vs_k.png")

println("Plots saved as benchmark_fit_vs_m.png and benchmark_fit_vs_k.png")

# Display results in the terminal
println("\nBenchmark Results (training):")
show(results, allrows=true)