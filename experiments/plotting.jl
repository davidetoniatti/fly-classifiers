using DataFrames
using CSV
using Plots
using Statistics
using LaTeXStrings

println("Julia environment for plotting configured successfully.")

"""
Generates plots to compare model performance (for Vision experiments).
"""
function generate_vision_plots(df::DataFrame, exp::Int, plots_dir::String)
    model_names = names(df)[end-3:end]
    dataset_names = unique(df.DatasetName)
    
    output_dir = joinpath(plots_dir, "vision")
    !isdir(output_dir) && mkpath(output_dir)

    for name in dataset_names
        println("Generating plot for dataset: $name (m=$(exp)d)")

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
            minorgrid = true,
            titlefontsize=10,
            tickfontsize=7
        )

        for model in model_names
            plot!(p,
                x_positions,
                subset_df[!, model] .* 100,
                label = replace(model, "_" => " "),
                marker = :circle,
                linewidth = 2
            )
        end
        
        filename = joinpath(output_dir, "accuracies_m_$(exp)d_$(name).png")
        savefig(p, filename)
    end
end


"""
Generates scatter plots to compare model performance (for OpenML experiments).
"""
function generate_scatter_plots(results_df::DataFrame, exp::Int, plots_dir::String)
    println("\nCreating scatter plots for m = $(exp)d...")
    output_dir = joinpath(plots_dir, "openml")
    !isdir(output_dir) && mkpath(output_dir)

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
            xlabel=replace(string(x_col), "_" => " "),
            ylabel=replace(string(y_col), "_" => " "),
            title="$title_str (m=$(exp)d)",
            legend=:topleft,
            aspect_ratio=1,
            xlims=lims,
            ylims=lims,
            markersize=5,
            markerstrokewidth=0,
            alpha=0.6,
            label=""
        )
        plot!(plt, [lims[1], lims[2]], [lims[1], lims[2]], style=:dash, color=:red, label="")

        centroid_x = mean(results_df[!, x_col])
        centroid_y = mean(results_df[!, y_col])

        scatter!(
            plt,
            [centroid_x],
            [centroid_y],
            label="Centroid",
            markercolor=:black,
            markershape=:star5,
            markersize=10,
            markerstrokewidth=1,
            markerstrokecolor=:yellow
        )

        filename = joinpath(output_dir, "comparison_$(x_col)_vs_$(y_col)_m_$(exp)d.png")
        savefig(plt, filename)
    end
    println("Scatter plots for m = $(exp)d saved to 'plots/' directory.")
end


"""
Generic function to find and process result files.
"""
function process_result_files(plot_type::String, file_pattern::Regex, plot_function::Function, results_dir::String, plots_dir::String)
    println("\n--- Generating $plot_type Plots ---")
    
    if !isdir(results_dir)
        println("'$results_dir' directory not found. Run experiments first.")
        return
    end

    result_files = filter(f -> occursin(file_pattern, f), readdir(results_dir))
    
    if isempty(result_files)
        println("No $plot_type result files found in '$results_dir'.")
        return
    end

    for file in result_files
        match_exp = match(r"m_(\d+)d", file)
        if isnothing(match_exp) continue end
        
        exp = parse(Int, match_exp.captures[1])
        filepath = joinpath(results_dir, file)
        df = CSV.read(filepath, DataFrame)
        plot_function(df, exp, plots_dir)
    end
    println("\n$plot_type plots generation complete.")
end


# Script entry point
function main()
    if !(1 <= length(ARGS) <= 3) || !(ARGS[1] in ["vision", "openml"])
        println("Usage: julia $(@__FILE__) [vision|openml] [results_dir] [plots_dir]")
        println("\nArguments:")
        println("  [vision|openml]: (Required) The type of plots to generate.")
        println("  [results_dir]:   (Optional) Directory to read CSV results from. Defaults to '../results'.")
        println("  [plots_dir]:     (Optional) Directory to save generated plots. Defaults to '../plots'.")
        return
    end

    plot_type = ARGS[1]
    results_dir = length(ARGS) >= 2 ? ARGS[2] : joinpath("..", "results")
    plots_dir = length(ARGS) >= 3 ? ARGS[3] : joinpath("..", "plots")

    println("Generating $plot_type plots. Reading from: $results_dir, Saving to: $plots_dir")

    if plot_type == "vision"
        process_result_files("Vision", r"accuracies_m_\d+d_vision\.csv", generate_vision_plots, results_dir, plots_dir)
    elseif plot_type == "openml"
        process_result_files("OpenML", r"accuracies_m_\d+d_openml\.csv", generate_scatter_plots, results_dir, plots_dir)
    end
end

main()