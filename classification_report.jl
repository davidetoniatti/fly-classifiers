using Printf

function classification_report(y_true::AbstractVector, y_pred::AbstractVector; digits::Int=3)
    @assert length(y_true) == length(y_pred) "y_true and y_pred must have the same length"

    labels = sort(unique(vcat(y_true, y_pred)))
    report = Dict{Any, Dict{Symbol, Real}}()

    support_list = Int[]
    for label in labels
        tp = sum((y_true .== label) .& (y_pred .== label))
        fp = sum((y_true .!= label) .& (y_pred .== label))
        fn = sum((y_true .== label) .& (y_pred .!= label))
        precision = tp + fp == 0 ? 0.0 : tp / (tp + fp)
        recall    = tp + fn == 0 ? 0.0 : tp / (tp + fn)
        f1        = (precision + recall == 0) ? 0.0 : 2 * precision * recall / (precision + recall)
        support::Int   = sum(y_true .== label)
        push!(support_list, support)

        report[label] = Dict(:precision => precision,
                             :recall => recall,
                             :f1 => f1,
                             :support => support)
    end

    total_support = sum(support_list)

    # Macro averages
    macro_measures = Dict(
        :precision => mean([report[l][:precision] for l in labels]),
        :recall    => mean([report[l][:recall] for l in labels]),
        :f1        => mean([report[l][:f1] for l in labels]),
        :support   => total_support
    )
    report[:macro_measures] = macro_measures

    # Weighted averages
    weighted_measures = Dict(
        :precision => sum([report[l][:precision]*report[l][:support] for l in labels]) / total_support,
        :recall    => sum([report[l][:recall]*report[l][:support] for l in labels]) / total_support,
        :f1        => sum([report[l][:f1]*report[l][:support] for l in labels]) / total_support,
        :support   => total_support
    )
    report[:weighted_measures] = weighted_measures

    # Accuracy
    accuracy = sum(y_true .== y_pred) / length(y_true)

    # Print
    println(rpad("class", 15), rpad("precision", 10), rpad("recall", 10),
            rpad("f1-score", 10), rpad("support", 10))
    println("-"^60)

    for label in labels
        row = report[label]
        println(rpad(string(label), 15),
                rpad(@sprintf("%.*f", digits, row[:precision]), 10),
                rpad(@sprintf("%.*f", digits, row[:recall]), 10),
                rpad(@sprintf("%.*f", digits, row[:f1]), 10),
                rpad(string(row[:support]), 10))
    end


    println("-"^60)
    
    println(rpad("accuracy", 35),
            rpad(@sprintf("%.*f", digits, accuracy), 10),
            rpad(string(macro_measures[:support]), 10))
            

    println(rpad("macro avg", 15),
            rpad(@sprintf("%.*f", digits, macro_measures[:precision]), 10),
            rpad(@sprintf("%.*f", digits, macro_measures[:recall]), 10),
            rpad(@sprintf("%.*f", digits, macro_measures[:f1]), 10),
            rpad(string(macro_measures[:support]), 10))

    println(rpad("weighted avg", 15),
            rpad(@sprintf("%.*f", digits, weighted_measures[:precision]), 10),
            rpad(@sprintf("%.*f", digits, weighted_measures[:recall]), 10),
            rpad(@sprintf("%.*f", digits, weighted_measures[:f1]), 10),
            rpad(string(weighted_measures[:support]), 10))

    println("-"^60)
    
    return report, accuracy
end