"""
    predict(X_test::AbstractMatrix, model::FNN) -> Vector

Performs inference on new data using a trained Flynn model.

# Arguments
- `X_test::AbstractMatrix`: The test data matrix (d x n).
- `model::FNN`: The trained Flynn model object.

# Returns
- `Vector`: A vector of predicted labels for each column in `X_test`.
"""
function predict(X_test::AbstractMatrix, model::FNN)
    H = fly_hash(X_test, model.M, model.ρ)

    fX = model.W * H

    l, n = size(fX)
    y_pred = Vector{eltype(model.class_labels)}(undef, n)
    nties = 0

    winner_buf = Vector{Int}(undef, l)

    @inbounds for i in 1:n
        min_val = typemax(eltype(fX))
        win_count = 0

        for j in 1:l
            val = fX[j, i]
            if val < min_val
                min_val = val
                win_count = 1
                winner_buf[1] = j
            elseif val == min_val
                win_count += 1
                winner_buf[win_count] = j
            end
        end

        # Tie-break
        winner_label = if win_count > 1
            nties += 1
            model.class_labels[winner_buf[rand(1:win_count)]]
        else
            model.class_labels[winner_buf[1]]
        end

        y_pred[i] = winner_label
    end

    return y_pred
end