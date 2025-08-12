"""
    flynn_infer(X_test::AbstractMatrix, model::FNN) -> Vector

Performs inference on new data using a trained Flynn model.

# Arguments
- `X_test::AbstractMatrix`: The test data matrix (d x n).
- `model::FNN`: The trained Flynn model object.

# Returns
- `Vector`: A vector of predicted labels for each column in `X_test`.
"""
function predict(X_test::AbstractMatrix, model::FNN)
    H = fly_hash(X_test, model.M, model.ρ)
    fX = transpose(model.W) * H
    
    n = size(fX, 2)
    
    # Pre-allocate the prediction vector with the correct type for stability.
    y_pred = Vector{eltype(model.class_labels)}(undef, n)
    nties = 0
    
    # Find the minimum score for each column (data point).
    min_bf_scores = mapslices(minimum, fX; dims=1)
    
    for i in 1:n
        fx_col = @view fX[:, i]
        
        min_score = min_bf_scores[i]
        winner_indices = findall(==(min_score), fx_col)
        
        winner_label = if length(winner_indices) > 1
            # If there is a tie, pick one of the winning labels randomly.
            nties += 1
            model.class_labels[rand(winner_indices)]
        else
            model.class_labels[winner_indices[1]]
        end
        
        y_pred[i] = winner_label
    end
    
    return y_pred
end