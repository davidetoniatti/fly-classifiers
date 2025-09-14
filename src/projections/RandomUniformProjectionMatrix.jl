import LinearAlgebra: mul!

"""
    RandomUniformProjectionMatrix

A type representing a dense projection matrix whose rows are sampled i.i.d.
from the uniform distribution over S^{d-1}.
"""
struct RandomUniformProjectionMatrix <: AbstractProjectionMatrix{Float64}
    matrix::Matrix{Float64}
end

"""
    RandomUniformProjectionMatrix(m::Int, d::Int; seed::Int=42) -> RandomUniformProjectionMatrix

Constructor for a random projection matrix of size `m x d`.
The rows are sampled i.i.d. from the uniform distribution over S^{d-1}.

# Arguments
- `m::Int`: Number of rows (projection dimension).
- `d::Int`: Number of columns (original dimension).
- `seed::Int`: Seed for the random number generator.
"""
function RandomUniformProjectionMatrix(m::Int, d::Int; seed::Int=42)
    rng = MersenneTwister(seed)
    mat = randn(rng, m, d)

    # Normalize each row in-place to have unit L2 norm.
    @threads for i in 1:m
        # Get a view of the current row to avoid copying data.
        row_view = @view mat[i, :]

        # Calculate the norm of the row.
        row_norm = norm(row_view)

        # Normalize the row. Add a small epsilon for numerical stability.
        # This prevents division by zero if a row happens to be all zeros.
        mat[i, :] ./= (row_norm + eps(eltype(mat)))
    end

    return RandomUniformProjectionMatrix(mat)
end


function mul!(y, M::RandomUniformProjectionMatrix, x)
    mul!(y, M.matrix, x)
    return y
end

# Implement the AbstractArray interface
Base.size(p::RandomUniformProjectionMatrix) = size(p.matrix)
Base.getindex(p::RandomUniformProjectionMatrix, i::Int, j::Int) = getindex(p.matrix, i, j)