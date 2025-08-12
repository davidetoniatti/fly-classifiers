using Random, LinearAlgebra, SparseArrays, StatsBase, .Threads

function random_binary_matrix_slow(m, d, s, seed)
    # Create an independent Random Number Generator (RNG) for each thread
    # to ensure reproducibility and avoid data races on the global RNG.
    rngs = [MersenneTwister(seed + i) for i in 1:Threads.nthreads()]
    
    M = falses(m,d)

    # Use multithreading to generate rows in parallel.
    @threads for i in 1:m
        tid = threadid()
        start_pos = (i - 1) * s + 1
        end_pos   = i * s

        #  Use the thread-specific RNG for sampling.
        sampled_cols = sample(rngs[tid], 1:d, s; replace=false)
        M[i,sampled_cols] .= true 
    end
    return M
end

function fly_hash_slow(X, M, ρ)
    d, n = size(X)
    m = size(M,1)
    
    H = falses(m,n)

    x_proj = Vector{Float64}(undef, m)

    for i in 1:n
        x = X[:, i]
        mul!(x_proj, M, x)

        # Trova gli indici delle ρ righe con valore massimo nella colonna i
        topk_idx = partialsortperm(x_proj, rev=true, 1:ρ)
        
        H[topk_idx,i] .= true
    end

    return H
end