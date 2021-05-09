using LinearAlgebra

@doc raw"""
    projectUnstructured(A,X,S;method="cholesky")

Computes the projection of X onto the range of A in the S-metric

# Arguments
- 'A::Vector{Matrix{Float64}}`: a list of linear independent symmetric matrices
                                of the same dimension
- 'X::Matrix{Float64}`: a symmetric matrix of the same dimension as A[1]
- 'S::Matrix{Float64}`: a symmetric, positive definite matrix of the same
                        dimension as X

# Returns
- 'x::Vector{Float64}`: a vector such that Ax is the projection of X 
"""
function projectUnstructured(A,X,S;method="cholesky")
    n = length(A)
    m = size(A[1],1)
    if method == "cholesky"
        #Create the matrix H and vector q
        H = zeros(n,n)
        q = zeros(n)
        for i in 1:n
            for j in i:n
                H[i,j] = tr(S*A[i]*S*A[j])
                H[j,i] = H[i,j]
            end
            q[i] = tr(S*X*S*A[i])
        end

        #Solve system Hx = q
        x = cholesky(H) \ q

        return x

    elseif method == "QR"
        #TODO Implement QR-decomposition method
        error("Not implemented yet")
    else
        error("Unknown method: method must be either \"cholesky\" or \"QR\"")
    end
end
