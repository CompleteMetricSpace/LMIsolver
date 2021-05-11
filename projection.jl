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
    M = div(m*(m+1),2)
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
        #TODO Efficiency when using upper-triangular matrices
        L = cholesky(S).U
        #Create basis matrix B
        B = zeros(M,n)
        for j in 1:n
            B[:,j] = symToVec(L'*A[j]*L)
        end
        q = symToVec(L'*X*L)
        Q, R = qr(B)
        x = UpperTriangular(R) \ (Q'*q)[1:n]
        return x
    else
        error("Unknown method: method must be either \"cholesky\" or \"QR\"")
    end
end

function symToVec(A::Matrix{Float64})
    m = size(A,1)
    v = zeros(div(m*(m+1),2))
    for i in 1:m
        for j in i:m
            v[m*(i-1)-div(i*(i-1),2)+j] = A[i,j]
        end
    end
    return v
end

function vecToSym(v,m)
    #TODO: Efficiency
    return [i <= j ? v[m*(i-1)-div(i*(i-1),2)+j] : v[m*(j-1)-div(j*(j-1),2)+i]
                                                         for i in 1:m, j in 1:m]
end
