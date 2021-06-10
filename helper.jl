using LinearAlgebra

export rref

"""
    rref!(A;B=nothing,tol=1e-8)

Perform a reduction to the reduces row echelon form of (A|B)

# Arguments:
- 'A`: a numerical matrix of size n × m
- 'B`: a numerical matrix of size n × r
- 'tol`: a positive number (tolerance)

# Returns:
- 'perm::Vector{Vector{Int64}}`: an array of completed permutations
- 'ic::Int64`: the number of non-zero rows

"""
function rref!(A;B=nothing,tol=1e-8)
    n, m = size(A)
    mB = (B == nothing) ? 0 : size(B,2)
    perm::Vector{Vector{Int64}} = []
    ic = 1 #Current row
    for j in 1:m
        #Find row with non-zero element
        i = ic
        while i <= n && abs(A[i,j]) < tol
            i += 1
        end
        if i > n
            continue #Next column, stay on the same row
        elseif i != ic
            #Switch rows (only if i != ic)
            for k = j:m
                tmp = A[i,k]
                A[i,k] = A[ic,k]
                A[ic,k] = tmp
            end
            for k = 1:mB
                tmp = B[i,k]
                B[i,k] = B[ic,k]
                B[ic,k] = tmp
            end
            #Save permutation
            push!(perm,[i,ic])
        end
        #Eliminate all elements in this column below ic
        for i = ic+1:n
            for k=1:mB #First do B
                B[i,k] = B[i,k] - B[ic,k]*A[i,j]/A[ic,j]
            end
            for k=m:-1:j #Go from back to front changing A[i,j] last
                A[i,k] = A[i,k] - A[ic,k]*A[i,j]/A[ic,j]
            end
        end
        #Increase current row
        ic += 1
    end
    return perm, ic-1
end

"""
    selectLIColumns!(A::Matrix{Float64};basis=false,tol=1e-8)

Selects the linear independent columns from the matrix A and computes the
basis for the null-space of A in column vector form
"""
function selectLIColumns!(A::Matrix{Float64};basis=false,tol=1e-8)
    n, m = size(A)
    E = basis ? Matrix{Float64}(I(m)) : nothing
    perm, l = rref!(A',B=E,tol=tol)
    cols = collect(1:m)
    for k in length(perm):-1:1
        tmp = cols[perm[k][1]]
        cols[perm[k][1]] = cols[perm[k][2]]
        cols[perm[k][2]] = tmp
    end
    return cols[1:l], E[:,l+1:end]
end

"""
    getNullSpace!(A::Matrix{Float64};tol=1e-8)

Computes a basis for the null-space of a matrix in column-vector form

"""
function getNullSpace!(A::Matrix{Float64};tol=1e-8)
    n, m = size(A)
    E = Matrix{Float64}(I(m))
    _, l = rref!(A',B=E,tol=tol)
    return E'[:,l+1:end]
end
