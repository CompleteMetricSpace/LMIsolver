using LinearAlgebra

export rref


function rref!(A;tol=1e-8)
    n, m = size(A)
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
            #Save permutation
            push!(perm,[i,ic])
        end
        #Eliminate all elements in this column below ic
        for i = ic+1:n, k=m:-1:j #Go from back to front changing A[i,j] last
            A[i,k] = A[i,k] - A[ic,k]*A[i,j]/A[ic,j]
        end
        #Increase current row
        ic += 1
    end
    return perm, ic-1
end


function selectLIColumns!(A::Matrix{Float64};tol=1e-8)
    (n,m) = size(A)
    perm, l = rref!(A')
    println(perm," ",l)
    cols = collect(1:m)
    for k in length(perm):-1:1
        tmp = cols[perm[k][1]]
        cols[perm[k][1]] = cols[perm[k][2]]
        cols[perm[k][2]] = tmp
    end
    return cols[1:l]
end
