using LinearAlgebra

@doc raw"""
    projectUnstructured(A,X,cholS,T,H,W;method="cholesky")

Computes the projection of X onto the range of A in the inv(S)-metric

# Arguments
- 'A::Vector{Matrix{Float64}}`: a list of linear independent symmetric matrices
                                of the same dimension
- 'X::Matrix{Float64}`: a symmetric matrix of the same dimension as A[1]
- 'cholS::Matrix{Float64}`: the cholesky decomposition of a symmetric, positive
                            definite matrix S of the same dimension as X

# Returns
- 'x::Vector{Float64}`: a vector such that Ax is the projection of X
- 'H::Matrix{Float64}': the cholesky decomposition of the gramian of A w.r.t. inv(S)
"""
function projectUnstructured!(A,X,cholS,workspace;method="cholesky")
    n = length(A)
    m = size(A[1],1)
    M = div(m*(m+1),2)
    T, H, W = workspace.T, workspace.H, workspace.W[1]
    if method == "cholesky"
        #Create the matrix H and vector q
        @info "Computing prehessian"
        @time begin
            F = cholS
            for i in 1:n
                Tm = reshape(view(T, 1:m^2, i), m, m)
                ldiv!(Tm,F.L,A[i])
                rdiv!(Tm,F.U)
            end
            ldiv!(W,F.L,X)
            rdiv!(W,F.U)
        end
        @info "Computing Hessian"
        @time begin
            mul!(H,T',T)
        end

        q = zeros(n)
        for i in 1:n
            q[i] = sum(W.*reshape(view(T, 1:m^2, i), m, m))
        end

        #Solve system Hx = q
        @time choleskyGramian = cholesky!(H,check=false)

        x = choleskyGramian \ q

        return x, choleskyGramian

    elseif method == "QR"
        #TODO Efficiency when using upper-triangular matrices
        L = cholS.L
        #Create basis matrix B
        B = zeros(M,n)
        for j in 1:n
            B[:,j] = symToVec(L*A[j]*L')
        end
        q = symToVec(L*X*L')
        Q, R = qr(B)
        x = UpperTriangular(R) \ (Q'*q)[1:n]
        return x, nothing
    else
        error("Unknown method: method must be either \"cholesky\" or \"QR\"")
    end
end

"""
    gramian(A,cholS)

Computes the gramian w.r.t. to the inner product induces by inv(S)
"""
function gramian!(A::Vector{Matrix{Float64}},cholS::Matrix{Float64},workspace)
    n = length(A)
    m = size(A[1],1)
    F = cholS
    for i in 1:n
        Tm = reshape(view(workspace.T, 1:m^2, i), m, m)
        ldiv!(Tm,F.L,A[i])
        rdiv!(Tm,F.U)
    end
    mul!(workspace.H,T',T)
end

function projection_data!(A, c, d, choleskyGramian, Xp, cholX, C, D)
    xc = choleskyGramian \ c
    xd = choleskyGramian \ d
    eval_LMI!(A,xc,C)
    eval_LMI!(A,xd,D)
    L = cholX.L
    LDL = (L \ D) / L'
    LCL = (L \ C) / L'
    LXpL = (L \ Xp) / L'

    DXp = sum(LDL.*LXpL)
    CXp = sum(LCL.*LXpL)
    CC = sum(LCL.*LCL)
    DD = sum(LDL.*LDL)
    CD = sum(LCL.*LDL)
    return [C,D,xc,xd,DXp,CXp,CC,DD,CD]
end

function project_theta!(pdata, theta, Xp, Xptheta)
    C, D, xc, xd, DXp, CXp, CC, DD, CD = pdata
    Xptheta .= Xp .- (CXp - theta*DXp)*(C.-theta.*D)./(CC-2*theta*CD+theta^2*DD)
end

function symToVec(A)
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
