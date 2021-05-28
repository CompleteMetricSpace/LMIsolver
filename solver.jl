using LinearAlgebra
include("projection.jl")
include("helper.jl")

function optimizeUnstructured(A,B,c;tol=1e15,method="cholesky")
    n = length(A)
    m = size(A[1],1)

    #Create homogeneous problem
    Ahli, li = convertToLIHomogeneous(A,B)

    #Solve problem
    (xli,X) = solveUnstructuredHomogeneous(Ahli,tol=tol,method=method)

    if xli == nothing
        return (nothing, X[1:end-1,1:end-1])
    end


    if c == nothing
        return substituteBackInhomogeneous(xli,X,li,n)
    else
        cli = vcat(c,0)
        dli = vcat(zeros())
        #TODO: How to modify c to get a problem equivalent to the initial problem
        error("Not implemented yet")
    end
end





function convertToLIHomogeneous(A,B)
    n = length(A)
    m = size(A[1],1)

    #TODO: Optimize for space usage

    #Select linearly independent set
    Avec = hcat([symToVec(A[i]) for i in 1:n+1]...)
    li = selectLIColumns!(Avec)
    k = length(li)

    #Homogenize problem
    Ah = Array{Matrix{Float64}}(undef,k+1)
    for j in li
        Ah[j] = [A[j] zeros(m,1); zeros(1,m) 0]
    end
    Ah[end] = [B zeros(m,1);zeros(1,m) 1]

    return Ah, li
end

function substituteBackInhomogeneous(xli,X,li,n)
    #Substitute back solution
    x = zeros(n+1)
    for j in 1:length(li)
        x[li[j]] = xli[j]
    end

    #Substitute and return inhomogeneous solution
    x = x/x[end]
    X = X/x[end]
    return (x[1:end-1],X[1:end-1,1:end-1])
end

function optimizeUnstructuredHomogeneous(A,c;tol=1e15,method="cholesky")
    n = length(A)
    m = size(A[1],1)
    d = [zeros(1,)]

    #Get feasible solution
    (x,X) = solveUnstructuredHomogeneous(A,tol=tol,method=method)

    if x == nothing
        return (nothing, X[1:end-1,1:end-1])
    end


end

function solveUnstructured(A,B;tol=1e15,method="cholesky")
    return optimizeUnstructured(A,B,nothing;tol=tol,method=method)
end

@doc raw"""
    solveUnstructuredHomogeneous(A;tol=1e15)

Solves an unstructured and homonegeous linear matrix inequality Ax > 0
with the projection algorithm

# Arguments:
- 'A::Vector{Matrix{Float64}}`: a list of linear independent symmetric matrices
                                of the same dimension
- 'tol::Float64`: a nonnegative number, the termination tolerance

# Returns:
- 'nothing` if the problem is infeasible
- '(x::Float64,X::Matrix{Float64})` if the problem is feasible, where x is the
  is the solution to Ax = X > 0

"""
function solveUnstructuredHomogeneous(A;tol=1e15,method="cholesky")
    n = length(A)
    m = size(A[1],1)

    #Initiate
    X = I(m)
    invX = I(m)
    x = zeros(n)

    terminate = false
    while !terminate
        #Project onto range of A
        x = projectUnstructured(A,X,invX,method=method)
        Xp = eval(A,x)
        #print("Eigvals Xp: ",min(eigvals(Xp)...))
        if isposdef(Xp)
            return (x,Xp)
        else
            #Compute step size gamma > 0
            X = getNextStep(X,Xp,invX)
            invX = inv(X)
        end
        if norm(invX) > tol
            return (nothing,X) #Probably infeasible
        end
    end
end

"""
    getNextStep(X,Xp,invX;search=false)

Computes the next step using a line search or the crude bound

"""
function getNextStep(X,Xp,invX;search=false)
    sqrtinvX = sqrt(invX)
    psi = sqrtinvX * (Xp - X) * sqrtinvX
    if search
        #TODO Line search for minimizing pi(gamma)
        error("Not implemented yet")
    else
        #Just compute the crude lower bound on pi(gamma)
        rho_inf = maximum(abs.(eigvals(psi)))
        gamma = 1/(1+rho_inf)
        return Symmetric(X*inv(X-gamma*(Xp-X))*X) #Need to symmetrize (rounding)
    end
end

"""
    eval(A,x)

Evaluates the vector of matrices on the input vector x
"""
function eval(A,x)
    return sum(A[i]*x[i] for i in 1:length(A))
end
