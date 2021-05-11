using LinearAlgebra
include("projection.jl")

function solveUnstructured(A,B)
    n = length(A)
    m = size(A[1],1)

    #Homogenize problem
    Ah = Array{Matrix{Float64}}(undef,n+1)
    for j in 1:n
        Ah[j] = [A[j] zeros(m,1); zeros(1,m) 0]
    end
    Ah[n+1] = [B zeros(m,1);zeros(1,m) 1]

    #Select linearly independent set
    

    error("Not implemented yet")
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
