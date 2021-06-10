using LinearAlgebra
include("projection.jl")
include("helper.jl")

function optimizeUnstructured(A,B,c;tol=1e-6,method="cholesky")
    n = length(A)
    m = size(A[1],1)

    #Create homogeneous problem, E basis of null(A)
    Ahli, li, E = convertToLIHomogeneous(A,B)

    #Solve problem
    (xli,X) = solveUnstructuredHomogeneous(Ahli,method=method)

    if xli == nothing #Problem is infeasible
        return (nothing, X[1:end-1,1:end-1],nothing)
    elseif c == nothing #Return feasible point with basis for null(A)
        (xih, Xih) = substituteBackInhomogeneous(xli,X,li,n)
        return (xih, Xih, E, nothing)
    end

    #Check if objective is bounded on null(A)+B
    index = abs.(c'*E).>tol
    if any(index) #Objective unbounded, return basis vector
        (xih, Xih) = substituteBackInhomogeneous(xli,X,li,n)
        return (xih,Xih,E[:,index],-Inf)
    end

    cli = vcat(c[li],0)
    dli = vcat(zeros(length(li),1),1)

    #Start Algorithm 4.2
    x = xli/xli[end] #Normalize the tau-component to 1
    X = X/xli[end]
    theta = c'*x/(d'*x) #Current theta level
    invX = inv(X)

    terminate = false
    iter = 0
    decreased = 5
    while !terminate
        #Step 1
        xp, choleskyGramian = projectUnstructured(Ahli,X,invX,method=method)
        Xp = eval(Ahli,xp)

        if isposdef(Xp) #Productive step (modify Xp)
            #Compute new theta level
            if method=="QR"
                #If method is QR, then the gramian was not computed before
                choleskyGramian = gramian(Ahli,invX)
            end
            #Get the matrices C and D
            xc = choleskyGramian \ c
            xd = choleksyGramian \ d
            C, D = eval(Ahli,xc), eval(Ahli,xd)
            thetanew, Xptheta = getNextThetaLevel(theta, C, D, X, Xp, invX)

            # Check if objective has decreased
            if theta - thetanew > tol
                decreased = 5
            else
                decreased -= 1
            end

        else #Unproductive step
             Xptheta = Xp
             #Objective has not decreased
             decreased -= 1
        end

        #Compute step size gamma > 0 with the new Xp(θ) (productive or not)
        X = getNextStep(X,Xptheta,invX)
        invX = inv(X)
        iter += 1

        #If objective has not decreased in 5 iterations, terminate
        if decreased < 0
            terminate = true
        end
    end

    #Substitute back
    error("Not implemented yet")

end

"""
    convertToLIHomogeneous(A,B;c=nothing)

Converts the LMI Ax+B into a homogeneous form ̃Ax̃ with Ã injective and
returns the basis of the null-space of A in column vector form

"""
function convertToLIHomogeneous(A,B)
    n = length(A)
    m = size(A[1],1)
    isorth = nothing

    #TODO: Optimize for space usage

    #Select linearly independent set
    Avec = hcat([symToVec(A[i]) for i in 1:n+1]...)
    li, E = selectLIColumns!(Avec,basis=true)
    k = length(li)

    #Homogenize problem
    Ah = Array{Matrix{Float64}}(undef,k+1)
    for j in li
        Ah[j] = [A[j] zeros(m,1); zeros(1,m) 0]
    end
    Ah[end] = [B zeros(m,1);zeros(1,m) 1]

    return Ah, li, E
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

    while true
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
    getNextThetaLevel(theta0, C, D, X, Xp, invX; tol=1e-3, low=-1e10)

Computes the lowest possible theta <= theta0 level such that the projection onto
the subspace corresponding to this level set is still positive definite
Returns theta and also the projection Xptheta corresponding to it

"""
function getNextThetaLevel(theta0, C, D, X, Xp, invX; tol=1e-6, low=-1e10)
    #Compute the coefficients for the line-search
    m = size(X,1)
    Im = I(m)
    DXp = tr(invX*D*invX*Xp)
    CXp = tr(invX*C*invX*Xp)
    CC = tr(invX*C*invX*C)
    DD = tr(invX*D*invX*D)
    CD = tr(invX*C*invX*D)

    #Perform a line search starting at theta0
    alpha = 1
    terminate = false
    double = true
    theta = theta0
    Xptheta = Xp
    while !terminate & theta > low
        #Decrease current level
        theta = theta - alpha
        #Check if the projection is positive definite
        Xptheta = Xp - (CXp - theta*DXp)*(C-theta*D)/(CC-2*theta*CD+theta^2*DD)
        if isposdef(Xptheta)
            if alpha < tol
                terminate = true
            end
            if double #If didn't find max stepsize yet, then double
                alpha = 2*alpha
            end
        else
            double = false #Stop searching for the upper bound on the step size
            #Go back to the previous level and half the step size
            theta = theta+alpha
            alpha = alpha/2
        end
    end
    return theta, Xptheta
end

"""
    eval(A,x)

Evaluates the vector of matrices on the input vector x
"""
function eval(A,x)
    return sum(A[i]*x[i] for i in 1:length(A))
end
