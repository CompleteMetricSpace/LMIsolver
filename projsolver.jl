using LinearAlgebra
using Logging
include("projection.jl")
include("lmitools.jl")

struct LMISolution
    x::Union{Array{<:Real},Nothing}
    X::Union{Symmetric{<:Real},Nothing}
    val::Union{Real,Nothing}
    inf_directions::Union{Matrix{<:Real},Nothing}
    status::String
    type::String
end

struct Workspace
    T::Matrix{Float64}
    H::Matrix{Float64}
    W::Array{Matrix{Float64}}
end

function optimizeUnstructured(A,B,c;tol=1e-8,stoptol=1e15,method="cholesky")
    n = length(A)
    m = size(A[1],1)

    #Define storage
    T = zeros((m+1)^2,n+1) #Allocate worst case size

    probtype = isnothing(c) ? "feasibility" : "minimization"

    #Create homogeneous problem, E basis of null(A)
    Ahli, li, E = convertToLIHomogeneous!(A,B,T)
    nli = length(Ahli)

    #Define storage
    W = zeros(m+1,m+1)
    H = zeros(nli,nli)
    Tli = view(T,(m+1)^2,nli) #Allocate worst case size
    workspace = Workspace(T,H,[W])

    ##println("Ahli = $Ahli")
    #Solve problem
    (xli,X) = solveUnstructuredHomogeneous(Ahli,workspace,method=method)

    #println("xli = $xli")
    if isnothing(xli) #Problem is infeasible
        val = isnothing(c) ? nothing : Inf
        @info "Value" X
        return LMISolution(nothing, X, val, nothing, "infeasible", probtype)
    elseif isnothing(c) #Return feasible point with basis for null(A)
        (xih, Xih) = substituteBackInhomogeneous(xli,X,li,n)
        return LMISolution(xih, Symmetric(Xih), nothing, nothing, "feasible", probtype)
    end

    #Check if objective is bounded on null(A)+B
    index = abs.(c'*E).>tol
    if any(index) #Objective unbounded, return basis vector
        (xih, Xih) = substituteBackInhomogeneous(xli,X,li,n)
        return LMISolution(xih,Xih,-Inf,E[:,index],"unbounded",probtype)
    end

    cli = vcat(c[li],0)
    dli = vcat(zeros(length(li)),1)

    #Start Algorithm 4.2
    x = xli/xli[end] #Normalize the tau-component to 1
    X = X/xli[end]
    theta = cli'*x/(dli'*x) #Current theta level
    cholX = cholesky(X)
    Xp = zeros(m+1,m+1)
    C = zeros(m+1,m+1)
    D = zeros(m+1,m+1)

    #Feasible points
    xstar = undef
    Xptheta = undef

    terminate = false
    iter = 0
    decreased = 5
    while !terminate
        println("")
        @info "In Main loop $iter"
        #println("Current θ = $theta, xstar = $xstar")

        #Step 1
        xp, choleskyGramian = projectUnstructured!(Ahli, X, cholX, workspace, method=method)
        @info "Hessian conditioning" cond(H)
        #@info "Orthogonality" inner_product(eval_LMI(Ahli,xp)-X,eval_LMI(Ahli,rand(nli)), cholX)
        eval_LMI!(Ahli,xp,Xp)

        if isposdef(Symmetric(Xp)) #Productive step (modify Xp) (Step 2.)
            @info "Productive step"
            #Compute new theta level
            if method=="QR"
                #If method is QR, then the gramian was not computed before
                choleskyGramian = gramian!(Ahli, cholS, workspace)
            end
            #Get the matrices C and D and projection data
            pdata = projection_data!(Ahli, cli, dli, choleskyGramian, Xp, cholX, C, D)

            #Mininize theta level
            thetanew, Xptheta, xstarnew = getNextThetaLevel(cli, dli, pdata, Xp, xp, cholX)
            @info "Theta" ((cli'*xp)/(dli'*xp)) theta thetanew
            @assert thetanew <= theta + tol
            # Check if objective has decreased
            if theta - thetanew > tol || (xstar != undef && norm(xstar - xstarnew) > tol)
                decreased = 5
            else
                decreased -= 1
            end

            project_theta!(pdata, thetanew-tol, Xp, workspace.W[1])
            @info "Terminating condition" minimum(eigvals(X-workspace.W[1]))
            if isposdef(Symmetric(X-workspace.W[1]))
                @info "Terminating with new condition"
                terminate = true
            end

            #Update theta and xstar
            theta = thetanew
            xstar = xstarnew

            #If objective has not decreased sufficiently in 5 productive steps,
            #terminate
            if decreased < 0
                @info "Terminating without decrease"
                terminate = true
            end
        else #Unproductive step
            @info "Unproductive step" minimum(eigvals(Xp))
            Xptheta = Xp
        end

        #Step 3.
        #Compute step size gamma > 0 with the new Xp(θ) (productive or not)
        oldX = copy(X)
        getNextStep(X,Xptheta,cholX,workspace)
        @info "X p.d.-ness" minimum(eigvals(X))
        @info "Update difference" norm(X-oldX)
        cholX = cholesky(X)

        if norm(inv(cholX)) > stoptol
            @info "Terminating" norm(inv(cholX))
            terminate = true #Probably infeasible at this theta level
        end

        iter += 1
        ##println("Xp = $Xp, xp = $xp")
    end
    #Substitute back
    (xih, Xih) = substituteBackInhomogeneous(xstar,Xptheta,li,n)
    return LMISolution(xih,Symmetric(Xih),c'*xih,nothing,"success",probtype)
end

"""
    convertToLIHomogeneous(A,B;c=nothing)

Converts the LMI Ax+B into a homogeneous form Ãx̃ with Ã injective and
returns the basis of the null-space of A in column vector form

"""
function convertToLIHomogeneous!(A::Array{Matrix{Float64}},B::Matrix{Float64},T::Matrix{Float64})
    n = length(A)
    m = size(A[1],1)
    v = div(m*(m+1),2)
    isorth = nothing

    #TODO: Optimize for space usage

    #Select linearly independent set
    Avec = view(T,1:v,1:n) #Use part of T
    for k in 1:n
        for i in 1:m
            for j in i:m
                Avec[m*(i-1)-div(i*(i-1),2)+j,k] = A[k][i,j]
            end
        end
    end
    li, E = selectLIColumns!(Avec,basis=true)
    k = length(li)

    #Homogenize problem
    Ah = Array{Matrix{Float64}}(undef,k+1) #TODO: Optimize space usage
    for j in li
        Ah[j] = [A[j] zeros(m,1); zeros(1,m) 0]
    end
    Ah[end] = [B zeros(m,1);zeros(1,m) 1]

    return Ah, li, E
end

function substituteBackInhomogeneous(xli,X,li,n)
    #Substitute back solution
    xih = zeros(n+1)
    for j in 1:length(li)
        xih[li[j]] = xli[j]
    end

    #Substitute and return inhomogeneous solution
    xih = xih/xli[end]
    Xih = X/xli[end]
    return (xih[1:end-1],Xih[1:end-1,1:end-1])
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
function solveUnstructuredHomogeneous(A,workspace;tol=1e15,method="cholesky")
    n = length(A)
    m = size(A[1],1)

    #Initiate
    X = I(m)
    cholX = cholesky(I(m))
    x = zeros(n)
    Xp = workspace.W[1]

    while true
        #Project onto range of A
        println("Computing projection")
        @time x, _ = projectUnstructured!(A,X,cholX,workspace,method=method)
        eval_LMI!(A,x,Xp)

        #println("Eigvals Xp: ",min(eigvals(Xp)...))
        if isposdef(Symmetric(Xp))
            return (x,Xp)
        else
            #Compute step size gamma > 0
            getNextStep(X,Xp,cholX,workspace)
            cholX = cholesky(X)
        end
        if norm(inv(cholX)) > tol
            return (nothing,X) #Probably infeasible
        end
    end
end

"""
    getNextStep(X,Xp,cholX,W;search=false)

Computes the next step using a line search or the crude bound

"""
function getNextStep(X,Xp,cholX,workspace;search=false)
    #Unpack
    W = workspace.W[1]
    R = workspace.W[2]
    #sqrtinvX = sqrt(inv(cholX))
    #psi = sqrtinvX * (Xp - X) * sqrtinvX
    L = cholX.L

    W .= Xp.-X #Stores the psi matrix
    ldiv!(L,W)
    rdiv!(W,L')
    if search
        #TODO Line search for minimizing pi(gamma)
        error("Not implemented yet")
    else
        #Just compute the crude lower bound on pi(gamma)
        rho_inf = maximum(abs.(eigvals!(W)))
        gamma = 1/(1+rho_inf)
        W .= X.-gamma.*(Xp.-X)
        @info "Condition X-γ(Xp-X)" cond(W) gamma
        R = cholesky(Symmetric(W))
        ldiv!(W,R.L,X)
        mul!(X,W',W)
        X.=Symmetric(X)
        #X .= Symmetric(X*inv(W)*X)
        @info "X p.d.-ness 1." minimum(eigvals(X)) minimum(eigvals(Symmetric(W))) cond(W)
        #Need to symmetrize (rounding)
    end
end

"""
     getNextThetaLevel(C, xc, D, xd, X, Xp, xp, invX; tol=1e-12, low=-1e16)

Computes the lowest possible theta <= theta0 level such that the projection onto
the subspace corresponding to this level set is still positive definite
Returns theta and also the projection Xptheta and the point xptheta corresponding to it

"""
function getNextThetaLevel(c, d, pdata, Xp, xp, cholX; tol=1e-8, low=-1e16)
    #Compute the coefficients for the line-search
    m = size(Xp,1)
    Im = I(m)

    C, D, xc, xd, DXp, CXp, CC, DD, CD = pdata

    #Perform a line search starting at theta0
    alpha = 1
    terminate = false
    double = true

    theta = (c'*xp) / (d'*xp) #Starting value of theta
    thetasave = theta
    Xptheta = Xp #Starting value of Xptheta

    thetapd = theta #Last saved theta such that Xptheta > 0 (p.d.)

    iter = 0 #Iteration count (for testing)
    while !terminate & (theta > low)
        if alpha > 0
            #println("α = $alpha, θ = $theta")
        end
        #Decrease current level
        theta = theta - alpha
        #Check if the projection is positive definite
        rest = (CXp - theta*DXp)*(C-theta*D)/(CC-2*theta*CD+theta^2*DD)
        Xptheta = Xp - rest
        if alpha > 0
            #println(minimum(eigvals(Xptheta)))
            #println(isposdef(Hermitian(Xptheta)))
        end
        if isposdef(Hermitian(Xptheta))
            thetapd = theta #Save theta
            if double #If didn't find max stepsize yet, then double
                alpha = 2*alpha
            end
        else
            #println("Not posdef")
            double = false #Stop searching for the upper bound on the step size
            #Go back to the previous level and half the step size
            theta += alpha
            alpha = alpha/2
        end
        #Terminate binary search if precision level is met or cancellation occurs
        if alpha < tol || theta == theta - alpha
            terminate = true
        end
        iter += 1
    end
    #Compute again Xp(θ) and xp(θ) with the last saved theta
    @assert thetapd <= thetasave
    denom = (CC-2*thetapd*CD+thetapd^2*DD)
    Xptheta = Xp - (CXp - thetapd*DXp)*(C-thetapd*D)/denom
    xptheta = xp - (c - d*thetapd)' * xp * (xc - thetapd*xd)/denom #Use prop. of C, D
    return thetapd, Xptheta, xptheta
end
