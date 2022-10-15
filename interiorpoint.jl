using LinearAlgebra
using Logging



include("lmitools.jl")

macro dinfo(exs...)
    return Expr(:macrocall,Symbol("@logmsg"), nothing, Logging.LogLevel(0), (esc.(exs))...)
end
macro uinfo(exs...)
    return Expr(:macrocall,Symbol("@logmsg"), nothing, Logging.LogLevel(1), (esc.(exs))...)
end

function interior_feas!(A, B; tol=1e-4, tol_newton=1e-8, t0=1.0, μ=20.0, R=1e10)
    n = length(A)
    m = size(A[1],1)
    #Allocate workspace matrices
    #TODO: Need only m*(m+1)/2 instead of m^2 rows
    prehessian = zeros(m^2,n+1)
    hessian = zeros(n+1,n+1)
    workspace = zeros(m,m)

    @dinfo "Size of prehessian" sizeof(prehessian)
    @dinfo "Size of hessian" sizeof(hessian)
    @dinfo "Size of workspace" sizeof(workspace)

    #Create LMI-feasibility problem with margin:
    Amar = Array{Matrix{Float64}}(undef,n+1)
    for i in 1:n
        Amar[i] = A[i]
    end
    Amar[end] = -1.0*I(m)
    en = zeros(n+1)
    en[end] = -1 #Maximization

    x = rand(n+1)
    x = min(R/2,1)*x/norm(x[1:end-1])
    x[end] = minimum(eigvals(eval_LMI(A,B,x[1:end-1]))) - 1
    t = t0
    while m/t > tol
        @uinfo "External iteration" t x[end] norm(x[1:end-1])
        x, flag = newton(Amar, B, en, x, t, R, workspace, hessian, prehessian,
                         tol=tol_newton, indx=1:n)
        if flag == 2
            error("Newton method did not converge at t = $t")
        elseif flag == 1
            printstyled("Warning: Slow convergence for Newton method at t = $t\n", color=:red)
        end
        t *= μ
    end
    return x[end], x[1:end-1]
end

function newton(A, B, c, x0, t, R, workspace, hessian, prehessian;
    tol=1e-8, α=1e-6, γ=0.1, β=0.5, p=1, indx=nothing, step_search=true, MAX_ITER=150)
    n = length(A)
    m = size(A[1],1)
    if isnothing(indx)
        indx = 1:n
    end

    #Allocate gradient
    grad = zeros(n)
    newton_dec = Inf

    x_points = []
    obj_values = []

    x = x0

    cur_lmi = eval_LMI_cholesky!(A, B, x, workspace) #Current LMI
    logdet_cur_lmi = logdet(cur_lmi) #logdet of current LMI
    fx = t*c'*x - logdet_cur_lmi - sum(log(R^2-x[i]^2) for i in indx) #Current objective value
    next_lmi = nothing

    term_flag = 0

    iter = 0
    while true
        #Pick descent direction
        push!(x_points,x)
        push!(obj_values,fx)
        iter += 1

        @dinfo "Internal iteration: $iter"

        #Compute gradient and hessian, prehessian contains then inv(L)A[i]inv(L')
        #where LL' = cur_lmi = Ax+B
        grad_and_hess_feas!(A, B, c, x, t, R, indx, grad, hessian, prehessian, cur_lmi)
        norm_grad = norm(grad)

        @dinfo "Value and gradient" fx norm(grad)
        if norm_grad < tol || newton_dec < tol
            term_flag = 0
            break
        elseif iter > 5 && abs(obj_values[end-5]-obj_values[end]) < tol
            term_flag = 1 #Objective didn't decrease, slow convergence
            break
        elseif iter > MAX_ITER
            println("Maximum iterations exceeded. Continue?")
            cont = readline()
            if cont == "yes"
                MAX_ITER += 40
            else
                term_flag = 2 #MAX_ITER was exceeded
                break
            end
        end
        d = -grad
        if cond(hessian) < 1e15 #If Hessian is invertible, pick Newton step
            d = - (cholesky(hessian) \ grad)
            if grad'*d/(norm_grad*norm(d)) > -α*norm_grad^p #Angle condition
                @dinfo "Angle condition not satisfied" left=(grad'*d/(norm_grad*norm(d))) right=(-α*norm_grad^p)
                d = -grad #Pick gradient step
            else
                @dinfo "Newton step taken"
            end
        end
        newton_dec = sqrt(-grad'*d)
        @dinfo "Newton decrement and hessian conditioning" newton_dec cond(hessian) norm(d)
        if step_search
            #Pick step size with the Armijo rule
            eval_LMI_vec!(prehessian, d, workspace, m) #Evaluate Td (T = prehessian)
            eigenvals = eigvals!(workspace) #Precompute the eigenvals of Td for efficient line-search
            step = 1
            xnext = x + step*d
            ispd = all(1+step*eigenvals[i] > 0 for i in 1:m) && all(abs(xnext[i]) < R for i in indx)
            # Use logdet(A(x+sd) + B) = logdet(Ax+B + sAd) = log(det(L)det(I+s*inv(L)*Ad*inv(L'))det(L'))
            # = logdet(Ax+B) + logdet(I+s*T) = logdet_cur_lmi + sum(log(1+s*eig(Td)))
            # where T[i] = inv(L)*A[i]*inv(L')
            logdet_next_lmi = ispd ? logdet_cur_lmi + sum(log(1+step*eigenvals[i]) for i in 1:m) : nothing
            fxnext = ispd ? t*c'*xnext - logdet_next_lmi - sum(log(R^2-xnext[i]^2) for i in indx) : nothing
            @dinfo "Step search: step = $step, ispd = $ispd"
            while !ispd || (fxnext - fx > 1e-10 + γ*step*grad'*d)
                step *= β
                xnext = x + step*d
                ispd = all(1+step*eigenvals[i] > 0 for i in 1:m) && all(abs(xnext[i]) < R for i in indx)
                logdet_next_lmi = ispd ? logdet_cur_lmi + sum(log(1+step*eigenvals[i]) for i in 1:m) : nothing
                fxnext = ispd ? t*c'*xnext - logdet_next_lmi - sum(log(R^2-xnext[i]^2) for i in indx) : nothing
                @dinfo "Step search: step = $step, ispd = $ispd"
                if step < 1e-16
                    @dinfo "Objective difference and Armijo" fxnext-fx γ*step*grad'*d ispd
                    error("Step is zero: Infeasible direction")
                end
            end
            #Calculate final next LMI
            next_lmi = eval_LMI_cholesky!(A, B, xnext, workspace)
        else
            #Fixed step-size as in Boyd's book
            step = newton_dec <= 0.25 ? 1 : 1/(1+newton_dec)
            xnext = x + step*d
            next_lmi = eval_LMI_cholesky!(A, B, xnext, workspace)
            logdet_next_lmi = logdet(next_lmi)
            fxnext = t*c'*xnext - logdet_next_lmi - sum(log(R^2-xnext[i]^2) for i in indx)
            @dinfo "Fixed step: step = $step, ispd = $ispd"
        end
        @dinfo "STEP SIZE TAKEN" step norm(step*d) x[end]
        #Update current point and gradient
        x = xnext
        cur_lmi = next_lmi
        logdet_cur_lmi = logdet_next_lmi
        fx = t*c'*x - logdet_cur_lmi - sum(log(R^2-x[i]^2) for i in indx) #Current objective value
    end
    return x, term_flag
end

"""
Computes the gradient and hessian from factored lmi
"""
function grad_and_hess_feas!(A, B, c, x, t, R, indx, G, H, T, F)
    n = length(A)
    m = size(A[1],1)
    # Precompute vector T given by
    #T = [(F.L \ A[i]) / F.U for i in 1:n]
    timeT = @elapsed begin
        for i in 1:n
            Tm = reshape(view(T, 1:m^2, i), m, m)
            ldiv!(Tm, F.U', A[i])
            rdiv!(Tm, F.U)
        end
    end
    timeH = @elapsed begin
        mul!(H,T',T) #TODO: H is symmetric, need only half the work
    end
    @dinfo "Calculation times T and H" timeT timeH
    #Compute additional terms for the constraints norm(x) < R
    for i in indx
        #|x[i]| < R for i in indx
        H[i,i] += 2/(R^2-x[i]^2) + 4*x[i]^2/(R^2-x[i]^2)^2
    end
    G .= [-tr(reshape(view(T, 1:m^2, i), m, m)) for i in 1:n]
    #G .= [-tr(T[i]) for i in 1:n]
    for i in indx
        G[i] += 2*x[i]/(R^2-x[i]^2) #|x[i]| < R for i in indx
    end
    G .+= t*c # Barrier with penalty t on c'*x
end
