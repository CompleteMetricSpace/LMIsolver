using LinearAlgebra
using BenchmarkTools
using Test
include("solver.jl")
include("projection.jl")


function test_LP(;A=nothing, b=nothing, c=nothing, n=5, m=5)
    if A == nothing
        A = rand(m,n)
    end
    if b == nothing
        x = rand(n)
        b = A*x + ones(m,1)
    end
    if c == nothing

    end
    error("Not implemented yet")
end


function random_feas_LMI(n,m;coef_size=5)
    A = Array{Matrix{Float64}}(undef,n)
    for j in 1:n
        C = coef_size*(2*rand(m,m).-1)
        A[j] = C+C'
    end
    x = rand(n)
    D = coef_size*(2*rand(m,m).-1)
    B = -eval(A,x) + D*D'
    return A, B, x
end

function test_feas(;coef_size=5,n=5,m=5,tol=1e-8)
    A, B, x = random_feas_LMI(n,m,coef_size=coef_size)
    @time lmisol = solveUnstructured(A,B)
    print("lmisol: $lmisol")
    x_sol, X_sol = lmisol.x, lmisol.X
    C1 = eval(A,x_sol) + B
    @test norm(X_sol - C1) < tol
    @test isposdef!(C1)
end

function benchmark_feas(;coef_size=5,n=5,m=5)
    @benchmark solveUnstructured(A,B) setup=((A, B, x) = random_feas_LMI($n,$m,coef_size=$coef_size))
end

function benchmark_gramian(;coef_size=5,n=5,m=5)
    @benchmark gramian(A,B) setup=((A, B, x) = random_feas_LMI($n,$m,coef_size=$coef_size))
end
