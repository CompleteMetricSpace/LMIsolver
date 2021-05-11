using LinearAlgebra
include("solver.jl")
#Generate Data
n = 30
m = 50
A = Array{Matrix{Float64}}(undef,n)
for j=1:n
    Z = 10*(rand(m,m) .- 0.5)
    A[j] = Z+Z'
end

## ---
#Solve LMI

(x,X) = solveUnstructuredHomogeneous(A,method="cholesky")

#Check solution
println("Solution correct: ",isposdef(eval(A,x)))
println("Solution vector: ",x)
