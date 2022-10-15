using LinearAlgebra
include("solver.jl")
#Generate Data
n = 10
m = 20
A = Array{Matrix{Float64}}(undef,n)
for j=1:n
    Z = 10*(rand(m,m) .- 0.5)
    A[j] = Z+Z'
end
B = 10*(rand(m,m) .- 0.5)
B = B+B'

## ---
#Solve LMI

(x,X) = solveUnstructured(A,B,method="QR")

#Check solution
println("Solution correct: ",isposdef(eval(A,x)+B))
println("Solution vector: ",x)
