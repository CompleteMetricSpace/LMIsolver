
@enum MatrixType Full Sym Diag Antisym Toeplitz

struct MatrixVar
    identifier::String
    type::MatrixType
    size::Tuple{Int64,Int64}
end

abstract type MatrixTerm end

struct SymmetricTerm <: MatrixTerm
    var::MatrixVar
    transposed::Bool
    left::Matrix{<:Real}
    right::Matrix{<:Real}
    scalar::Real
    size::Tuple{Int64,Int64}
end

struct FullTerm <: MatrixTerm
    var::MatrixVar
    transposed::Bool
    left::Matrix{<:Real}
    right::Matrix{<:Real}
    scalar::Real
    size::Tuple{Int64,Int64}
end

struct LeftTerm <: MatrixTerm
    var::MatrixVar
    transposed::Bool
    left::Matrix{<:Real}
    scalar::Real
    size::Tuple{Int64,Int64}
end

struct RightTerm <: MatrixTerm
    var::MatrixVar
    transposed::Bool
    right::Matrix{<:Real}
    scalar::Real
    size::Tuple{Int64,Int64}
end

struct ConstantTerm <: MatrixTerm
    constant::Matrix{<:Real}
    size::Tuple{Int64,Int64}
end

struct TermSum
    terms::Array{<:MatrixTerm}
    size::Tuple{Int64,Int64}
end

struct LMI
    lmi::Matrix{TermSum}
    vars::Array{MatrixVar}
    size::Tuple{Array{Int64},Array{Int64}}
end
