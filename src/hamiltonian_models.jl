abstract type HamiltonianModel end

"""
    Heisenberg(Ni::Int,Nj::Int,Jx::T,Jy::T,Jz::T) where {T<:Real}
    
return a struct representing the `Ni`x`Nj` heisenberg model with couplings `Jz`, `Jx` and `Jy`
"""
@kwdef mutable struct Heisenberg{T<:Real} <: HamiltonianModel
    Ni::Int = 1
    Nj::Int = 1
    Jx::T = -1.0
    Jy::T = -1.0
    Jz::T = 1.0
end

const Sx = Float64[0 1; 1 0]/2
const Sy = ComplexF64[0 -1im; 1im 0]/2
const Sz = Float64[1 0; 0 -1]/2
"""
    hamiltonian(model::Heisenberg)

return the heisenberg hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::Heisenberg)
    h = model.Jx * ein"ij,kl -> ijkl"(Sx, Sx) +
        model.Jy * ein"ij,kl -> ijkl"(Sy, Sy) +
        model.Jz * ein"ij,kl -> ijkl"(Sz, Sz)
    h = ein"ijcd,kc,ld -> ijkl"(h,Sx*2,(Sx*2)')
    U, S, V = svd(reshape(h,4,4))
    truc = sum(S .> 1e-10)
    h1 = U[:,1:truc] * Diagonal(S[1:truc]) 
    h2 = V[:,1:truc]'
    d = size(Sx, 1)
    return reshape(h1, d,d,truc), reshape(h2, truc,d,d)
end
