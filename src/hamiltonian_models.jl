using TeneT: _arraytype
using OMEinsum
using Zygote

export Ising, Heisenberg, J1J2
export hamiltonian

const isingβc = log(1+sqrt(2))/2

abstract type HamiltonianModel end

"""
    Ising(Ni::Int,Nj::Int,β)
    
return a struct representing the `Ni`x`Nj` Ising model with inverse temperature `β`
"""
struct Ising <: HamiltonianModel 
    Ni::Int
    Nj::Int
    β::Float64
end

"""
    Heisenberg(Ni::Int,Nj::Int,Jx::T,Jy::T,Jz::T) where {T<:Real}
    
return a struct representing the `Ni`x`Nj` heisenberg model with couplings `Jz`, `Jx` and `Jy`
"""
struct Heisenberg{T<:Real} <: HamiltonianModel
    Ni::Int
    Nj::Int
    Jx::T
    Jy::T
    Jz::T
end
Heisenberg(Ni,Nj) = Heisenberg(Ni,Nj,1.0,1.0,1.0)

"""
    J1J2(Ni::Int,Nj::Int,Jx::T,Jy::T,Jz::T) where {T<:Real}
    
return a struct representing the `Ni`x`Nj` J1J2 model with couplings `J1` and `J2`
"""
struct J1J2{T<:Real} <: HamiltonianModel
    Ni::Int
    Nj::Int
    J1::T
    J2::T
end
J1J2(Ni,Nj,J2) = J1J2(Ni,Nj,1.0,J2)

const Sx = Float64[0 1; 1 0]/2
const Sy = ComplexF64[0 -1im; 1im 0]/2
const Sz = Float64[1 0; 0 -1]/2
"""
    hamiltonian(model::Heisenberg)

return the heisenberg hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::Heisenberg)
    model.Jx * ein"ij,kl -> ijkl"(Sx, Sx) +
    model.Jy * ein"ij,kl -> ijkl"(Sy, Sy) +
    model.Jz * ein"ij,kl -> ijkl"(Sz, Sz)
end

"""
    hamiltonian(model::J1J2)

return the J1J2 hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::J1J2)
    hamiltonian(Heisenberg(model.Ni,model.Nj,model.J1,model.J1,model.J1)),
    hamiltonian(Heisenberg(model.Ni,model.Nj,model.J2,model.J2,model.J2))
end