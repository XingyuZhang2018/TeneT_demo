abstract type HamiltonianModel end

function hamiltonian_trunc(model)
    h = hamiltonian(model)
    d = size(h, 1)
    U, S, V = svd(reshape(h,d^2,d^2))
    truc = sum(S .> 1e-10)
    h1 = U[:,1:truc] * Diagonal(S[1:truc]) 
    h2 = V[:,1:truc]'
    return reshape(h1, d,d,truc), reshape(h2, truc,d,d)
end

"""
    Heisenberg(Ni::Int,Nj::Int,Jx::T,Jy::T,Jz::T) where {T<:Real}
    
return a struct representing the `Ni`x`Nj` heisenberg model with couplings `Jz`, `Jx` and `Jy`
"""
@kwdef mutable struct Heisenberg <: HamiltonianModel
    Jx::Real = -1.0
    Jy::Real = -1.0
    Jz::Real = 1.0
    ifrotate::Bool = true
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
    if model.ifrotate
        h = ein"ijcd,kc,ld -> ijkl"(h,Sx*2,(Sx*2)')
    end
    return h
end

@kwdef mutable struct J1J2 <: HamiltonianModel
    J1::Real = 1.0
    J2::Real = 0.0
    ifrotate::Bool = true
end

function hamiltonian(model::J1J2)
    if model.ifrotate
        h = - ein"ij,kl -> ijkl"(Sx, Sx) -
            ein"ij,kl -> ijkl"(Sy, Sy) +
            ein"ij,kl -> ijkl"(Sz, Sz)
        h = ein"ijcd,kc,ld -> ijkl"(h,Sx*2,(Sx*2)')
        return h
    else
        h = ein"ij,kl -> ijkl"(Sx, Sx) +
            ein"ij,kl -> ijkl"(Sy, Sy) +
            ein"ij,kl -> ijkl"(Sz, Sz)
        return h
    end
end

# Shastry-Sutherland model
@kwdef mutable struct SS <: HamiltonianModel
    J1::Real = 1.0
    J2::Real = 0.0
end

function hamiltonian(::SS)
    h = ein"ij,kl -> ijkl"(Sx, Sx) +
        ein"ij,kl -> ijkl"(Sy, Sy) +
        ein"ij,kl -> ijkl"(Sz, Sz)
    return h
end