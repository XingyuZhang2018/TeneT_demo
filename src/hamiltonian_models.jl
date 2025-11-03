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

function const_Sx(S::Real)
    dims = Int(2*S + 1)
    ms = [-S+i-1 for i in 1:dims]
    Sx = zeros(ComplexF64, dims, dims)
    for j in 1:dims, i in 1:dims
        if abs(i-j) == 1
            Sx[i,j] = 1/2 * sqrt(S*(S+1)-ms[i]*ms[j]) 
        end
    end
    return Sx
end

function const_Sy(S::Real)
    dims = Int(2*S + 1)
    ms = [-S+i-1 for i in 1:dims]
    Sy = zeros(ComplexF64, dims, dims)
    for j in 1:dims, i in 1:dims
        if i-j == 1
            Sy[i,j] = -1/2/1im * sqrt(S*(S+1)-ms[i]*ms[j]) 
        elseif j-i == 1
            Sy[i,j] =  1/2/1im * sqrt(S*(S+1)-ms[i]*ms[j]) 
        end
    end
    return Sy
end

function const_Sz(S::Real)
    dims = Int(2*S + 1)
    ms = [S-i+1 for i in 1:dims]
    Sz = zeros(ComplexF64, dims, dims)
    for i in 1:dims
        Sz[i,i] = ms[i]
    end
    return Sz
end

"""
    Heisenberg(Ni::Int,Nj::Int,Jx::T,Jy::T,Jz::T) where {T<:Real}
    
return a struct representing the `Ni`x`Nj` heisenberg model with couplings `Jz`, `Jx` and `Jy`
"""
@kwdef mutable struct Heisenberg <: HamiltonianModel
    S::Real = 1/2
    Jx::Real = -1.0
    Jy::Real = -1.0
    Jz::Real = 1.0
    ifrotate::Bool = true
end

"""
    hamiltonian(model::Heisenberg)

return the heisenberg hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian(model::Heisenberg)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    h = model.Jx * ein"ij,kl -> ijkl"(Sx, Sx) +
        model.Jy * ein"ij,kl -> ijkl"(Sy, Sy) +
        model.Jz * ein"ij,kl -> ijkl"(Sz, Sz)
    if model.ifrotate
        h = ein"ijcd,kc,ld -> ijkl"(h,Sx*2,(Sx*2)')
    end
    return h
end

@kwdef mutable struct J1J2 <: HamiltonianModel
    S::Real = 1/2
    J1::Real = 1.0
    J2::Real = 0.0
    ifrotate::Bool = true
end

function hamiltonian(model::J1J2)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
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
    S::Real = 1/2
    J1::Real = 1.0
    J2::Real = 0.0
end

function hamiltonian(::SS)
    S = model.S
    Sx = const_Sx(S)
    Sy = const_Sy(S)
    Sz = const_Sz(S)
    h = ein"ij,kl -> ijkl"(Sx, Sx) +
        ein"ij,kl -> ijkl"(Sy, Sy) +
        ein"ij,kl -> ijkl"(Sz, Sz)
    return h
end