using TeneT: _arraytype
using OMEinsum
using Zygote
using ITensors
using LinearAlgebra: I

export Ising, Heisenberg, Heisenberg_bilayer
export hamiltonian, hamiltonian_hand

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

struct Heisenberg_bilayer <: HamiltonianModel
    Ni::Int
    Nj::Int
    J二::Real
    J⊥::Real
end
Heisenberg_bilayer(Ni,Nj) = Heisenberg_bilayer(Ni,Nj,1.0,1.0)

function hamiltonian(model::Heisenberg_bilayer)
    J二 = model.J二
    J⊥ = model.J⊥

    ampo = OpSum()
    sites = siteinds("S=1/2",4)

    ampo .+= J二/2, "S+",1,"S-",3
    ampo .+= J二/2, "S-",1,"S+",3
    ampo .+= J二, "Sz",1,"Sz",3

    ampo .+= J二/2, "S+",2,"S-",4
    ampo .+= J二/2, "S-",2,"S+",4
    ampo .+= J二, "Sz",2,"Sz",4
    
    H = MPO(ampo,sites)

    H1 = Array(H[1],inds(H[1])...)
    H2 = Array(H[2],inds(H[2])...)
    H3 = Array(H[3],inds(H[3])...)
    H4 = Array(H[4],inds(H[4])...)
    H二 = reshape(ein"iae,ijbf,jkcg,kdh->cdabghef"(H1,H2,H3,H4),4,4,4,4)

    ampo = OpSum()
    sites = siteinds("S=1/2",2)
    ampo .+= J⊥/2, "S+",1,"S-",2
    ampo .+= J⊥/2, "S-",1,"S+",2
    ampo .+= J⊥, "Sz",1,"Sz",2
    H = MPO(ampo,sites)

    H1 = Array(H[1],inds(H[1])...)
    H2 = Array(H[2],inds(H[2])...)
    H⊥ = reshape(ein"aij,apq->ipjq"(H1,H2),4,4)

    return H二, H⊥
end

kronplus(A) = mapreduce(x->kron(x[1], x[2]), +, A)

function hamiltonian_hand(model::Heisenberg_bilayer)
    J二 = model.J二
    J⊥ = model.J⊥

    op=ein"ab,cd->acbd"
    Sx1 = reshape(op(Sx, I(2)),4,4)
    Sx2 = reshape(op(I(2), Sx),4,4)
    Sy1 = reshape(op(Sy, I(2)),4,4)
    Sy2 = reshape(op(I(2), Sy),4,4)
    Sz1 = reshape(op(Sz, I(2)),4,4)
    Sz2 = reshape(op(I(2), Sz),4,4)

    HJ = J二 * mapreduce(x->kron(x[1], x[2]), +,
        [
         [Sx1,Sx1], [Sy1,Sy1], [Sz1,Sz1],
         [Sx2,Sx2], [Sy2,Sy2], [Sz2,Sz2]
        ]
    )

    H二 = HJ
    H二 = reshape(H二, 4,4,4,4)

    HJ = J⊥ * mapreduce(x->kron(x[1], x[2]), +,
        [
         [Sx,Sx], [Sy,Sy], [Sz,Sz]
        ]
    )

    H⊥ = HJ

    return H二, H⊥
end
