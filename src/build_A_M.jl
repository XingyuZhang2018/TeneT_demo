Zygote.@adjoint function LinearAlgebra.svd(A)
    res = LinearAlgebra.svd(A)
    res, function (dy)
        dU, dS, dVt = dy
        return (svd_back(res.U, res.S, res.V, dU, dS, dVt === nothing ? nothing : dVt'),)
    end
end

struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero

"""
    svd_back(U, S, V, dU, dS, dV)

adjoint for SVD decomposition.

References:
    https://j-towns.github.io/papers/svd-derivative.pdf
    https://giggleliu.github.io/2019/04/02/einsumbp.html
"""
function svd_back(U::AbstractArray, S::AbstractArray{T}, V, dU, dS, dV; η::Real=1e-40) where T
    all(x -> x isa Nothing, (dU, dS, dV)) && return nothing
    η = T(η)
    NS = length(S)
    S2 = S .^ 2
    Sinv = @. S/(S2+η)
    F = S2' .- S2
    F ./= (F.^ 2 .+ η)

    res = ZeroAdder()
    if !(dU isa Nothing)
        UdU = U'*dU
        J = F.*(UdU)
        res += (J+J')*LinearAlgebra.Diagonal(S) + LinearAlgebra.Diagonal(1im*imag(LinearAlgebra.diag(UdU)) .* Sinv)
    end
    if !(dV isa Nothing)
        VdV = V'*dV
        K = F.*(VdV)
        res += LinearAlgebra.Diagonal(S) * (K+K')
    end
    if !(dS isa Nothing)
        res += LinearAlgebra.Diagonal(dS)
    end

    res = U*res*V'

    if !(dU isa Nothing) && size(U, 1) != size(U, 2)
        res += (dU - U* (U'*dU)) * LinearAlgebra.Diagonal(Sinv) * V'
    end

    if !(dV isa Nothing) && size(V, 1) != size(V, 2)
        res = res + U * LinearAlgebra.Diagonal(Sinv) * (dV' - (dV'*V)*V')
    end
    res
end

function build_A(A, params::iPEPSOptimize)
    D, d, Ni, Nj = Zygote.@ignore size(A)[[1,5,6,7]]
    A′ = [A[:,:,:,:,:,i,j] for i = 1:Ni, j = 1:Nj]
    if params.SUτ != 0.0
        h = hamiltonian(Heisenberg(Ni,Nj,-1.0,-1.0,1.0))
        # h = ein"ab,cd ->abcd"(I(d),I(d)) # for testing purpose
        exp_h = _arraytype(A)(reshape(exp(-params.SUτ * reshape(permutedims(h,(1,3,2,4)),d^2,d^2)), d,d,d,d))
        Ah = Zygote.Buffer(A′)
        for j in 1:Nj, i in 1:Ni
            jr = mod1(j + 1, Nj)
            AAh_h = ein"(abgfh,gcdei),hijk->fabjcdek"(A′[i,j], A′[i,jr], exp_h)
            U, S, V = svd(reshape(AAh_h, D^3*d, D^3*d))
            Ah[i,j] = permutedims(reshape(U[:,1:D] * Diagonal(sqrt.(S[1:D])), D,D,D,d,D), (2,3,5,1,4))
            Ah[i,jr] = reshape(Diagonal(sqrt.(S[1:D])) * V'[1:D,:], D,D,D,D,d)
        end
        Ah = copy(Ah)

        Av = Zygote.Buffer(Ah)
        for j in 1:Nj, i in 1:Ni
            ir = mod1(i + 1, Ni)
            AAh_v = ein"(bgfah,cdegi),hijk->fabjcdek"(Ah[i,j], Ah[ir,j], exp_h)
            U, S, V = svd(reshape(AAh_v, D^3*d, D^3*d))
            Av[i,j] = permutedims(reshape(U[:,1:D] * Diagonal(sqrt.(S[1:D])), D,D,D,d,D), (3,5,1,2,4))
            Av[ir,j] = permutedims(reshape(Diagonal(sqrt.(S[1:D])) * V'[1:D,:], D,D,D,D,d), (2,3,4,1,5))
        end
        Av = copy(Av)

        return Av/norm(Av)
    else
        return A′/norm(A′)
    end
end

function build_M(A, params)
    D = size(A[1], 1)
    ap = StructArray([reshape(ein"abcde,fghmn->afbgchdmen"(A[1], conj(A[1])), D^2,D^2,D^2,D^2, 2,2)], params.pattern)
    M  = StructArray([ein"abcdee->abcd"(ap[1])], params.pattern)
    return ap, M
end