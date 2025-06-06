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

function hv_SU_update(A, params)
    Ni, Nj = size(A)
    D, d = size(A[1])[[1,5]]
    h = hamiltonian(Heisenberg(Ni,Nj,-1.0,-1.0,1.0))
    # h = ein"ab,cd ->abcd"(I(d),I(d)) # for testing purpose
    exp_h = _arraytype(A[1])(reshape(exp(-params.SUτ * reshape(permutedims(h,(1,3,2,4)),d^2,d^2)), d,d,d,d))
    Ah = Zygote.Buffer(A)
    for j in 1:Nj, i in 1:Ni
        jr = mod1(j + 1, Nj)
        AAh_h = ein"(abgfh,gcdei),hijk->fabjcdek"(A[i,j], A[i,jr], exp_h)
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

    return Av
end

function hv_FU_update(A, params, rt) # does not work 
    _, M = build_M(A, params)
    rt′ = Zygote.@ignore leading_boundary(rt, M, params.boundary_alg)
    Zygote.@ignore params.reuse_env && update!(rt, rt′)
    env = Zygote.@ignore VUMPSEnv(rt′, M, params.boundary_alg)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    χ = size(ACu[1],1)
    re(x) = reshape(x, (χ,D,D,χ))
    Ni, Nj = size(A)
    D, d = size(A[1])[[1,5]]
    h = hamiltonian(Heisenberg(Ni,Nj,-1.0,-1.0,1.0))
    # h = ein"ab,cd ->abcd"(I(d),I(d)) # for testing purpose
    exp_h = _arraytype(A[1])(reshape(exp(-params.SUτ * reshape(permutedims(h,(1,3,2,4)),d^2,d^2)), d,d,d,d))
    Ah = Zygote.Buffer(A)
    for j in 1:Nj, i in 1:Ni
        jr = mod1(j + 1, Nj)
        ir = mod1(i + 1, Ni)
        n = sum(ein"(((agj,abc),gkhb),jkl),(((fio,cef),hnie),lno) ->"(FLo[i,j],ACu[i,j],M[i,j],conj(ACd[ir,j]),FRo[i,jr],ARu[i,jr],M[i,jr],conj(ARd[ir,jr])))
        AAh_h = ein"((((pfoq,paju),abgfh),ubkt),(((qenr,rdms),gcdei),tcls)),hivw->ojkvlmnw"(re(ACu[i,j]), re(FLo[i,j]), A[i,j], re(conj(ACd[ir,j])), re(ARu[i,jr]), re(FRo[i,jr]), A[i,jr], re(conj(ARd[ir,jr])), exp_h) / n
        U, S, V = svd(reshape(AAh_h, D^3*d, D^3*d))
        Ah[i,j] = permutedims(reshape(U[:,1:D] * Diagonal(sqrt.(S[1:D])), D,D,D,d,D), (2,3,5,1,4))
        Ah[i,jr] = reshape(Diagonal(sqrt.(S[1:D])) * V'[1:D,:], D,D,D,D,d)
    end
    Ah = copy(Ah)
    # return Ah
    Av = Zygote.Buffer(Ah)
    for j in 1:Nj, i in 1:Ni
        ir = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni) 
        n = sum(ein"(((abc,aeg),ehfb),cfi),(gjl,(jmkh,(ikn,lmn))) -> "(ACu[i,j],FLu[i,j],M[i,j],FRu[i,j],FLo[ir,j],M[ir,j],FRo[ir,j],conj(ACd[irr,j])))
        AAh_v = ein"((((uajp,ubkt),bgfah),pfoq),(((tcls,sdmr),cdegi),qenr)),hivw-> ojkvldnw"(re(ACu[i,j]),re(FLu[i,j]),A[i,j],re(FRu[i,j]),re(FLo[ir,j]),re(conj(ACd[irr,j])),A[ir,j],re(FRo[ir,j]), exp_h) / n
        U, S, V = svd(reshape(AAh_v, D^3*d, D^3*d))
        Av[i,j] = permutedims(reshape(U[:,1:D] * Diagonal(sqrt.(S[1:D])), D,D,D,d,D), (3,5,1,2,4))
        Av[ir,j] = permutedims(reshape(Diagonal(sqrt.(S[1:D])) * V'[1:D,:], D,D,D,D,d), (2,3,4,1,5))
    end
    Av = copy(Av)

    return Av
end


"""
    takagi_decomposition(M; D_trunc)

Perform Takagi factorization for complex symmetric matrices. 
Decomposes matrix `M` into `M = A * transpose(A)` where:
- `M` is a complex symmetric matrix (M = M^T)
- `D_trunc`: truncation dimension to retain in the decomposition

This implementation:
1. Diagonalizes M†M to find eigenvectors (V) and eigenvalues (λ)
2. Extracts singular values from diagonal matrix D = VᵀMV
3. Constructs matrix A using first D_trunc eigenvectors scaled by sqrt(singular values)

Primarily used for SU parameterization in iPEPS tensor network simulations.
"""
function takagi_decomposition(M; D_trunc)
    norm(M - transpose(M)) < 1e-10 || throw(ArgumentError("M should be a complex symmetric matrix"))
    # Diagonalize M†M and sort eigenvectors by descending eigenvalues
    _, V = eigen(M'*M; sortby=x->-x)
    
    # Calculate diagonal matrix of singular values squared (D = VᵀMV)
    D = diag(transpose(V) * M * V)
    
    # Construct decomposition matrix: A = V*_trunc * sqrt(D_trunc)
    A = conj(V[:,1:D_trunc]) * diagm(sqrt.(D[1:D_trunc]))
    
    return A
end

function one_bond_SU(A, params)
    Ni, Nj = 1, 1
    # Ni == Nj == 1 ||  throw(ArgumentError("Ni Nj should be 1"))
    D, d = size(A[1])[[1,5]]
    h = hamiltonian(Heisenberg(Ni,Nj,-1.0,-1.0,1.0))
    exp_h = _arraytype(A)(reshape(exp(-params.SUτ * reshape(permutedims(h,(1,3,2,4)),d^2,d^2)), d,d,d,d))
    Ah = Zygote.Buffer(A)
    for j in 1:Nj, i in 1:Ni
        jr = mod1(j + 1, Nj)
        AAh_h = reshape(ein"(abgfh,degci),hijk->fabjcdek"(A[i,j], A[i,jr], exp_h), D^3*d, D^3*d)
        A′ = takagi_decomposition(AAh_h; D_trunc=D)
        Ah[i,j] = permutedims(reshape(A′, D,D,D,d,D), (2,3,5,1,4))
    end
    return copy(Ah)
end

function build_A(A, params)
    return StructArray([A[:,:,:,:,:,i] for i in 1:length(unique(params.pattern))], params.pattern)
end

function build_A(A, params, rt)
    A = StructArray(A, params.pattern)
    if params.SUτ != 0.0
        for i in 1:4
            A = one_bond_SU(A, params)
            A = map(x->permutedims(x, (2,3,4,1,5)), A)
        end
        
        A = hv_SU_update(A, params)
        # A = hv_FU_update(A, params, rt)
        return A
    else
        return A
    end
end

function build_M(A, params)
    D = size(A[1], 1)
    len = length(unique(params.pattern))
    if params.ifflatten
        return StructArray([reshape(ein"abcde,fghme->afbgchdm"(A[i], conj(A[i])), D^2,D^2,D^2,D^2) for i in 1:len], params.pattern)
    else
        return A
    end
end