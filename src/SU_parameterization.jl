function SU_parameterization(A, params; D_new)
    Ni, Nj = size(A)
    D, d = size(A[1])[[1,5]]
    h = hamiltonian(params.model)
    # h = ein"ab,cd ->abcd"(I(d),I(d)) # for testing purpose
    exp_h = _arraytype(A[1])(reshape(exp(-params.SUτ * reshape(permutedims(h,(1,3,2,4)),d^2,d^2)), d,d,d,d))
    Ah = Zygote.Buffer(A)
    if D_new > D
        for p in 1:length(A)
            Ah[p] = zeros(ComplexF64, D_new,D_new,D_new,D_new,d)
            Ah[p][1:D,1:D,1:D,1:D,:] = A[p]
        end
    else
        for p in 1:length(A)
            Ah[p] = A[p]
        end
    end
    for p in 1:length(A)
        i, j = Tuple(findfirst(==(p), A.pattern))
        jr = mod1(j + 1, Nj)
        AAh_h = ein"(abgfh,gcdei),hijk->fabjcdek"(Ah[i,j], Ah[i,jr], exp_h)
        size_AAh_h = size(AAh_h)
        U, S, V = svd(reshape(AAh_h, prod(size_AAh_h[1:4]), prod(size_AAh_h[5:8])))
        Ah[i,j][:,:,1:D_new,:,:] = permutedims(reshape(U[:,1:D_new] * Diagonal(sqrt.(S[1:D_new])), size_AAh_h[1:4]...,D_new), (2,3,5,1,4))
        Ah[i,jr][1:D_new,:,:,:,:] = reshape(Diagonal(sqrt.(S[1:D_new])) * V'[1:D_new,:], D_new,size_AAh_h[5:8]...)
        if D_new < D
            Ah[i,j][:,:,D_new+1:D,:,:] .= 0
            Ah[i,jr][D_new+1:D,:,:,:,:] .= 0   
        end
    end
    Ah = copy(Ah)

    Av = Zygote.Buffer(Ah)
    for p in 1:length(A)
        Av[p] = Ah[p]
    end
    for p in 1:length(A)
        i, j = Tuple(findfirst(==(p), A.pattern))
        ir = mod1(i + 1, Ni)
        AAh_v = ein"(bgfah,cdegi),hijk->fabjcdek"(Av[i,j], Av[ir,j], exp_h)
        size_AAh_v = size(AAh_v)
        U, S, V = svd(reshape(AAh_v, prod(size_AAh_v[1:4]), prod(size_AAh_v[5:8])))
        Av[i,j][:,1:D_new,:,:,:] = permutedims(reshape(U[:,1:D_new] * Diagonal(sqrt.(S[1:D_new])), size_AAh_v[1:4]...,D_new), (3,5,1,2,4))
        Av[ir,j][:,:,:,1:D_new,:] = permutedims(reshape(Diagonal(sqrt.(S[1:D_new])) * V'[1:D_new,:], D_new, size_AAh_v[5:8]...), (2,3,4,1,5))
        if D_new < D
            Av[i,j][:,D_new+1:D,:,:,:] .= 0
            Av[ir,j][:,:,:,D_new+1:D,:] .= 0
        end
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
        A′ = takagi_decomposition(AAh_h; D_new=D)
        Ah[i,j] = permutedims(reshape(A′, D,D,D,d,D), (2,3,5,1,4))
    end
    return copy(Ah)
end
