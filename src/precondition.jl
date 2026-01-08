function precondition_invese_single_envir(A, grad, rt, params, restriction_ipeps, fδEi, iter_precond)
    # size(A) == (1,) || throw(Base.error("precondition only supports 1x1 unit cell currently"))
    if fδEi[3] <= iter_precond
        return grad
    end
    δ = fδEi[2]
    # A′ = restriction_ipeps(A)
    Gh, Gv = find_local_hermite_G(A, params)
    A′ = guage_transfer(A, [Gh, Gv], params)
    A′ = build_A(A′, params)
    M = build_M(A′, params) 
    # Random.seed!(4564135)
    # rt = VUMPSRuntime(M, 1, params.boundary_alg)
    # rt = leading_boundary(rt, M, params.boundary_alg)
    # # Zygote.@ignore params.reuse_env && update!(rt, rt′)

    env = VUMPSEnv(rt, M, params.boundary_alg)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env

    function re(x)
        χ,D = size(ACu[1])[[1,2]]
        D = Int(sqrt(D))
        reshape(x, χ, D, D, χ)
    end

    gradnew = deepcopy(grad)
    Ni,Nj = size(M)
    forloop_iter = params.forloop_iter
    pattern = M.pattern
    for p in 1:length(M)
        i, j = Tuple(findfirst(==(p), pattern))
        ir = Ni + 1 - i
        n = contract_n1(FLo[i,j], ACu[i,j], A′[i,j], ACd[ir,j], FRo[i,j]; forloop_iter)
        # @show n
        # P = ein"((iaej,jbfk),lcgk),idhl->abcdefgh"(re(FLo[i,j]),re(conj(ACd[ir,j])),re(FRo[i,j]),re(ACu[i,j]))/n
        # λ,_ = eigen(reshape(P,D^4,D^4))
        # @show real(λ[end-10:end])
        if params.ifflatten
            gradnew[:,:,:,:,:,p], _ = linsolve(x->δ * x + TeneT.Mumap_forloop(re(ACu[i,j]),re(conj(ACd[ir,j])),re(FLo[i,j]),re(FRo[i,j]),x;forloop_iter)/n, grad[:,:,:,:,:,p]; isposdef = true, maxiter=1, verbosity=0)
        else
            gradnew[:,:,:,:,:,p], _ = linsolve(x->δ * x + TeneT.Mumap_forloop(ACu[i,j],ACd[ir,j],FLo[i,j],FRo[i,j],x; forloop_iter)/n, grad[:,:,:,:,:,p]; isposdef = true, maxiter=1, verbosity=0)
            irr = mod1(i - 1, Ni)
            jr = mod1(j - 1, Nj)
            G_temp = [Gh[:,:,pattern[i,jr]], inv(Gv[:,:,p]), inv(Gh[:,:,p]), Gv[:,:,pattern[irr,j]]]
            gradnew[:,:,:,:,:,p] = local_gauge_contraction(gradnew[:,:,:,:,:,p], G_temp)

            
            # gradnew[:,:,:,:,:,q], _ = linsolve(grad[:,:,:,:,:,q]; isposdef = true, maxiter=1, verbosity=0) do x
            #     irr = mod1(i - 1, Ni)
            #     jr = mod1(j - 1, Nj) 
            #     G_temp = [inv(Gh[:,:,pattern[i,jr]]), Gv[:,:,q], Gh[:,:,q], inv(Gv[:,:,pattern[irr,j]])]
            #     function f(Au)
            #         ForwardDiff.gradient(y -> real((@tensor Mumap_forloop(ACu[i,j],ACd[ir,j],FLo[i,j],FRo[i,j], local_gauge_contraction(Au[:,:,:,:,:,q], G_temp); ifparallel)[a,b,c,d,p] * conj(local_gauge_contraction(y, G_temp))[a,b,c,d,p])/n), A[:,:,:,:,:,q])
            #     end
            #     gN = ForwardDiff.derivative(t -> f(A + t * x), 0.0)
            #     return δ * x + gN
            # end
        end
            # gradnew[:,:,:,:,:,p], _ = linsolve(grad[:,:,:,:,:,p]; isposdef = true, maxiter=1, verbosity=0) do x
            #     function f(Au) 
            #         G = find_local_min_norm_G(Au)
            #         ForwardDiff.gradient(y -> (@tensor TeneT.Mumap_forloop(ACu[i,j],conj(ACd[ir,j]),FLo[i,j],FRo[i,j],guage_transfer(A, G); forloop_iter)[a,b,c,d,p] * conj(guage_transfer(y, G))[a,b,c,d,p])/n, Au)
            #     end
            #     gN = ForwardDiff.derivative(t -> f(A + t * x), 0.0)
            #     return δ * x + gN
            # end
        # end
    end

    return gradnew
end

function precondition_invese_hessian(A, grad, rt, rt′, params, restriction_ipeps, fδEi, iter_precond)
    # size(A) == (1, ) || throw(Base.error("precondition only supports 1x1 unit cell currently"))
    # if fδEi[2] > 0.01 || fδEi[3] <= iter_precond
    #     return grad
    # end

    δ = fδEi[2]

    function contract_n1(FLo, ACu, A, Ap, ACd, FRo; forloop_iter)
        D1,D2,D3,D4,_ = size(A)
        # M = reshape(ein"abcde,fghme->afbgchdm"(A, Ap), D1^2,D2^2,D3^2,D4^2)
        @tensor M[a,f,b,g,c,h,d,m] := A[a,b,c,d,e] * Ap[f,g,h,m,e]
        return sum(oc1_leg3(FLo, ACu, M, ACd, FRo; forloop_iter))
    end
    
    function build_M(A, Ap, params)
        D = size(A[1], 1)
        len = length(unique(params.pattern))
        if params.ifflatten
            # return StructArray([reshape(ein"abcde,fghme->afbgchdm"(A[i], Ap[i]), D^2,D^2,D^2,D^2) for i in 1:len], params.pattern)
            return StructArray([begin
                @tensor M[a,f,b,g,c,h,d,m] := A[i][a,b,c,d,e] * Ap[i][f,g,h,m,e]
                reshape(M, D^2,D^2,D^2,D^2)
            end for i in 1:len], params.pattern)
        else
            throw(Base.error("precondition only supports ifflatten=true currently"))
        end
    end

    function ipeps_norm(A, Ap, rt, rt′, params::iPEPSOptimize)
        M = build_M(A, Ap, params)
        rt, _ = leading_boundary(rt, M, params.boundary_alg)
        Zygote.@ignore update!(rt′, rt)
        env = VUMPSEnv(rt, M, params.boundary_alg)
        @unpack ACu, ACd, FLo, FRo = env
        n = contract_n1(FLo[1], ACu[1], A[1], Ap[1], conj(ACd[1]), FRo[1]; params.forloop_iter)
        return real(n)
    end

    function f(A, Ap)
        A = restriction_ipeps(A)
        A = build_A(A, params)
        Ap = restriction_ipeps(Ap)
        Ap = build_A(Ap, params)
        return ipeps_norm(A, Ap, rt, rt′, params)
    end

    @show dot(grad, Zygote.gradient(x1 -> f(x1, conj(A)), A)[1])
    @show ForwardDiff.derivative(t -> f(t * A, conj(A)), 0)[1]
    # @show ForwardDiff.gradient(x2 -> dot(grad, Zygote.gradient(x1 -> f(x1, x2), A)[1]), conj(A))[1] 
    # gradnew, info = linsolve(v->v*δ + hessian_vec_prod(f, A, v), grad; isposdef = true, maxiter=1)
    # @show info
    return grad
end

function precondition_invese_BP_envir(A, grad, rt, params, restriction_ipeps, fδEi)
    # size(A) == (1,) || throw(Base.error("precondition only supports 1x1 unit cell currently"))
    if fδEi[2] > 0.01 || fδEi[3] <= 20
        return grad
    end
    δ = fδEi[2]
    A = restriction_ipeps(A)
    A = build_A(A, params)
    _, M = build_M(A, params) 

    D = Int(sqrt(size(M[1],1)))
    B = _arraytype(M[1])(randn(ComplexF64, D^2))
    error = 1.0
    Z = 1.0
    for i in 1:100
        # B = ein"((abcd,d),c),b -> a"(M[1],B,B,B)
        @tensor B[a] := M[1][a,b,c,d] * B[d] * B[c] * B[b]
        Z_n = dot(B,B)
        normalize!(B)
        error = norm(Z_n - Z)
        if error < 1e-16
            break
        end
        Z = Z_n
    end
    println("================================")
    @show error, Z
    println("================================")
    gradnew = deepcopy(grad)
    Ni = size(M)[1]
    # reB = (reshape(B, D,D)+I*δ)^(-1)
    reB = reshape(B, D,D)
    for p in 1:length(M)
        i, j = Tuple(findfirst(==(p), M.pattern))
        ir = Ni + 1 - i
        # n = sum(ein"((abcd,d),c),b,a ->"(M[1],B,B,B,B))
        n = @tensor M[1][a,b,c,d] * B[d] * B[c] * B[b] * B[a]
        # P = ein"ae,bf,cg,dh->abcdefgh"(reB,reB,reB,reB)/n
        # λ,_ = eigen(reshape(P,D^4,D^4))
        # @show real(λ[end-10:end])
        # gradnew[p], _ = linsolve(x->δ * x + ein"(((abcdp,ae),bf),cg),dh->efghp"(x,reB,reB,reB,reB)/n, grad[p]; isposdef = true, maxiter=1)
        gradnew[p], _ = linsolve(x->δ * x + (@tensor xout[e,f,g,h,p] := x[a,b,c,d,p] * reB[a,e] * reB[b,f] * reB[c,g] * reB[d,h]
        )/n, grad[p]; isposdef = true, maxiter=1)
        # gradnew[p] = ein"(((abcdp,ae),bf),cg),dh->efghp"(grad[p],reB,reB,reB,reB)*n
    end

    return gradnew
end
