function precondition_invese_single_envir(A, grad, rt, params, restriction_ipeps, fδEi)
    # size(A) == (1,) || throw(Base.error("precondition only supports 1x1 unit cell currently"))
    if fδEi[2] > 0.01 || fδEi[3] <= 20
        return grad
    end
    δ = fδEi[2]
    A = restriction_ipeps(A)
    A = build_A(A, params)
    _, M = build_M(A, params) 
    # rt′ = leading_boundary(rt, M, params.boundary_alg)
    # Zygote.@ignore params.reuse_env && update!(rt, rt′)

    env = VUMPSEnv(rt, M, params.boundary_alg)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env

    χ,D = size(ACu[1])[[1,2]]
    D = Int(sqrt(D))
    re(x) = reshape(x, χ, D, D, χ)

    gradnew = deepcopy(grad)
    Ni = size(M)[1]
    for p in 1:length(M)
        i, j = Tuple(findfirst(==(p), M.pattern))
        ir = Ni + 1 - i
        n = sum(ein"(((abc,adf),dgeb),fgh),ceh->"(ACu[i,j],FLo[i,j],M[i,j],conj(ACd[ir,j]),FRo[i,j]))
        gradnew[p], _ = linsolve(x->δ * x + ein"(((iaej,jbfk),abcdp),lcgk),idhl->efghp"(re(FLo[i,j]),re(conj(ACd[ir,j])),x,re(FRo[i,j]),re(ACu[i,j]))/n, grad[p]; isposdef = true, maxiter=1)
    end

    return gradnew
end

function hessian_vec_prod(f, V, v)
    return ForwardDiff.derivative(t -> Zygote.gradient(f, V + t * v)[1], 0.0)
end

function precondition_invese_hessian(f, A, grad, fδEi)
    size(A) == (1, ) || throw(Base.error("precondition only supports 1x1 unit cell currently"))
    # if fδEi[2] > 0.01 || fδEi[3] <= 10
    #     return grad
    # end

    δ = 1e-8
    @show δ
    gradnew, info = linsolve(v->v*δ + hessian_vec_prod(f, A, v), grad; isposdef = true, maxiter=1)
    @show info
    return gradnew
end

function Base.Float64(x::ForwardDiff.Dual) 
    # @show dump(x)
    return x.value
    # return x
end
