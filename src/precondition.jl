function precondition_invese_single_envir(A, grad, rt, params, restriction_ipeps, fδEi)
    size(A) == (1, 1) || throw(Base.error("precondition only supports 1x1 unit cell currently"))
    if fδEi[2] > 0.01 || fδEi[3] <= 20
        return grad
    end
    A = build_A(A, params)
    A = restriction_ipeps(A)
    _, M = build_M(A, params) 
    # rt′ = leading_boundary(rt, M, params.boundary_alg)
    # Zygote.@ignore params.reuse_env && update!(rt, rt′)

    env = VUMPSEnv(rt, M, params.boundary_alg)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env

    n = sum(ein"(((abc,adf),dgeb),fgh),ceh->"(ACu[1],FLo[1],M[1],conj(ACd[1]),FRo[1]))
    # @show n n/ein"abc,abc->"(FLo[1], FRo[1])[]
    # nn, _ = rightenv(ARu, conj(ARd), M, FLo; ifobs=true) 
    # @show nn
    χ,D = size(ACu[1])[[1,2]]
    D = Int(sqrt(D))
    ACu = reshape(ACu[1], χ, D, D, χ)
    ACd = reshape(ACd[1], χ, D, D, χ)
    FLo = reshape(FLo[1], χ, D, D, χ)
    FRo = reshape(FRo[1], χ, D, D, χ)
    # ρ = ein"((jafk,kbgl),mchl),jdim -> abcdfghi"(FLo,conj(ACd),FRo,ACu)/n
    # D = size(ρ, 1)
    # ρo = reshape(ρ, D^4, D^4)
    # eigvals = eigen(Array(ρo)).values
    # ConditionNumber = eigvals[end] / eigvals[1] 
    # @show ConditionNumber
    # if norm(ConditionNumber) > 1e7
    #     return grad
    # end
    # @show eigvals ConditionNumber
    # @show eigvals

    # δ = norm(grad) > 1e-1 ? norm(grad)/1e3 : 1e-12
    δ = fδEi[2]
    # δ = 1e-8
    # @show δ
    gradnew, info = linsolve(x->δ * x + ein"(((iaej,jbfk),abcdp),idhl),lcgk->efghp"(FLo,conj(ACd),x,FRo,ACu)/n, grad[1]; isposdef = true, maxiter=1)
    # @show info
    # gradnew = ein"abcdexy, abcdfghi->fghiexy"(grad, reshape(pinv(reshape(ρ, D^4, D^4) + I * δ), D, D, D, D, D, D, D, D))
    return StructArray([gradnew], grad.pattern)
end