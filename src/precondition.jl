function reinv(ρ, δ)
    D = size(ρ, 1)
    ρ = reshape(ein"afbgchdi->abcdfghi"(ρ), D^4, D^4) + I * δ
    ρ = reshape(ρ, D, D, D, D, D, D, D, D)
end

function precondition_invese_single_envir(A, grad, rt, params, restriction_ipeps)
    A = build_A(A, params)
    A = restriction_ipeps(A)
    _, M = build_M(A, params) 
    # rt′ = leading_boundary(rt, M, params.boundary_alg)
    # Zygote.@ignore params.reuse_env && update!(rt, rt′)

    env = VUMPSEnv(rt, M, params.boundary_alg)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env

    n = Array(ein"(((abc,adf),dgeb),fgh),ceh->"(ACu[1],FLo[1],M[1],conj(ACd[1]),FRo[1]))[]
    # @show n n/ein"abc,abc->"(FLo[1], FRo[1])[]
    # nn, _ = rightenv(ARu, conj(ARd), M, FLo; ifobs=true) 
    # @show nn
    χ,D = size(ACu[1])[[1,2]]
    D = Int(sqrt(D))
    ACu = reshape(ACu[1], χ, D, D, χ)
    ACd = reshape(ACd[1], χ, D, D, χ)
    FLo = reshape(FLo[1], χ, D, D, χ)
    FRo = reshape(FRo[1], χ, D, D, χ)
    ρ = ein"((jafk,kbgl),mchl),jdim -> afbgchdi"(FLo,conj(ACd),FRo,ACu)

    # δ = norm(grad) > 1e-1 ? norm(grad)/1e3 : 1e-12
    δ = 1e-8
    gradnew, _ = linsolve(x->ein"abcdexy, abcdfghi->fghiexy"(x, reinv(ρ, δ)), grad*n; isposdef = true, maxiter=1)
    return gradnew
end