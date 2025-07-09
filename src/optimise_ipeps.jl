@kwdef mutable struct GradientOptimize <: iPEPSOptimize
    pattern::Matrix{Int}
    boundary_alg::VUMPS
    reuse_env::Bool = Defaults.reuse_env
    verbosity::Int = Defaults.verbosity
    maxiter::Int = Defaults.fpgrad_maxiter
    tol::Real = Defaults.fpgrad_tol
    SUτ::Real = Defaults.SUτ
    optimizer = Defaults.optimizer
    folder::String = Defaults.folder
    show_every::Int = Defaults.show_every
    save_every::Int = Defaults.save_every
    ifsave_env::Bool = Defaults.ifsave_env
    ifload_env::Bool = Defaults.ifload_env
    ifprecondition::Bool = Defaults.ifprecondition
    ifflatten::Bool = Defaults.ifflatten
    forloop_iter::Int = Defaults.forloop_iter
    iter_precond::Int = Defaults.iter_precond
end

"""
    restriction_ipeps(ipeps)
return a `SquareIPEPS` based on `ipeps` that is symmetric under
permutation of its virtual indices.
```
        4
        │
 1 ── ipeps ── 3
        │
        2
```
"""
function _restriction_ipeps(A)
    return A / norm(A)
end

"""
    energy(h, bcipeps; χ, tol, maxiter)
return the energy of the `bcipeps` 2-site hamiltonian `h` and calculated via a
BCVUMPS with parameters `χ`, `tol` and `maxiter`.
"""
function energy(A, h, rt, rt′, params::iPEPSOptimize)
    M = build_M(A, params)
    # n = 1
    # Zygote.@ignore begin
    #     rt′ = leading_boundary(rt, M, params.boundary_alg)
    #     Zygote.@ignore params.reuse_env && update!(rt, rt′)
    #     env = VUMPSEnv(rt′, M)
    #     @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    #     # n, _ = rightenv(ARu, conj(ARd), M, FLo; ifobs=true) 
    #     λFLo, _ =  rightenv(ARu, conj.(ARd), M; ifobs=true)
    #     λC, _ = rightCenv(ARu, conj.(ARd);    ifobs=true)
    #     n = prod(λFLo./λC)
    # end
    # A /= sqrt(n[1])
    # ap = [reshape(ein"abcde,fghmn->afbgchdmen"(A, conj(A)), D^2,D^2,D^2,D^2, 2,2) for A in A]
    # M  = [ein"abcdee->abcd"(ap) for ap in ap]
    rt, _ = leading_boundary(rt, M, params.boundary_alg)
    Zygote.@ignore update!(rt′, rt)
    env = VUMPSEnv(rt, M, params.boundary_alg)
    return expectation_value(h, A, env, params)
end


"""
    optimise_ipeps(A::AbstractArray, key; f_tol = 1e-6, opiter = 100, optimmethod = LBFGS(m = 20))

return the tensor `A'` that describes an ipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimise_ipeps(A, h, χ1::Int, χ2::Int, params::iPEPSOptimize;
                        restriction_ipeps = _restriction_ipeps)
    D = size(A, 1)
    rt1 = initialize_vumps_runtime(A, D, χ1, params; restriction_ipeps)
    rt1′ = deepcopy(rt1)
    rt2 = initialize_vumps_runtime(A, D, χ2, params; restriction_ipeps)
    function f(A)
        A = restriction_ipeps(A)
        A = build_A(A, params)
        return real(energy(A, h, rt1, rt1′, params))
    end
    function fg(x)
        e, vjp = pullback(f, x)
        g = vjp(1)[1]
        if CUDA.available_memory() / CUDA.total_memory() < 0.1
            GC.gc(true)
            CUDA.reclaim()
        end
        return e, g
    end
    alg = params.optimizer
    t0 = time()
    fδEi = [1.0,1.0,0,1.0,1.0]
    # _precondition(x, g) = params.ifprecondition ? precondition_invese_single_envir(x, g, rt, params, restriction_ipeps, fδEi) : g
    _precondition(x, g) = params.ifprecondition ? precondition_invese_single_envir(x, g, rt1, params, restriction_ipeps, fδEi, params.iter_precond) : g
    # _precondition(x, g) = precondition_invese_hessian(x, g, rt, rt′, params, restriction_ipeps, fδEi, params.iter_precond)
    x, f, g, numfg, normgradhistory = optimize(fg, A, alg; 
                                               precondition=_precondition, 
                                               inner = _inner, 
                                               finalize! = (x, f, g, iter)->_finalize!(x, f, g, iter, rt1, rt1′, rt2, h, D, χ1, χ2, params, t0, fδEi; restriction_ipeps)
    )
    return x, fδEi
end

_inner(x, dx1, dx2) = real(dot(dx1, dx2))
function _finalize!(x, f, g, iter, rt1, rt1′, rt2, h, D, χ1, χ2, params, t0, fδEi; restriction_ipeps)
    params.reuse_env && update!(rt1, rt1′)
    @unpack folder = params

    fδEi[3] = iter
    fδEi[2] = abs(fδEi[1] - f)
    fδEi[1] = f
    if χ1 < χ2
        x′ = restriction_ipeps(x)
        x′ = build_A(x′, params)
        e2 = real(energy(x′, h, rt2, rt2, params))
    else
        e2 = f
    end
    fδEi[5] = e2 - fδEi[4]
    fδEi[4] = e2
    message = @sprintf("i = %5d\tt = %0.2f sec\te_χ%d = %.15f\te_χ%d = %.15f\tgnorm = %.3e\n", iter, time() - t0, χ1, f, χ2, e2, norm(g))
    if fδEi[5] > params.tol || fδEi[2] ≈ 0 || abs(f-e2) > params.tol
        g = zero(g)
    end

    folder0 = joinpath(folder, "D$(D)")
    !(ispath(folder0)) && mkpath(folder0)
    folder1 = joinpath(folder, "D$(D)", "VUMPS_rt_env")
    !(ispath(folder1)) && mkpath(folder1)
    params.ifsave_env && save_rt(folder1, rt1; file="χ$(χ1).jld2")
    params.ifsave_env && save_rt(folder1, rt2; file="χ$(χ2).jld2")
    if params.verbosity >= 3 && iter % params.show_every == 0
        printstyled(message; bold=true, color=:red)
        flush(stdout)

        logfile = open(joinpath(folder0, "history.log"), "a")
        write(logfile, message)
        close(logfile)
    end
    if params.save_every != 0 && iter % params.save_every == 0
        save(joinpath(folder0, "ipeps", "ipeps_No.$(iter).jld2"), "bcipeps", Array(x))
    end
    
    return x, f, g
end 