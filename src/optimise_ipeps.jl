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
function energy(A, h, χ, params::iPEPSOptimize; restriction_ipeps)
    A = restriction_ipeps(A)
    A = build_A(A, params)
    M = build_M(A, params)
    D = size(A[1], 1)
    
    rt = Zygote.@ignore initialize_vumps_runtime(A, D, χ, params)
    rt, _ = leading_boundary(rt, M, params.boundary_alg)

    Zygote.@ignore begin
        folder = joinpath(params.folder, "D$(D)", "VUMPS_rt_env")
        !(ispath(folder)) && mkpath(folder)
        params.ifsave_env && save_rt(folder, rt; file="χ$(χ).jld2")
    end

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
    χs = [χ1, χ2]
    function f(A)
        return real(energy(A, h, χs[1], params; restriction_ipeps))
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
                                               finalize! = (x, f, g, iter)->_finalize!(x, f, g, iter, h, D, χs, params, t0, fδEi; restriction_ipeps)
    )
    return x, fδEi
end

_inner(x, dx1, dx2) = real(dot(dx1, dx2))
function _finalize!(x, f, g, iter, h, D, χs, params, t0, fδEi; restriction_ipeps)
    @unpack folder = params

    fδEi[3] = iter
    fδEi[2] = abs(fδEi[1] - f)
    fδEi[1] = f
    if χs[1] < χs[2]
        e2 = real(energy(x, h, χs[2], params; restriction_ipeps))
    else
        e2 = f
    end
    fδEi[5] = e2 - fδEi[4]
    fδEi[4] = e2
    message = @sprintf("i = %5d\tt = %0.2f sec\te_χ%d = %.15f\te_χ%d = %.15f\tgnorm = %.3e\n", iter, time() - t0, χs[1], f, χs[2], e2, norm(g))
    if fδEi[5] > params.tol || fδEi[2] ≈ 0 || abs(f-e2) > params.tol
        f = e2
        χs[1] += 1
        χs[2] += 1
    end

    folder0 = joinpath(folder, "D$(D)")
    !(ispath(folder0)) && mkpath(folder0)
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