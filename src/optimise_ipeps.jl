@kwdef mutable struct GradientOptimize <: iPEPSOptimize
    model::HamiltonianModel
    pattern::Matrix{Int}
    boundary_alg::VUMPS
    reuse_env::Bool = true
    verbosity::Int = VERBOSE_ITER
    maxiter::Int = 100
    SUτ::Real = 0.0
    ifSU::Bool = false
    optimizer = LBFGS(; verbosity = 0)
    folder::String = joinpath(pwd(), "data", "ipeps")
    show_every::Int = 1
    save_every::Int = 1
    ifsave_env::Bool = true
    save_env_tol::Real = 1e-4
    ifload_env::Bool = true
    ifflatten::Bool = false
    forloop_iter::Int = 1
    ifcheckpoint::Bool = false
    ifprecondition::Bool = false
    iter_precond::Int = 20

    ifsave_lbfgs::Bool = true
    ifload_lbfgs::Bool = true

    order::Symbol = :none
    bondratio = 1.0
end

"""
    energy(h, bcipeps; χ, tol, maxiter)
return the energy of the `bcipeps` 2-site hamiltonian `h` and calculated via a
BCVUMPS with parameters `χ`, `tol` and `maxiter`.
"""
function energy(A, rt, rt′, fδEiEI, params::iPEPSOptimize)
    A = build_A(A, params)
    M = build_M(A, params)
    # rt, _ = params.ifcheckpoint ? checkpoint(leading_boundary, rt, M, params.boundary_alg) : leading_boundary(rt, M, params.boundary_alg)
    rt, _ = leading_boundary(rt, M, params.boundary_alg)
    Zygote.@ignore update!(rt′, rt)
    env = VUMPSEnv(rt, M, params.boundary_alg)
    return expectation_value(params.model, A, env, fδEiEI, params)[1]
end


"""
    optimise_ipeps(A::AbstractArray, key; f_tol = 1e-6, opiter = 100, optimmethod = LBFGS(m = 20))

return the tensor `A'` that describes an ipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimise_ipeps(A, χ::Int, χshift::Int, params::iPEPSOptimize;
                        restriction_ipeps = _restriction_ipeps)
    D = size(A, 1)
    rt = initialize_vumps_runtime(A, D, χ, params; restriction_ipeps)
    rt′ = deepcopy(rt)
    fδEiEI = [1.0,1.0,0,0]

    params_obs = deepcopy(params)
    params_obs.boundary_alg.maxiter = params.boundary_alg.maxiter * 10
    function fenergy(A)
        A = restriction_ipeps(A)
        return real(energy(A, rt, rt′, fδEiEI, params))
    end
    function fg(x)
        t1 = time()
        e, vjp = pullback(fenergy, x)
        params.verbosity >= 2 && printstyled(" forward calculation took $(round(time() - t1, digits = 2)) s\n"; bold=true, color=:green) 
        TeneT.reclaim(x)
        t2 = time()
        g = vjp(1)[1]
        params.verbosity >= 2 && printstyled("backward calculation took $(round(time() - t2, digits = 2)) s\n"; bold=true, color=:green)
        TeneT.reclaim(g)
        return e, g
    end
    alg = params.optimizer
    t0 = time()
    
    _precondition(x, g) = params.ifprecondition ? precondition_invese_single_envir(x, g, rt, params, restriction_ipeps, fδEiEI, params.iter_precond) : g
    
    state_path = joinpath(params.folder, "D$(D)", "lbfgs_checkpoint")
    # _precondition(x, g) = precondition_invese_hessian(x, g, rt, rt′, params, restriction_ipeps, fδEiEI , params.iter_precond)
    # local x, f, g, numfg, normgradhistory
    for _ in 1:100
        A, e, _ = optimize_reload(fg, A, alg; 
                                  resume_from = params.ifload_lbfgs ? joinpath(state_path, "χ$χ.jld2") : nothing,
                                  save_state_to = params.ifsave_lbfgs ? joinpath(state_path, "χ$χ.jld2") : nothing,
                                  save_every=params.save_every,
                                  precondition=_precondition, 
                                  inner = _inner, 
                                  finalize! = (x, f, g, iter)->_finalize!(x, f, g, iter, rt, rt′, D, χ, params, t0, fδEiEI)
        )
        χ += χshift 
        enew, = observable(A, χ, fδEiEI, params_obs; restriction_ipeps)
        rt = initialize_vumps_runtime(A, D, χ, params; restriction_ipeps)
        rt′ = deepcopy(rt)
        if abs(real(enew[1]) - e) < 1e-7
            break
        end
    end
end

_inner(x, dx1, dx2) = real(dot(dx1, dx2))
function _finalize!(x, f, g, iter, rt, rt′, D, χ, params, t0, fδEiEI )
    @unpack folder = params

    fδEiEI[3] = iter
    fδEiEI[2] = abs(fδEiEI[1] - f)
    fδEiEI[1] = f
    message = @sprintf("i = %5d\tt = %0.2f sec\te_χ%d = %.15f\tgnorm = %.3e\n", iter, time() - t0, χ, f, norm(g))

    folder0 = joinpath(folder, "D$(D)")
    !(ispath(folder0)) && mkpath(folder0)
    folder1 = joinpath(folder, "D$(D)", "VUMPS_rt_env")
    params.reuse_env && update!(rt, rt′)
    params.ifsave_env && save_rt(folder1, rt; file="χ$(χ).jld2")
    if params.verbosity >= 3 && iter % params.show_every == 0
        printstyled(message; bold=true, color=:red)
        flush(stdout)

        logfile = open(joinpath(folder0, "history.log"), "a")
        write(logfile, message)
        close(logfile)
    end
    if params.save_every != 0 && iter % params.save_every == 0
        save(joinpath(folder0,  "ipeps", "χ$χ", "No.$(iter).jld2"), "bcipeps", Array(x))
    end
    
    if abs(fδEiEI[2]) < 1e-12 || abs(fδEiEI[4]) > 1e-7
        g .= 0
    end
    return x, f, g
end 

function Z(M,rt,alg)
    @unpack AL, AR, C, FL, FR = rt
    AC = TeneT.ALCtoAC(AL, C)
    λAC, = TeneT.ACenv(AC, FL, M, FR; ifvalue=true, alg)
    λC,  = TeneT.Cenv( C, FL, FR; ifvalue=true, alg)
    return real(λAC[1]/λC[1])
end