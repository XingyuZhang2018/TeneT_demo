@kwdef mutable struct iPEPSOptimize
    boundary_alg::VUMPS
    reuse_env::Bool = Defaults.reuse_env
    verbosity::Int = Defaults.verbosity
    maxiter::Int = Defaults.fpgrad_maxiter
    tol::Real = Defaults.fpgrad_tol
    optimizer = Defaults.optimizer
    folder::String = Defaults.folder
    show_every::Int = Defaults.show_every
    save_every::Int = Defaults.save_every
end

"""
    indexperm_symmetrize(ipeps)
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
function indexperm_symmetrize(ipeps)
    ipeps += permutedims(ipeps, (1,4,3,2,5)) # up-down
    ipeps += permutedims(ipeps, (3,2,1,4,5)) # left-right
    ipeps += permutedims(ipeps, (2,1,4,3,5)) # diagonal
    ipeps += permutedims(ipeps, (4,3,2,1,5)) # rotation
    return ipeps / norm(ipeps)
end

"""
    init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)
Initial `bcipeps` and give `key` for use of later optimization. The key include `model`, `D`, `χ`, `tol` and `maxiter`. 
The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(;atype = Array, Ni::Int, Nj::Int, D::Int)
    A = atype(randn(D,D,D,D,2,Ni,Nj))
    A /= norm(A)
    return A
end

"""
    energy(h, bcipeps; χ, tol, maxiter)
return the energy of the `bcipeps` 2-site hamiltonian `h` and calculated via a
BCVUMPS with parameters `χ`, `tol` and `maxiter`.
"""
function energy(A, h, rt, oc, params::iPEPSOptimize)
    A = indexperm_symmetrize.(A)
    D = size(A[1], 1)
    ap = [reshape(ein"abcde,fghmn->afbgchdmen"(A, conj(A)), D^2,D^2,D^2,D^2, 2,2) for A in A]
    M  = [ein"abcdee->abcd"(ap) for ap in ap]
    rt′ = leading_boundary(rt, M, params.boundary_alg)
    Zygote.@ignore params.reuse_env && update!(rt, rt′)
    env = VUMPSEnv(rt′, M)
    return expectation_value(h, ap, env, oc, params)
end

"""
    optimise_ipeps(A::AbstractArray, key; f_tol = 1e-6, opiter = 100, optimmethod = LBFGS(m = 20))

return the tensor `A'` that describes an ipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimise_ipeps(A::AbstractArray, h, χ::Int, params::iPEPSOptimize)
    D = size(A, 1)
    oc = optcont(D, χ)

    Ni, Nj = size(A)[end-1:end]
    A′ = [A[:,:,:,:,:,i,j] for i = 1:Ni, j = 1:Nj]
    A′ = indexperm_symmetrize.(A′)
    M = [reshape(ein"abcde,fghme->afbgchdm"(A′, conj(A′)), D^2,D^2,D^2,D^2) for A′ in A′]
    rt = VUMPSRuntime(M, χ, params.boundary_alg)

    function f(A) 
        A′ = [A[:,:,:,:,:,i,j] for i = 1:Ni, j = 1:Nj]
        return real(energy(A′, h, rt, oc, params))
    end
    function g(A)
        # f(x)
        grad = Zygote.gradient(f,A)[1]
        return grad
    end
    res = optimize(f, g, 
        A, params.optimizer, inplace = false,
        Optim.Options(f_tol=params.tol, iterations=params.maxiter,
        extended_trace=true,
        callback=os->writelog(os, params, D, χ)
        ))
    return res
end

"""
    writelog(os::OptimizationState, key=nothing)

return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_chi_tol_maxiter.log`. Save the final `ipeps` in file `/data/model_D_chi_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, params::iPEPSOptimize, D::Int, χ::Int)
    @unpack folder = params

    message = @sprintf("i = %5d\tt = %0.2f sec\tenergy = %.15f \tgnorm = %.3e\n", os.iteration, os.metadata["time"], os.value, os.g_norm)

    maxiter = params.boundary_alg.maxiter
    folder = joinpath(folder, "D$(D)_χ$(χ)_maxiter$(maxiter)")
    !(ispath(folder)) && mkpath(folder)
    if params.verbosity >= 3 && os.iteration % params.show_every == 0
        printstyled(message; bold=true, color=:red)
        flush(stdout)

        logfile = open(joinpath(folder, "history.log"), "a")
        write(logfile, message)
        close(logfile)
    end
    if params.save_every != 0 && os.iteration % params.save_every == 0
        
        save(joinpath(folder, "ipeps_No.$(os.iteration).jld2"), "bcipeps", os.metadata["x"])
    end

    return false
end