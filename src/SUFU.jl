@kwdef mutable struct SUOptimize <: iPEPSOptimize
    pattern::Matrix{Int}
    boundary_alg::VUMPS
    reuse_env::Bool = Defaults.reuse_env
    verbosity::Int = Defaults.verbosity
    maxiter::Int = Defaults.fpgrad_maxiter
    tol::Real = Defaults.fpgrad_tol
    SUτ::Real = Defaults.SUτ
    folder::String = Defaults.folder
    show_every::Int = Defaults.show_every
    save_every::Int = Defaults.save_every
end

@kwdef mutable struct FUOptimize <: iPEPSOptimize
    pattern::Matrix{Int}
    boundary_alg::VUMPS
    reuse_env::Bool = Defaults.reuse_env
    verbosity::Int = Defaults.verbosity
    maxiter::Int = Defaults.fpgrad_maxiter
    tol::Real = Defaults.fpgrad_tol
    SUτ::Real = Defaults.SUτ
    folder::String = Defaults.folder
    show_every::Int = Defaults.show_every
    save_every::Int = Defaults.save_every
end

function optimise_ipeps(A, h, χ::Int, params::SUOptimize)
    D = size(A[1], 1)
    oc = optcont(D, χ)
    A = build_A(A, params)
    for i in 1:params.maxiter
        A = hv_SU_update(A, params)
        A /= norm(A)
        _, M = build_M(A, params)
        rt = VUMPSRuntime(M, χ, params.boundary_alg)
        e = real(energy(A, h, rt, oc, params))
        println("SU@$i: energy: $e")
    end
end