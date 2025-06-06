module Defaults
    const VERBOSE_NONE = 0
    const VERBOSE_WARN = 1
    const VERBOSE_CONV = 2
    const VERBOSE_ITER = 3
    const VERBOSE_ALL = 4

    using OptimKit
    using KrylovKit
    const contr_maxiter = 100
    const contr_miniter = 4
    const contr_tol = 1e-8
    const fpgrad_maxiter = 100
    const fpgrad_tol = 1e-6
    const verbosity = VERBOSE_ITER
    const reuse_env = true
    const rrule_alg = GMRES(; tol=1e1contr_tol)
    const optimizer = LBFGS(; verbosity = 0)
    const folder = "data"
    const show_every = 1
    const save_every = 1
    const ifload_env = true
    const ifsave_env = true
    const SUτ = 0.0
    const ifprecondition = false
    const ifflatten = true
    const forloop_iter = 1
    const iter_precond = 20

    finalize!(x, f, g, numiter) = (x, f, g)
end