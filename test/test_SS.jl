using TeneT_demo
using Random
using CUDA
using TeneT
using OptimKit
using LinearAlgebra
# using Zygote

seed = 42
Random.seed!(seed)
atype = CuArray
D, χ = 3, 64
pattern = [1 3;
           2 4]
# pattern = [1;;]
model = SS(0.63,1.0)
No = 0
SUτ = 0.0
ifprecondition = false
if ifprecondition
    folder = joinpath(pkgdir(TeneT_demo), "data/$model/$pattern/seed$seed/withprecondition/")
else
    folder = joinpath(pkgdir(TeneT_demo), "data/$model/$pattern/seed$seed/withoutprecondition/")
end
boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false,
                     ifsimple_eig=true,
                     ifparallelupdown=false,
                     ifcheckpoint=false,
                     forloop_iter=1,
                     maxiter=30, 
                     miniter=1, 
                     maxiter_ad=4,
                     miniter_ad=4,
                     power_iter=5,
                     power_iter_obs=40,
                     show_every=10,
                     tol=1e-10,
                     verbosity=3,
)
params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg, 
                     #    optimizer=GradientDescent(),
                          optimizer=LBFGS(200; maxiter=100, verbosity=4, gradtol=1e-7),
                          ifcheckpoint=false,
                          forloop_iter=1,
                          verbosity=4, 
                          folder=folder,
                          ifSU=false,
                          SUτ=SUτ,
                          ifprecondition=ifprecondition,
                          iter_precond=0,
                          reuse_env=true, 
                          ifflatten=false,
                          ifsave_env=true,
                          ifload_env=true,
                          ifsave_lbfgs=true,
                          ifload_lbfgs=true
)
A = init_ipeps(;atype, No, d=2, pattern, D, params)
# A = TeneT_demo.init_ipeps_to_D(;atype, No, D, D_new=3, params)

function _restriction_ipeps(A)
   return A
end

optimise_ipeps(A, χ, params;
               restriction_ipeps = _restriction_ipeps);
