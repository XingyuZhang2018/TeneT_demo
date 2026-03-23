using TeneT_demo
using Random
using CUDA
using TeneT
using OptimKit
using LinearAlgebra
using Zygote

seed = 44
Random.seed!(seed)
atype = Array
etype = Float64
D, χ, χshifit = 2, 16, 0
pattern = [1 3;
           2 4]
# pattern = [1;;]
model = J1J2J3(J1=1.6, J2=1.0, J3=0.5, ifrotate=false)
No = 97
SUτ = 0.0
folder = joinpath(pkgdir(TeneT_demo), "data/$model/$pattern/seed$seed/")
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
                          optimizer=LBFGS(200; maxiter=100, verbosity=4, gradtol=1e-7, linesearch=HagerZhangLineSearch(maxfg=5)),
                          ifcheckpoint=false,
                          forloop_iter=1,
                          verbosity=4, 
                          folder=folder,
                          ifSU=false,
                          SUτ=SUτ,
                          ifprecondition=false,
                          iter_precond=0,
                          reuse_env=true, 
                          ifflatten=false,
                          ifsave_env=true,
                          ifload_env=true,
                          ifsave_lbfgs=true,
                          ifload_lbfgs=false,
                          order=:none,
                          bondratio=1.0
)
# A = init_ipeps(;atype, etype, No, d=2, pattern, D, χ, params)
# A = TeneT_demo.init_ipeps_to_D(;atype, No, D, D_new=3, params)
A = TeneT_demo.init_ipeps_perturbation(;atype, No, D, D_new=3, χ, params)

function restriction_ipeps(A)
    # Ar = Zygote.Buffer(A)
    # Ar[:,:,:,:,:,1] = A[:,:,:,:,:,1]
    # Ar[:,:,:,:,:,1] += permutedims(Ar[:,:,:,:,:,1],(4,3,2,1,5))
    # Ar[:,:,:,:,:,1] += permutedims(Ar[:,:,:,:,:,1],(4,2,3,1,5))
    # Ar[:,:,:,:,:,1] += permutedims(Ar[:,:,:,:,:,1],(1,3,2,4,5))

    # Ar[:,:,:,:,:,2] = permutedims(Ar[:,:,:,:,:,1], (4,1,2,3,5))
    # Ar[:,:,:,:,:,3] = permutedims(Ar[:,:,:,:,:,1], (2,3,4,1,5))
    # Ar[:,:,:,:,:,4] = permutedims(Ar[:,:,:,:,:,1], (3,4,1,2,5))

    # Ar[:,:,:,:,:,2] = permutedims(Ar[:,:,:,:,:,1], (1,4,3,2,5))
    # Ar[:,:,:,:,:,3] = permutedims(Ar[:,:,:,:,:,1], (3,2,1,4,5))
    # Ar[:,:,:,:,:,4] = permutedims(Ar[:,:,:,:,:,1], (3,4,1,2,5))

    # Ar = copy(Ar)
    # return Ar/norm(Ar)
    # return A/norm(A)
    A /= norm(A)
    A = TeneT_demo.local_min_norm(A, params)
end

optimise_ipeps(A, 32, χshifit, params; restriction_ipeps);

# observable(A, χ, params; restriction_ipeps);
# println(1)
