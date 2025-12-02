using TeneT_demo
using Random
using CUDA
using TeneT
using OptimKit
using LinearAlgebra
using Zygote

seed = 42
Random.seed!(seed)
atype = Array
D, χ, χshifit = 2, 20, 1
pattern = [1 3;
           2 4]
# pattern = [1;;]
model = J1J2(J2=0.0, ifrotate=true)
No = 0
SUτ = 0.0
order = :mixed
bondratio = [1.0,1.0]
folder = joinpath(pkgdir(TeneT_demo), "data/$model/$pattern/$order/seed$seed/")
boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false,
                     ifsimple_eig=true,
                     ifparallelupdown=false,
                     ifcheckpoint=false,
                     forloop_iter=1,
                     maxiter=300, 
                     miniter=1, 
                     maxiter_ad=3,
                     miniter_ad=3,
                     power_iter=5,
                     power_iter_obs=20,
                     show_every=10,
                     tol=1e-10,
                     verbosity=3,
)
params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg, 
                     #    optimizer=GradientDescent(),
                          optimizer=LBFGS(200; maxiter=0, verbosity=4, gradtol=1e-7, linesearch=HagerZhangLineSearch(maxfg=5)),
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
                          ifload_env=false,
                          ifsave_lbfgs=true,
                          ifload_lbfgs=false,
                          order=order,
                          bondratio=bondratio
)
A = init_ipeps(;atype, No, d=2, pattern, D, χ, params)
# A = TeneT_demo.init_ipeps_to_D(;atype, No, D, D_new=3, params)

function _restriction_ipeps(A)
    # Ar = Zygote.Buffer(A)
    # Ar[:,:,:,:,:,1] = A[:,:,:,:,:,1]
    # Ar[:,:,:,:,:,1] += permutedims(Ar[:,:,:,:,:,1],(4,3,2,1,5))
    # Ar[:,:,:,:,:,1] += permutedims(Ar[:,:,:,:,:,1],(4,2,3,1,5))
    # Ar[:,:,:,:,:,1] += permutedims(Ar[:,:,:,:,:,1],(1,3,2,4,5))

    # Ar[:,:,:,:,:,2] = permutedims(Ar[:,:,:,:,:,1], (4,1,2,3,5))
    # Ar[:,:,:,:,:,3] = permutedims(Ar[:,:,:,:,:,1], (2,3,4,1,5))
    # Ar[:,:,:,:,:,4] = permutedims(Ar[:,:,:,:,:,1], (3,4,1,2,5))

    # Ar[:,:,:,:,:,2] = permutedims(conj(Ar[:,:,:,:,:,1]), (1,4,3,2,5))
    # Ar[:,:,:,:,:,3] = permutedims(Ar[:,:,:,:,:,1], (3,2,1,4,5))
    # Ar[:,:,:,:,:,4] = permutedims(conj(Ar[:,:,:,:,:,1]), (3,4,1,2,5))

    # Ar = copy(Ar)
    # return Ar
    return A
end

optimise_ipeps(A, χ, χshifit, params;
               restriction_ipeps = _restriction_ipeps
);

# observable(A, 20, params;
#            restriction_ipeps = _restriction_ipeps
# );
# println(1)
