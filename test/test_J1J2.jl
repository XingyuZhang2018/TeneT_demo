using TeneT_demo
using Random
using CUDA
using TeneT
using OptimKit
using LinearAlgebra
# using Zygote

seed = 42
Random.seed!(seed)
atype = Array
D, χ = 2, 10
# pattern = [1 2;
#            2 1]
pattern = [1;;]
model = J1J2(1.0,0.5,true)
No = 0
SUτ = 0.0
ifprecondition = true
if ifprecondition
    folder = joinpath(pkgdir(TeneT_demo), "data/$model/$pattern/seed$seed/withprecondition/")
else
    folder = joinpath(pkgdir(TeneT_demo), "data/$model/$pattern/seed$seed/withoutprecondition/")
end
boundary_alg = VUMPS(ifupdown=false,
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
                          ifload_env=false,
                          ifsave_lbfgs=true,
                          ifload_lbfgs=false
)
A = init_ipeps(;atype, No, d=2, pattern, D, params)
# A = TeneT_demo.init_ipeps_to_D(;atype, No, D, D_new=3, params)

function _restriction_ipeps(A)
   A += permutedims(conj(A), (1,4,3,2,5,6)) # up-down
   A += permutedims(conj(A), (3,2,1,4,5,6)) # left-right
   A += permutedims(conj(A), (2,1,4,3,5,6)) # diagonal
   A += permutedims(conj(A), (4,3,2,1,5,6)) # rotation

   # Ar = Zygote.Buffer(A)
   # for i in 1:length(A)
   # for j in 1:Nj, i in 1:Ni
   #     if (i,j) in [(2,1)]
   #         Ar[i,j] = A[i,j] + permutedims(conj(A[i,j]), (1,4,3,2,5))
   #     elseif (i,j) in [(3,1)]
   #         Ar[i,j] = permutedims(conj(A[1,1]), (1,4,3,2,5))
   #     else
   #         Ar[i,j] = A[i,j]
   #     end
   # end
   # Ar = copy(Ar)
   # return Ar/norm(Ar)
   # λ = Zygote.@ignore norm(A)
   return A
end

optimise_ipeps(A, χ, params;
               restriction_ipeps = _restriction_ipeps);
