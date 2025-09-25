using TeneT_demo
using Random
using CUDA
using TeneT
using OptimKit
using LinearAlgebra
# using Zygote

seed = 100
Random.seed!(seed)
atype = CuArray
D, χ = 3, 50
pattern = [1;;]
model = Heisenberg(-1.0,-1.0,1.0)
No = 127
SUτ = 0.0
ifprecondition = true
if ifprecondition
    folder = joinpath(pkgdir(TeneT_demo), "../data/$model/$pattern/seed$seed/withprecondition/test3/")
else
    folder = joinpath(pkgdir(TeneT_demo), "../data/$model/$pattern/seed$seed/withoutprecondition/")
end
boundary_alg = VUMPS(ifupdown=false,
                     ifdownfromup=false,
                     ifsimple_eig=true,
                     ifparallel=false,
                     ifcheckpoint=false,
                     maxiter=100, 
                     miniter=1, 
                     maxiter_ad=10,
                     miniter_ad=3,
                     power_iter=5,
                     power_iter_obs=40,
                     show_every=10,
                     tol=1e-10,
                     verbosity=3
)
params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg, 
                     #    optimizer=GradientDescent(),
                          optimizer=LBFGS(10; maxiter=1000, verbosity=1, gradtol=1e-7),
                          verbosity=4, 
                          folder=folder,
                          ifSU=false,
                          SUτ=SUτ,
                          ifprecondition=ifprecondition,
                          iter_precond=10,
                          reuse_env=true, 
                          ifflatten=false,
                          ifsave_env=true,
                          ifsave_lbfgs=true,
                          ifload_lbfgs=false
)
A = init_ipeps(;atype, No, d=2, pattern, D, params)
# A = TeneT_demo.init_ipeps_to_D(;atype, No, D, D_new=4, params)

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

TeneT_demo.observable(A, χ, params; restriction_ipeps=_restriction_ipeps)
