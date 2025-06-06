using TeneT_demo
using Random
using CUDA
using TeneT
using OptimKit
using LinearAlgebra
using Zygote

seed = 42
Random.seed!(seed)
atype = CuArray
D, χ = 4, 20
pattern = [1;;]
Ni,Nj = size(pattern)
model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
h = atype.(hamiltonian(model))
No = 69
SUτ = 0.0
ifprecondition = true
if ifprecondition
    folder = "data/$model/$pattern/seed$seed/withprecondition/"
else
    folder = "data/$model/$pattern/seed$seed/withoutprecondition/"
end
boundary_alg = VUMPS(ifupdown=false,
                     ifdownfromup=false,
                     ifsimple_eig=true,
                     maxiter=10, 
                     miniter=0, 
                     maxiter_ad=30,
                     miniter_ad=10,
                     verbosity=3,
                     power_iter=5,
                     power_iter_obs=20,
                     show_every=100,
                     tol=1e-10
)
params = GradientOptimize(pattern=pattern,
                       boundary_alg=boundary_alg, 
                    #    optimizer=GradientDescent(),
                       optimizer=LBFGS(200; maxiter=100, verbosity=1, gradtol=1e-7),
                       reuse_env=true, 
                       verbosity=4, 
                       folder=folder,
                       SUτ=SUτ,
                       ifprecondition=ifprecondition,
                       ifflatten=true,
                       iter_precond=20
)
A = init_ipeps(;atype, No, d=2, pattern, D, χ, params)
# A = TeneT_demo.init_ipeps_from_small_D(;atype, No, d=2, Ni, Nj, D,D_new=3,ϵ=1e-3, χ, params)

# @show A[1] == A[1,1] A[2]==A[2,1] A[3]==A[1,2] A[4]==A[2,2]
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

optimise_ipeps(A, h, χ, params;
               restriction_ipeps = _restriction_ipeps);