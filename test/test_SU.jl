using TeneT_demo
using Random
using CUDA
using TeneT
using OptimKit
using LinearAlgebra

seed = 100
Random.seed!(seed)
atype = Array
D, χ = 2, 10
pattern = [1 3; 2 4]
Ni,Nj = size(pattern)
model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
h = atype(hamiltonian(model))
No = 0
SUτ = 0.01
ifprecondition = false
if ifprecondition
    folder = "data/$model/seed$seed/withprecondition/"
else
    folder = "data/$model/seed$seed/withoutprecondition/FU/"
end
boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false,
                     ifsimple_eig=true,
                     maxiter=10, 
                     miniter=1, 
                    #  miniter_ad=10,
                     verbosity=0
)
params = SUOptimize(pattern=pattern,
                    boundary_alg=boundary_alg, 
                    reuse_env=true, 
                    verbosity=0, 
                    folder=folder,
                    maxiter=1000,
                    SUτ=SUτ

)
A = init_ipeps(;atype, No, d=2, pattern, D, χ, params)
# A = TeneT_demo.init_ipeps_from_small_D(;atype, No, d=2, Ni, Nj, D,D_new=3,ϵ=1e-3, χ, params)

# @show A[1] == A[1,1] A[2]==A[2,1] A[3]==A[1,2] A[4]==A[2,2]
function _restriction_ipeps(A)

   return A / norm(A)
end

optimise_ipeps(A, h, χ, params)