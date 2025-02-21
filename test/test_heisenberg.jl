using TeneT_demo
using Random
using CUDA
using TeneT
using OptimKit
using LinearAlgebra

seed = 47
Random.seed!(seed)
atype = CuArray
D, χ = 4, 20
Ni, Nj = 1, 1
model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
h = atype(hamiltonian(model))
No = 20
SUτ = 0.0
folder = "data/$model/seed$seed/withprecondition/"
pattern = [1;;]
boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false,
                     ifsimple_eig=true,
                     maxiter=10, 
                     miniter=3, 
                     verbosity=3
)
params = iPEPSOptimize(pattern=pattern,
                       boundary_alg=boundary_alg, 
                    #    optimizer=GradientDescent(),
                       optimizer=LBFGS(; maxiter=1000, verbosity=0, gradtol=1e-8),
                       reuse_env=true, 
                       verbosity=4, 
                       folder=folder,
                       SUτ=SUτ,
                       ifprecondition=true,

)
A = init_ipeps(;atype, No, d=2, Ni, Nj, D, χ, params)

function _restriction_ipeps(A)
   A += map(A->permutedims(conj(A), (1,4,3,2,5)), A) # up-down
   A += map(A->permutedims(conj(A), (3,2,1,4,5)), A) # left-right
   A += map(A->permutedims(conj(A), (2,1,4,3,5)), A) # diagonal
   A += map(A->permutedims(conj(A), (4,3,2,1,5)), A) # rotation

   # Ar = Zygote.Buffer(A)
   # Ni, Nj = size(A)
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
   return A / norm(A)
end

optimise_ipeps(A, h, χ, params;
               restriction_ipeps = _restriction_ipeps);