using TeneT_demo
using Random
using CUDA
using TeneT
using Optim
using LinearAlgebra

seed = 47
Random.seed!(seed)
atype = Array
D, χ = 2, 20
Ni, Nj = 2, 2
model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
h = atype(hamiltonian(model))
No = 0
SUτ = -0.1
folder = "data/$model/seed$seed/AD+SU$SUτ/"

boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false,
                     ifsimple_eig=true,
                     maxiter=10, 
                     miniter=3, 
                     verbosity=2
)
params = iPEPSOptimize(boundary_alg=boundary_alg, 
                    #    optimizer=GradientDescent(),
                       optimizer=LBFGS(m=20),
                       reuse_env=true, 
                       verbosity=4, 
                       maxiter=1000,
                       tol=1e-10,
                       folder=folder,
                       SUτ=SUτ,
                       ifprecondition=false,
)
A = init_ipeps(;atype, No, d=2, Ni, Nj, D, χ, params)

function _restriction_ipeps(A)
   # A += permutedims(conj(A), (1,4,3,2,5)) # up-down
   # A += permutedims(conj(A), (3,2,1,4,5)) # left-right
   # A += permutedims(conj(A), (2,1,4,3,5)) # diagonal
   # A += permutedims(conj(A), (4,3,2,1,5)) # rotation
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
               restriction_ipeps = _restriction_ipeps)