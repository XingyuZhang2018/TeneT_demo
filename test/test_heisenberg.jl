using TeneT_demo
using Random
using CUDA
using TeneT
using Optim
using LinearAlgebra

Random.seed!(42)
atype = Array
D, χ = 2, 20
Ni, Nj = 1, 1
model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
h = atype(hamiltonian(model))
No = 23
folder = "data/$model/"

boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false,
                     ifsimple_eig=false,
                     maxiter=10, 
                     miniter=1, 
                     verbosity=2
)
params = iPEPSOptimize(boundary_alg=boundary_alg, 
                    #    optimizer=GradientDescent(),
                       optimizer=LBFGS(m=20),
                       reuse_env=true, 
                       verbosity=4, 
                       maxiter=100,
                       tol=1e-6,
                       folder=folder,
                       ifprecondition=false
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