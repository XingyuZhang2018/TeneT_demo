using TeneT_demo
using Random
using CUDA
using TeneT
using OptimKit
using LinearAlgebra

seed = 42
Random.seed!(seed)
atype = CuArray
D, χ = 3, 50
pattern = [1;;]
Ni,Nj = size(pattern)
model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
h = atype(hamiltonian(model))
No = 0
SUτ = 0.0
ifprecondition = true
if ifprecondition
    folder = "data/$model/seed$seed/withprecondition/"
else
    folder = "data/$model/seed$seed/withoutprecondition/"
end
boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false,
                     ifsimple_eig=true,
                     maxiter=10, 
                     miniter=1, 
                     verbosity=3
)
params = iPEPSOptimize(pattern=pattern,
                       boundary_alg=boundary_alg, 
                    #    optimizer=GradientDescent(),
                       optimizer=LBFGS(; maxiter=1000, verbosity=1, gradtol=1e-7),
                       reuse_env=true, 
                       verbosity=4, 
                       folder=folder,
                       SUτ=SUτ,
                       ifprecondition=ifprecondition

)
A = init_ipeps(;atype, No, d=2, pattern, D, χ, params)
# A = TeneT_demo.init_ipeps_from_small_D(;atype, No, d=2, Ni, Nj, D,D_new=3,ϵ=1e-3, χ, params)

# @show A[1] == A[1,1] A[2]==A[2,1] A[3]==A[1,2] A[4]==A[2,2]
function _restriction_ipeps(A)
   A += StructArray([permutedims(conj(A[1]), (1,4,3,2,5))], A.pattern) # up-down
   A += StructArray([permutedims(conj(A[1]), (3,2,1,4,5))], A.pattern) # left-right
   A += StructArray([permutedims(conj(A[1]), (2,1,4,3,5))], A.pattern) # diagonal
   A += StructArray([permutedims(conj(A[1]), (4,3,2,1,5))], A.pattern) # skew-diagonal

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