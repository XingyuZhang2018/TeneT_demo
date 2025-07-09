using TeneT_demo
using Random
using CUDA
using TeneT
using OptimKit
using LinearAlgebra
using Zygote

seed = 100
Random.seed!(seed)
atype = CuArray
D = 4
pattern = [1;;]
Ni,Nj = size(pattern)
model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
h = atype.(hamiltonian(model))
No = 2
SUτ = 0.0
ifprecondition = false
if ifprecondition
    folder = "data/$model/$pattern/seed$seed/withprecondition/"
else
    folder = "data/$model/$pattern/seed$seed/withoutprecondition/"
end
boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false,
                     ifsimple_eig=true,
                     maxiter=30, 
                     miniter=0, 
                     maxiter_ad=10,
                     miniter_ad=3,
                     verbosity=3,
                     power_iter=5,
                     power_iter_obs=20,
                     show_every=100,
                     ifcheckpoint=true,
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
                       ifsave_env=true,
                       iter_precond=1,
                       tol=1e-6
)
A = init_ipeps(;atype, No, d=2, pattern, D, params)
# A = TeneT_demo.init_ipeps_from_small_D(;atype, No, d=2, Ni, Nj, D,D_new=3,ϵ=1e-3, χ, params)

function _restriction_ipeps(A)
#    A += permutedims(conj(A), (1,4,3,2,5,6)) # up-down
#    A += permutedims(conj(A), (3,2,1,4,5,6)) # left-right
#    A += permutedims(conj(A), (2,1,4,3,5,6)) # diagonal
#    A += permutedims(conj(A), (4,3,2,1,5,6)) # rotation

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
   return A/norm(A)
end

fδEi = [1.0,1.0,0,1.0,1.0]
χ1 = 51
while χ1 <= 100
    χ2 = χ1 + 1 
    A, fδEi = @time optimise_ipeps(A, h, χ1, χ2, params;
                                    restriction_ipeps = _restriction_ipeps);
    if abs(fδEi[1] - fδEi[4]) > 1e-3
        χ1 *= 2
    else
        χ1 += 1
    end                
end