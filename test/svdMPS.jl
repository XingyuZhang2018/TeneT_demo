using TeneT_demo
using Random
using CUDA
using TeneT
using OptimKit
using LinearAlgebra
using FileIO

seed = 42
Random.seed!(seed)
atype = CuArray
D, χ = 3, 20
pattern = [1;;]
Ni,Nj = size(pattern)
model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
h = atype(hamiltonian(model))
No = 60
SUτ = 0.0
ifprecondition = true
if ifprecondition
    folder = "data/$model/seed$seed/withprecondition/"
else
    folder = "data/$model/seed$seed/withoutprecondition/"
end
boundary_alg = VUMPS(ifupdown=false,
                     ifdownfromup=false,
                     ifsimple_eig=true,
                     maxiter=100, 
                     miniter=1, 
                     show_every=1,
                     maxiter_ad=0,
                     miniter_ad=0,
                     verbosity=3
)
params = iPEPSOptimize(pattern=pattern,
                       boundary_alg=boundary_alg, 
                    #    optimizer=GradientDescent(),
                       optimizer=LBFGS(; maxiter=20, verbosity=1, gradtol=1e-7),
                       reuse_env=true, 
                       verbosity=4, 
                       folder=folder,
                       SUτ=SUτ,
                       ifprecondition=ifprecondition

)
A = init_ipeps(;atype, No, d=2, pattern, D, χ, params)
# A = TeneT_demo.init_ipeps_from_small_D(;atype, No, d=2, Ni, Nj, D,D_new=3,ϵ=1e-3, χ, params)
# A = reshape([A[:,:,:,:,:,1,1]],1,1)

# @show A[1] == A[1,1] A[2]==A[2,1] A[3]==A[1,2] A[4]==A[2,2]
function _restriction_ipeps(A)
   A += map(A->permutedims(conj(A), (1,4,3,2,5)), A) # up-down
   A += map(A->permutedims(conj(A), (3,2,1,4,5)), A) # left-right
   A += map(A->permutedims(conj(A), (2,1,4,3,5)), A) # diagonal
   A += map(A->permutedims(conj(A), (4,3,2,1,5)), A) # rotation
   
   return A / norm(A)
end

A′ = _restriction_ipeps(A)

D = size(A[1], 1)
oc = TeneT_demo.optcont(D, χ)
A′ = TeneT_demo.build_A(A′, params)
ap, M = TeneT_demo.build_M(A′, params)
################## loaded ################## 

for χ in 20:10:80
    rt = VUMPSRuntime(M, χ, params.boundary_alg)
    rt′ = leading_boundary(rt, M, params.boundary_alg)
    save_rt(joinpath(folder, "D$(D)_χ$(χ)"), rt′)
end
# rt′ = load_rt(joinpath(folder, "D$(D)_χ$(χ)"), CuArray);
# @show typeof(rt′.AL)

# env = VUMPSEnv(rt′, M, params.boundary_alg)
# TeneT_demo.expectation_value(h, ap, env, oc, params)
# function trunc_env(env, χ, χ′)
#     function trunc_MPS(A, χ′)
#         U, S, V = svd(reshape(A, χ*D, χ*D))
#         A′ = reshape(U[:, 1:χ′] * Diagonal(S[1:χ′]) * V[:, 1:χ′]', χ,D^2,χ)
#         # @show S
#         return A′
#     end
#     data = []
#     for field in fieldnames(typeof(env))
#         A = getfield(env, field)
#         push!(data, StructArray([trunc_MPS(A[1], χ′)], pattern))
#     end
#     return VUMPSEnv(data...)
# end
energys = []
for χ in 20:10:60
    χ′ = χ
    rt′ = load_rt(joinpath(folder, "D$(D)_χ$(χ)"), CuArray)
    env = VUMPSEnv(rt′, M, params.boundary_alg)
    # env′ = trunc_env(env, χ, χ′)
    env′ = env
    e = TeneT_demo.expectation_value(h, ap, env′, oc, params)
    push!(energys, real(e))
    print("{$(χ′), $(e)},")
end
for (χ,e) in zip(20:10:60,energys)
    print("{$(χ), $(e)},")
end