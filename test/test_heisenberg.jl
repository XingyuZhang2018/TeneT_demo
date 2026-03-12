using TeneT_demo
using Random
using CUDA
using TeneT
using OptimKit
using LinearAlgebra
using TensorOperations
using ProfileView
# using Zygote

seed = 72
Random.seed!(seed)
atype = Array
etype = ComplexF64
D, χ, χshift = 2, 16, 0
# pattern = [1 2;
#            2 1]
pattern = [1;;]
# pattern = [1 3;
#            2 4]
model = Heisenberg(0.5,-1.0,-1.0,1.0, true)
No = 30
SUτ = 0.0
# ifMCF = false
# if ifMCF
    # folder = joinpath(pkgdir(TeneT_demo), "data/$model/$pattern/seed$seed/MCF/")
# else
    # folder = joinpath(pkgdir(TeneT_demo), "data/$model/$pattern/seed$seed/general/")
# end
folder = joinpath(pkgdir(TeneT_demo), "data/$model/$pattern/seed$seed/BiVUMPS")
boundary_alg = VUMPS(ifupdown=true,
                     ifdownfromup=false,
                     ifsimple_eig=true,
                     ifparallelupdown=false,
                     ifcheckpoint=false,
                     forloop_iter=1,
                     maxiter=30, 
                     miniter=0, 
                     maxiter_ad=4,
                     miniter_ad=4,
                     power_iter=1,
                     power_iter_obs=40,
                     show_every=10,
                     tol=1e-6,
                     verbosity=3,
)
params = GradientOptimize(model=model,
                          pattern=pattern,
                          boundary_alg=boundary_alg, 
                     #    optimizer=GradientDescent(),
                          optimizer=LBFGS(200; maxiter=100, verbosity=4, gradtol=1e-7, linesearch=HagerZhangLineSearch(maxfg=5)),
                          ifcheckpoint=false,
                          forloop_iter=1,
                          verbosity=4, 
                          folder=folder,
                          ifSU=false,
                          SUτ=SUτ,
                          ifprecondition=false,
                          iter_precond=0,
                          reuse_env=true, 
                          ifflatten=false,
                          ifsave_env=true,
                          ifload_env=true,
                          ifsave_lbfgs=true,
                          ifload_lbfgs=false
)
read_last_log(params, D)
A = init_ipeps(;atype, etype, No, d=2, pattern, D, χ, params)
# A = TeneT_demo.init_ipeps_to_D(;atype, No, D, D_new=3, params)

function restriction_ipeps(A)
#    A += permutedims(conj(A), (1,4,3,2,5,6)) # up-down
#    A += permutedims(conj(A), (3,2,1,4,5,6)) # left-right
#    A += permutedims(conj(A), (2,1,4,3,5,6)) # diagonal
#    A += permutedims(conj(A), (4,3,2,1,5,6)) # rotation

#    Ar = Zygote.Buffer(A)
#    for i in 1:length(A)
#    for j in 1:Nj, i in 1:Ni
#        if (i,j) in [(2,1)]
#            Ar[i,j] = A[i,j] + permutedims(conj(A[i,j]), (1,4,3,2,5))
#        elseif (i,j) in [(3,1)]
#            Ar[i,j] = permutedims(conj(A[1,1]), (1,4,3,2,5))
#        else
#            Ar[i,j] = A[i,j]
#        end
#    end
#    Ar = copy(Ar)
#    return Ar/norm(Ar)
#    λ = Zygote.@ignore norm(A)
    # A = TeneT_demo._restriction_ipeps(A)
    # A = TeneT_demo.central_canonical1(A)
    # A = TeneT_demo.pepsgeneral(A)[1]
#    A = TeneT_demo.rand_gauge(A)
    A /= norm(A)
    # if ifMCF
        A = TeneT_demo.local_min_norm(A, params)
    # end

    # A = TeneT_demo.local_hermite(A, params)
   return A
end

# ProfileView.@profview optimise_ipeps(A, χ, χshift, params;
#                restriction_ipeps
# );
# optimise_ipeps(A, χ, χshift, params; restriction_ipeps);
# es = []
# for χ in 20:10:20
    e, ξ = observable(A, 120, params; restriction_ipeps)
    # push!(es, real(e))
# end
# @show es
