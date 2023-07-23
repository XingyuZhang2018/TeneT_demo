using Random
using Test
using TeneT
using TeneT_demo
using TeneT_demo: init_ipeps, spin
using Optim
using CUDA

@testset "optimise_ipeps" for Ni = [1], Nj = [1], χ in [100]
    D = [4,4,4,4]
    model = TFIsing(Ni,Nj,3.04438)
    A, key = init_ipeps(model; atype = CuArray, 
                        Ni=Ni, Nj=Nj, D=D, χ=χ, 
                        maxiter = 50,
                        verbose= true
                        )
    # spin(A, key; savefile = true)
    optimise_ipeps(A, key; maxiter_ad=10, miniter_ad=3, 
                   f_tol = 1e-10, opiter = 200, 
                   optimmethod = LBFGS(m = 20)
                   ) 
end