using Random
using Test
using TeneT
using TeneT_demo
using Optim
using CUDA

@testset "optimise_ipeps" for Ni = [2], Nj = [2], D in [2], χ in [10]
    model = J1J2(Ni,Nj,1.0,0.5)
    A, key = init_ipeps(model; atype = CuArray, 
                        Ni=Ni, Nj=Nj, D=D, χ=χ, 
                        maxiter = 50,
                        verbose= true
                        )
    # spin(A, key; savefile = true)
    optimise_ipeps(A, key; maxiter_ad=10, miniter_ad=3, 
                   f_tol = 1e-10, opiter = 0, 
                   optimmethod = LBFGS(m = 20)
                   ) 
end