using Random
using Test
using TeneT
using TeneT_demo
using TeneT_demo: init_ipeps, spin
using Optim
using CUDA

@testset "optimise_ipeps" for Ni = [1], Nj = [1], χ in [10]
    D = [2,2,2,2]
    model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
    A, key = init_ipeps(model; atype = CuArray, 
                        Ni=Ni, Nj=Nj, D=D, χ=χ, 
                        maxiter = 50,
                        verbose= true
                        )
    # spin(A, key; savefile = true)
    optimise_ipeps(A, key; 
                   ifprecondition = true,
                   maxiter_ad=10, miniter_ad=3, 
                   f_tol = 1e-10, opiter = 200, 
                   optimmethod = LBFGS(m = 20)
                   ) 
end