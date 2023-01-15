using Random
using Test
using TeneT
using TeneT_demo
using TeneT_demo: init_ipeps, spin
using Optim
using CUDA

@testset "optimise_ipeps" for Ni = [1], Nj = [1], χ in [80]
    D = [3,3,3,3]
    model = TFIsing(Ni,Nj,3.04438)
    A, key = init_ipeps(model; atype = CuArray, Ni=Ni, Nj=Nj, D=D, χ=χ, verbose= true)
    # spin(A, key; savefile = true)
    optimise_ipeps(A, key; f_tol = 1e-10, opiter = 10000, optimmethod = LBFGS(m = 20))
end