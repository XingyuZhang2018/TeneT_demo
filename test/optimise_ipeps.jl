using Random
using Test
using TeneT
using TeneT_demo
using TeneT_demo: init_ipeps, energy, optcont
using Optim
using CUDA

@testset "init_ipeps" for Ni = [1,2], Nj = [1,2], D in [2,3], χ in [10]
    model = Heisenberg(Ni,Nj)
    A, key = init_ipeps(model; Ni=Ni, Nj=Nj, D=D, χ=χ);
    @test size(A) == (D,D,D,D,2,Ni,Nj)
end

@testset "energy" for Ni = [1], Nj = [1], χ in [10]
    D = [1,2,1,2]
    model = Heisenberg(Ni,Nj)
    A, key = init_ipeps(model; Ni=Ni, Nj=Nj, D=D, χ=χ)
    oc = optcont(D[1], χ)
    h = hamiltonian(model)
    @show energy(h, A, oc, key; verbose = true, savefile = true)
end

@testset "optimise_ipeps" for Ni = [1], Nj = [1], χ in [20]
    D = [3,3,3,3]
    model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
    A, key = init_ipeps(model; atype = CuArray, Ni=Ni, Nj=Nj, D=D, χ=χ, verbose= false)
    optimise_ipeps(A, key; f_tol = 1e-10, opiter = 1000, optimmethod = LBFGS(m = 20))
end

@testset "optimise_ipeps" for Ni = [2], Nj = [2], D in [2], χ in [10]
    model = Heisenberg(Ni,Nj,1.0,1.0,1.0)
    A, key = init_ipeps(model; Ni=Ni, Nj=Nj, D=D, χ=χ, verbose= false)
    optimise_ipeps(A, key; f_tol = 1e-6, opiter = 10, optimmethod = LBFGS(m = 20))
end